"""Render manuscript tables and a gain diagnostic from independently audited rows."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def pm(values):
    x=np.asarray(values,dtype=float)
    return f'${x.mean():.3f}\\pm{x.std(ddof=1):.3f}$'


def percent_sd(values):
    x=np.asarray(values,dtype=float)*100
    return f'{x.mean():.1f}({x.std(ddof=1):.1f})'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--audit',type=Path,required=True)
    ap.add_argument('--latex',type=Path,required=True)
    args=ap.parse_args()
    evidence=json.loads((args.audit/'evidence.json').read_text())
    assert evidence['teachers']==80 and evidence['students']==310
    rows=pd.read_csv(args.audit/'paired_rows.csv')
    teachers=pd.read_csv(args.audit/'teacher_rows.csv')
    sigma=sorted(rows.sigma.unique())
    assert len(rows)==50 and rows.seed.nunique()==10
    target=args.latex/'cv17_tables';target.mkdir(exist_ok=True)
    teacher_lines=[]
    for noise,loss,name in [(0.,'ce','Non-DP CE'),(0.,'balanced_softmax','Non-DP BS'),
                            (1.,'ce','DP CE, $\\sigma=1$')]+[(s,'balanced_softmax',f'DP BS, $\\sigma={s:g}$') for s in sigma]:
        t=teachers[(teachers.sigma==noise)&(teachers.loss==loss)]
        assert len(t)==10
        eps='--' if noise==0 else f"{t.epsilon.min():.3f}--{t.epsilon.max():.3f}"
        health=f'{t.health.min():.3f}--{t.health.max():.3f}'
        teacher_lines.append(' & '.join([name,eps,pm(t.bal_acc),pm(t.macro_f1),health])+r'\\')
    (target/'teachers.tex').write_text('\n'.join(teacher_lines)+'\n')
    compact=[]
    for noise,loss,name in [(0.,'ce','Non-DP CE'),(0.,'balanced_softmax','Non-DP BS'),
                            (1.,'ce','CE, $\\sigma=1$')]+[(s,'balanced_softmax',f'BS, $\\sigma={s:g}$') for s in sigma]:
        t=teachers[(teachers.sigma==noise)&(teachers.loss==loss)]
        eps='--' if noise==0 else f"{t.epsilon.min():.3f}--{t.epsilon.max():.3f}"
        health=f'{t.health.min():.3f}--{t.health.max():.3f}'
        compact.append(' & '.join([name,eps,percent_sd(t.bal_acc),percent_sd(t.macro_f1),health])+r'\\')
    (target/'teachers_compact.tex').write_text('\n'.join(compact)+'\n')
    methods=[('hard','Hard'),('kd','KD / uniform HAKD'),('hakd','HAKD, instance'),
             ('kd_meanmatched','Mean-matched KD'),('hakd_class','HAKD, class'),
             ('hakd_confidence','HAKD, confidence'),('ctkd_global','CTKD, global'),
             ('tcrd','TCRD'),('rc_tcrd','RC-TCRD'),('t_bacc','T-BAcc'),('s_bacc','S-BAcc')]
    for metric,suffix in [('balacc',''),('mf1','_mf1')]:
        lines=[]
        for method,name in methods:
            cells=[pm(rows.loc[rows.sigma==s,method+suffix]) for s in sigma]
            lines.append(' & '.join([name]+cells)+r'\\')
        (target/f'students_{metric}.tex').write_text('\n'.join(lines)+'\n')
    lines=[]
    for method,name in methods:
        cells=[]
        for s in sigma:
            r=rows.loc[rows.sigma==s]
            cells.append(percent_sd(r[method])+' / '+percent_sd(r[method+'_mf1']))
        lines.append(' & '.join([name]+cells)+r'\\')
    (target/'students_compact.tex').write_text('\n'.join(lines)+'\n')
    weights=[([1,1,1],'Equal'),([1,0,0],'$B$ only'),([0,1,0],'$F$ only'),([0,0,1],'$1-M$ only'),
             ([0,1,1],'Leave $B$ out'),([1,0,1],'Leave $F$ out'),([1,1,0],'Leave $1-M$ out'),
             ([2,1,1],'Biased toward $B$'),([1,2,1],'Biased toward $F$'),([1,1,2],'Biased toward $1-M$')]
    lines=[]
    for weight,name in weights:
        key='w_'+'_'.join(map(str,weight))
        counts=rows[key+'_mode'].value_counts()
        counts='/'.join(str(counts.get(m,0)) for m in ['hard','hakd','kd'])
        grouped=rows.groupby('seed')[[key,key+'_mf1']].mean()
        lines.append(' & '.join([name,counts,pm(grouped[key]),pm(grouped[key+'_mf1'])])+r'\\')
    (target/'components.tex').write_text('\n'.join(lines)+'\n')
    lines=[]
    for key,name in [('kd-minus-hard','KD $-$ Hard'),('hakd-minus-hard','HAKD $-$ Hard'),
                     ('hakd-minus-kd','HAKD $-$ KD'),('hakd-minus-kd_meanmatched','HAKD $-$ mean-matched'),
                     ('tcrd-minus-t_bacc','TCRD $-$ T-BAcc'),('tcrd-minus-hard','TCRD $-$ Hard'),
                     ('tcrd-minus-s_bacc','TCRD $-$ S-BAcc'),('tcrd-minus-w_1_0_0','TCRD $-$ three-way $B$'),
                     ('rc_tcrd-minus-tcrd','RC-TCRD $-$ TCRD')]:
        item=evidence['contrasts'][key]['equal_noise_weight_seed_summary']
        lo,hi=np.asarray(item['bootstrap95'])*100
        mean=item['mean']*100
        shown='0.00' if abs(mean)<.005 else f'{mean:+.2f}'
        lines.append(f"{name} & ${shown}$ & $[{lo:+.2f},{hi:+.2f}]$"+r'\\')
    (target/'contrasts.tex').write_text('\n'.join(lines)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(3.45,2.10))
    colors=plt.get_cmap('viridis')(np.linspace(.1,.9,len(sigma)))
    xmin=min(.42,float(rows.health.min())-.025);xmax=max(.63,float(rows.health.max())+.025)
    for lo,hi,color in [(xmin,.45,'#f2f2f2'),(.45,.60,'#e8f2fb'),(.60,xmax,'#e8f3e9')]:
        ax.axvspan(lo,hi,color=color,zorder=0)
    for s,color in zip(sigma,colors):
        r=rows[rows.sigma==s]
        ax.scatter(r.health,r.kd_gain*100,label=f'$\\sigma={s:g}$',s=24,color=color,edgecolor='white',linewidth=.45)
    ax.axhline(0,color='#333333',lw=.7)
    for value in [.45,.60]:ax.axvline(value,color='#666666',lw=.7,ls='--')
    ax.set(xlim=(xmin,xmax),xlabel='Calibration teacher health',ylabel='KD − Hard (BAcc points)')
    ax.spines[['top','right']].set_visible(False)
    ax.legend(frameon=False,fontsize=7,ncol=3,loc='best',handletextpad=.3,columnspacing=.5)
    fig.tight_layout(pad=.6)
    for ext in ['pdf','svg','png']:
        fig.savefig(args.latex/'figure'/f'cv17_health_gain.{ext}',dpi=220)
    plt.close(fig)
    print('AUDITED_TABLES_AND_GAIN_FIGURE_COMPLETE')


if __name__=='__main__':main()
