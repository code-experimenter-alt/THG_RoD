"""Restore the old diagnostic views using the corrected, recorded run pools."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    return json.loads(path.read_text())


def save(fig, out, name):
    for ext in ['pdf', 'svg']:
        fig.savefig(out/f'{name}.{ext}', facecolor='white')
    plt.close(fig)


def diagnostic(records, names, out, name, clip):
    modes = ['teacher', 'hard', 'kd', 'hakd']
    labels = ['Teacher', 'Hard', 'KD', 'HAKD']
    colors = ['#64748b', '#355f86', '#d48230', '#23897d']
    fig, axs = plt.subplots(3, 2, figsize=(3.15, 2.95))
    axs = np.asarray(axs).reshape(2, 3)
    ax = axs[0, 0]
    for j, (metric, title) in enumerate([('bal_acc', 'BalAcc'), ('macro_f1', 'Macro-F1'), ('maj_pred', 'MajPred')]):
        a = np.array([[r[m][metric] for m in modes] for r in records])
        ax.bar(np.arange(4)+(j-1)*.24, a.mean(0), .23, yerr=a.std(0, ddof=1),
               capsize=1.4, error_kw={'elinewidth': .55}, label=title)
    ax.set(xticks=range(4), xticklabels=['T', 'H', 'K', 'A'], ylim=(0, 1.04))
    ax.set_title('(a) Test metrics', loc='left')
    fig.legend(*ax.get_legend_handles_labels(), fontsize=6, frameon=False, ncol=3,
               loc='lower center', bbox_to_anchor=(.5, -.008), handlelength=1, columnspacing=1)
    ax = axs[0, 1]
    gain = np.array([[r['kd'][k]-r['hard'][k] for k in ['bal_acc', 'macro_f1']] for r in records])*100
    for j, title in enumerate(['BalAcc', 'Macro-F1']):
        ax.plot(np.arange(len(records)), gain[:, j], marker=['o', 's'][j], ms=2.5, lw=.75, label=title)
    ax.axhline(0, color='gray', lw=.6)
    ax.set(xticks=range(len(records)), ylabel='Gain (pp)')
    ax.set_title('(b) Gain by seed', loc='left')
    ax.legend(fontsize=5.7, frameon=False, loc='best')
    ax = axs[0, 2]
    recall = np.mean([[r[m]['per_class_recall'] for m in modes] for r in records], axis=0)
    ax.pcolormesh(np.arange(len(names)+1)-.5, np.arange(5)-.5, recall,
                  vmin=0, vmax=1, cmap='YlGnBu', shading='flat', rasterized=False)
    ax.set(xlim=(-.5, len(names)-.5), ylim=(3.5, -.5))
    ax.set(yticks=range(4), yticklabels=['T', 'H', 'K', 'A'], xticks=range(len(names)), xticklabels=names)
    for i in range(4):
        for j in range(len(names)):
            ax.text(j, i, f'{recall[i,j]:.2f}', ha='center', va='center', fontsize=6,
                    color='white' if recall[i,j]>.6 else '#202020')
    ax.set_title('(c) Class recall', loc='left')
    ax = axs[1, 0]
    proportions = np.mean([[np.asarray(r[m]['pred_counts'])/sum(r[m]['pred_counts']) for m in modes] for r in records], axis=0)
    left = np.zeros(4)
    for j, label in enumerate(names):
        ax.barh(range(4), proportions[:, j], left=left, height=.72, label=label)
        left += proportions[:, j]
    ax.set(yticks=range(4), yticklabels=['T', 'H', 'K', 'A'], xlim=(0, 1), xticks=[0, .5, 1])
    ax.invert_yaxis()
    ax.set_title('(d) Prediction share', loc='left')
    ax.legend(fontsize=5.8, frameon=False, ncol=4, loc='upper center', bbox_to_anchor=(.5, -.23),
              handlelength=.7, columnspacing=.65, handletextpad=.2)
    ax = axs[1, 1]
    for i, r in enumerate(records):
        y = r['grad_stats']['clip_rate']
        ax.plot(np.arange(1, len(y)+1), y, marker='o', ms=2, lw=.65, label=f's{i}')
    ax.set(ylim=(.97, 1.005), yticks=[.98, 1.0], xticks=[1, 2, 3, 4], xlabel='Teacher epoch')
    ax.set_title('(e) Clipped fraction', loc='left')
    ax = axs[1, 2]
    for key, label, line in [('gnorm_p90', 'q90', '-'), ('gnorm_p99', 'q99', '--')]:
        a = np.array([r['grad_stats'][key] for r in records])/clip
        ax.plot(range(1, a.shape[1]+1), a.mean(0), line, lw=1, label=label)
        ax.fill_between(range(1, a.shape[1]+1), a.mean(0)-a.std(0, ddof=1), a.mean(0)+a.std(0, ddof=1), alpha=.15)
    ax.set(xticks=[1, 2, 3, 4], xlabel='Teacher epoch', ylabel='Norm / C')
    ax.set_title('(f) Gradient norm', loc='left')
    ax.legend(fontsize=5.7, frameon=False, loc='best')
    fig.subplots_adjust(left=.10, right=.96, bottom=.16, top=.94, wspace=.69, hspace=.90)
    save(fig, out, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', type=Path, required=True)
    ap.add_argument('--cv-evidence', type=Path, required=True)
    ap.add_argument('--manifests', type=Path, required=True)
    ap.add_argument('--latex', type=Path, required=True)
    args = ap.parse_args()
    out = args.latex/'figure'
    tables = args.latex/'restored_tables'
    tables.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':7, 'axes.titlesize':6.6,
        'axes.labelsize':6.8, 'xtick.labelsize':6.3, 'ytick.labelsize':6.3,
        'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none',
        'axes.spines.top':False, 'axes.spines.right':False, 'axes.linewidth':.6})
    vctk = []
    for seed in range(3):
        root = args.results/'strict_bs_pair'
        branches = {m:read(root/f'CV_MEL_BS_WEAK_RELEASE_seed{seed}_sigma1_tagstrictpair_bs_s{seed}_{m}.json') for m in ['hard', 'kd', 'hakd']}
        r = branches['kd']
        vctk.append(dict(teacher=r['teacher_test'], grad_stats=r['teacher_info']['grad_stats'],
                         **{m:d['student_test'] for m,d in branches.items()}))
    emotion = []
    for seed in range(5):
        r = read(args.results/'iemocap_disjoint_20260907'/f'iemocap_precomputed_seed{seed}.json')
        emotion.append(dict(teacher=r['teacher']['test'], grad_stats=r['teacher']['info']['grad_stats'],
                            **{m:d['test'] for m,d in r['branches'].items()}))
    diagnostic(vctk, ['Eng', 'Iri', 'Sco', 'US'], out, 'vctk_main_rich_figure', 8)
    diagnostic(emotion, ['0', '1', '2', '3'], out, 'ieo_iemocap_composite', 1)
    fig, axes = plt.subplots(1, 2, figsize=(3.15, 1.55))
    for ax, records, title in zip(axes, [vctk, emotion], ['VCTK (3 seeds)', 'IEMOCAP (5 seeds)']):
        for j, (metric, label) in enumerate([('bal_acc','BalAcc'),('macro_f1','Macro-F1')]):
            a=np.array([[r[m][metric] for m in ['teacher','hard','kd','hakd']] for r in records])
            ax.errorbar(np.arange(4)+(j-.5)*.12, a.mean(0), yerr=a.std(0,ddof=1), marker=['o','s'][j],
                        lw=.8, ms=2.5, capsize=1.5, label=label)
        ax.set(xticks=range(4), xticklabels=['T','H','K','A'], ylim=(0,.60))
        ax.set_title(title)
    axes[0].set_ylabel('Test utility')
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=2, frameon=False, fontsize=6,
               loc='lower center', bbox_to_anchor=(.5,-.02))
    fig.subplots_adjust(left=.14,right=.98,bottom=.25,top=.85,wspace=.4)
    save(fig,out,'vctk_iemocap_summary')
    cv = read(args.cv_evidence)
    assert cv['complete'] and len(cv['rows'])==30
    fig, axes = plt.subplots(1, 2, figsize=(3.15,1.6))
    for ax, metric, title in zip(axes,['bal_acc','maj_pred'],['Balanced accuracy','Majority share']):
        for m,label,color in [('hard','Hard / TCRD','#355f86'),('kd','KD','#d48230'),('hakd','HAKD','#23897d'),('ctkd_global','CTKD','#8659a6')]:
            groups=[[r['scores'][m][metric] for r in cv['rows'] if r['family']=='mel' and r['sigma']==s] for s in [1,1.5,2]]
            ax.errorbar([1,1.5,2],np.mean(groups,axis=1), yerr=np.std(groups,axis=1,ddof=1),
                        marker='o',ms=2.3,lw=.7,capsize=1.2,label=label,color=color)
        ax.set(xticks=[1,1.5,2],xlabel=r'Noise $\sigma$')
        ax.set_title(title)
    fig.legend(*axes[0].get_legend_handles_labels(),ncol=4,frameon=False,fontsize=5.7,
               loc='upper center',bbox_to_anchor=(.5,1.02),handlelength=1,columnspacing=.6)
    fig.subplots_adjust(left=.13,right=.98,bottom=.26,top=.71,wspace=.43)
    save(fig,out,'cv_weak_teacher_tcrd')
    # The data table replaces unrecoverable old splits with the fixed CV17 manifests.
    lines=[]
    for name,label in [('train','Training'),('selection','Selection'),('calibration','Calibration'),('test','Test')]:
        with (args.manifests/(name+'_manifest.csv')).open() as f:
            rows=list(csv.DictReader(f))
        counts=Counter(r['label'] for r in rows)
        cells=[label,str(len(rows)),str(counts['0']),str(counts['1']),str(counts['2']),str(len({r['client_id'] for r in rows}))]
        lines.append(' & '.join(cells)+r'\\')
    (tables/'data_stats.tex').write_text('\n'.join(lines)+'\n')
    # Keep the original routing diagnostic fields, with the current executed policies.
    lines=[]
    for family,label in [('ssl','SSL'),('mel','Mel')]:
        for mode,title in [('kd','KD'),('TCRD','TCRD'),('T-BAcc','T-BAcc'),('S-BAcc','S-BAcc'),('S-MF1','S-MF1'),('S-Maj','S-MajPred')]:
            group=[r for r in cv['rows'] if r['family']==family]
            cells=[label,title]
            for sigma in [1,1.5,2]:
                g=[r['scores'][mode] for r in group if r['sigma']==sigma]
                cells.append(f"${np.mean([r['macro_f1'] for r in g]):.3f}/{np.mean([r['maj_pred'] for r in g]):.3f}$")
            modes=[r['routes'][mode] if mode!='kd' else 'kd' for r in group]
            cells.append('/'.join(str(modes.count(m)) for m in ['hard','hakd','kd']))
            lines.append(' & '.join(cells)+r'\\')
        lines.append(r'\midrule')
    (tables/'routing_diagnostics.tex').write_text('\n'.join(lines[:-1])+'\n')
    lines=[]
    for sigma in [1,1.5,2]:
        g=[r for r in cv['rows'] if r['family']=='ssl' and r['sigma']==sigma]
        cells=[f'{sigma:g}',f'{np.mean([r["epsilon"] for r in g]):.3f}']
        cells += [f'{np.mean([r["scores"][m]["bal_acc"] for r in g]):.3f}' for m in ['hard','kd']]
        cells += [f'${100*np.mean([r["scores"]["kd"][k]-r["scores"]["hard"][k] for r in g]):+.2f}$' for k in ['bal_acc','macro_f1','maj_pred']]
        lines.append(' & '.join(cells)+r'\\')
    (tables/'privacy_gain.tex').write_text('\n'.join(lines)+'\n')
    lines=[]
    for name,rows in [('VCTK',vctk),('IEMOCAP',emotion)]:
        for mode,label in [('teacher','DP teacher'),('hard','Hard / TCRD'),('kd','KD'),('hakd','HAKD')]:
            cells=[name,label]
            for k in ['acc','bal_acc','macro_f1','maj_pred']:
                a=np.array([r[mode][k] for r in rows])
                cells.append(f'{100*a.mean():.1f}({100*a.std(ddof=1):.1f})')
            lines.append(' & '.join(cells)+r'\\')
    (tables/'cross_dataset.tex').write_text('\n'.join(lines)+'\n')
    print('Restored four vector diagnostic figures and four audited table fragments.')


if __name__=='__main__':
    main()
