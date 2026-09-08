"""Independent prediction/accounting audit and prespecified CV17 comparisons."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from opacus.accountants import PRVAccountant
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from scipy.special import softmax, ndtr


def health_samples(c):
    diag=np.diagonal(c,axis1=-2,axis2=-1)
    row=c.sum(-1);col=c.sum(-2)
    b=np.divide(diag,row,out=np.zeros_like(diag,dtype=float),where=row>0).mean(-1)
    f=np.divide(2*diag,row+col,out=np.zeros_like(diag,dtype=float),where=row+col>0).mean(-1)
    return (b+f+1-col.max(-1)/c.sum(axis=(-2,-1)))/3


def verify_uncertainty(y,logits,groups,seed,record):
    pred=logits.argmax(1)
    speakers=sorted(set(groups))
    conf=np.array([confusion_matrix(y[groups==g],pred[groups==g],labels=[0,1,2]) for g in speakers])
    rng=np.random.default_rng(seed)
    multiplicity=rng.multinomial(len(conf),np.full(len(conf),1/len(conf)),size=2000)
    h=health_samples((multiplicity@conf.reshape(len(conf),9)).reshape(-1,3,3))
    cluster=record['health']['cluster']
    assert cluster['clusters']==len(conf) and cluster['draws']==2000
    assert abs(h.std(ddof=1)-cluster['se'])<1e-12
    assert np.allclose(np.quantile(h,[.025,.975]),cluster['interval95'],rtol=0,atol=1e-12)
    total=conf.sum(0);rng=np.random.default_rng(seed)
    draws=rng.multinomial(len(y),total.ravel()/len(y),size=2000).reshape(-1,3,3)
    iid=health_samples(draws).std(ddof=1)
    assert abs(iid-record['health']['iid_se'])<1e-12
    point=health_samples(total)
    for sd,name in [(cluster['se'],'rc_route'),(iid,'iid_rc_route')]:
        cdf=ndtr((np.array([.45,.60])-point)/max(sd,1e-12))
        masses={'hard':cdf[0],'hakd':cdf[1]-cdf[0],'kd':1-cdf[1]}
        chosen=max(masses,key=masses.get);certified=masses[chosen]>=.9
        route=record[name]
        assert route['mode']==(chosen if certified else 'hard')
        assert route['certified']==certified and route['fallback']==(not certified)
        assert all(abs(route['masses'][k]-v)<1e-12 for k,v in masses.items())


def score(y,logits):
    pred=logits.argmax(1)
    c=confusion_matrix(y,pred,labels=[0,1,2])
    return dict(bal_acc=float(balanced_accuracy_score(y,pred)),
        macro_f1=float(f1_score(y,pred,labels=[0,1,2],average='macro',zero_division=0)),
        maj_pred=float(np.bincount(pred,minlength=3).max()/len(pred)),confusion=c.tolist(),
        per_class_recall=(np.diag(c)/c.sum(1)).tolist())


def region(h):return 'hard' if h<.45 else 'hakd' if h<.60 else 'kd'


def interval(values):
    values=np.asarray(values,dtype=float)
    rng=np.random.default_rng(20260908)
    samples=values[rng.integers(0,len(values),size=(20000,len(values)))].mean(1)
    return dict(mean=float(values.mean()),sd=float(values.std(ddof=1)),
                bootstrap95=np.quantile(samples,[.025,.975]).tolist(),n_seeds=len(values))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--bundle',type=Path,required=True)
    ap.add_argument('--results',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    p=json.loads((args.results/'protocol.json').read_text())
    labels={s:np.load(args.bundle/f'{s}_labels.npy') for s in ['train','selection','calibration','test']}
    features={s:np.load(args.bundle/f'{s}_ssl.npy',mmap_mode='r') for s in labels}
    frames={s:pd.read_csv(args.bundle/f'{s}_manifest.csv') for s in labels}
    source=json.loads((args.bundle/'source_manifest.json').read_text())
    assert source['sampling_revision']=='quote_none_20260908'
    assert {s:len(f) for s,f in frames.items()}=={'train':50000,'selection':735,'calibration':723,'test':1675}
    assert frames['train'].groupby('client_id').size().max()<=20
    for a in frames:
        assert frames[a].path.is_unique
        assert (frames[a].label.to_numpy()==labels[a]).all()
        for b in frames:
            if a!=b:
                assert set(frames[a].path).isdisjoint(frames[b].path)
                assert set(frames[a].client_id).isdisjoint(frames[b].client_id)
    rows=[];teacher_count=student_count=0;eps_error=0.;baselines=[];split_counts=[]
    for seed in p['split_seeds']:
        root=args.results/f'seed{seed}'
        assert (root/'complete.json').is_file(), root
        sp=json.loads((root/'split.json').read_text())
        priv=np.asarray(sp['private']);aux=np.asarray(sp['auxiliary'])
        assert sorted(np.r_[priv,aux].tolist())==list(range(len(frames['train'])))
        assert set(frames['train'].iloc[priv].client_id).isdisjoint(frames['train'].iloc[aux].client_id)
        split_counts.append(dict(seed=seed,private=len(priv),auxiliary=len(aux),
            private_counts=np.bincount(labels['train'][priv],minlength=3).tolist(),
            auxiliary_counts=np.bincount(labels['train'][aux],minlength=3).tolist()))
        scaler=StandardScaler().fit(features['train'][aux])
        saved=np.load(root/'normalization.npz')
        assert np.allclose(saved['mean'],scaler.mean_,rtol=0,atol=1e-12)
        assert np.allclose(saved['scale'],scaler.scale_,rtol=0,atol=1e-12)
        normalized={}
        for split in ['selection','calibration','test']:
            x=scaler.transform(features[split]).astype(np.float32)
            normalized[split]=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
        init=None;cached={}
        for f in sorted(root.glob('*.json')):
            r=json.loads(f.read_text())
            if r.get('kind') not in ['student','teacher']:continue
            pred=np.load(root/f'{f.stem}_predictions.npz')
            state=torch.load(root/f'{f.stem}.pt',map_location='cpu',weights_only=True)
            for split in ['selection','calibration','test']:
                assert pred[split].shape==(len(labels[split]),3) and np.isfinite(pred[split]).all()
                calculated=normalized[split]@state['fc.weight'].numpy().T+state['fc.bias'].numpy()
                assert np.allclose(calculated,pred[split],rtol=1e-5,atol=1e-4),(f,split,'checkpoint logits')
                m=score(labels[split],pred[split])
                for key,value in m.items():
                    assert np.allclose(value,r['metrics'][split][key],rtol=0,atol=1e-10),(f,split,key)
            if r['kind']=='teacher':
                teacher_count+=1
                info=r['info'];acct=info['accountant']
                assert len(info['trace'])==p['teacher_epochs']
                best=max(info['trace'],key=lambda z:z['selection_macro_f1'])
                assert best['epoch']==info['best_epoch']
                assert info['class_counts']==np.bincount(labels['train'][aux],minlength=3).tolist()
                if acct:
                    assert acct['dataset_size']==len(priv)
                    n_batches=int(np.ceil(len(priv)/p['batch_size']))
                    assert acct['steps']==p['teacher_epochs']*n_batches
                    assert abs(acct['sample_rate']-1/n_batches)<1e-12
                    assert len(acct['history'])==1
                    assert np.allclose(acct['history'][0],[info['sigma'],1/n_batches,acct['steps']])
                    accountant=PRVAccountant();accountant.history=[tuple(t) for t in acct['history']]
                    err=abs(accountant.get_epsilon(p['delta'])-acct['epsilon']);eps_error=max(eps_error,err)
                    assert err<1e-6
                m=r['metrics']['calibration']; h=(m['bal_acc']+m['macro_f1']+1-m['maj_pred'])/3
                assert abs(h-r['health']['global_health'])<1e-10
                assert np.allclose(r['health']['class_health'],m['per_class_recall'],rtol=0,atol=1e-7)
                assert region(h)==r['route']
                verify_uncertainty(labels['calibration'],pred['calibration'],
                                   frames['calibration'].client_id.to_numpy(),seed+7919,r)
                baselines.append(dict(seed=seed,sigma=info['sigma'],loss=info['loss'],
                    epsilon=None if not acct else acct['epsilon'],health=h,
                    **{k:r['metrics']['test'][k] for k in ['bal_acc','macro_f1']}))
            else:
                student_count+=1
                assert 1<=r['info']['best_epoch']<=p['student_epochs']
                trace=r['info']['selection_trace']
                assert len(trace)==p['student_epochs']
                best=max(trace,key=lambda v:v['macro_f1'])
                assert best['epoch']==r['info']['best_epoch']
                assert abs(best['macro_f1']-r['metrics']['selection']['macro_f1'])<1e-10
                assert r['info']['order_seed']==seed+1000004
                if init is None:init=r['info']['initial_state']
                assert r['info']['initial_state']==init
            cached[f.stem]=r
        for sigma in p['noise_multipliers']:
            tid=f'teacher_sigma{sigma:g}_balanced_softmax';t=cached[tid]
            candidates={'hard':cached['hard']}
            for mode in ['kd','hakd','ctkd_global','kd_meanmatched']:
                candidates[mode]=cached[f'{tid}_{mode}_instance']
            for weight in ['class','confidence']:
                candidates[f'hakd_{weight}']=cached[f'{tid}_hakd_{weight}']
            for name,candidate in candidates.items():
                if name!='hard':
                    assert candidate['teacher_id']==tid
                    assert np.allclose(candidate['info']['class_health'],t['health']['class_health'],atol=1e-7)
            ctkd=candidates['ctkd_global']['info']['ctkd']
            assert ctkd['source_commit']=='56112892d5aca069bd56bb057feee9f7ff9e4141'
            assert any(abs(v['raw_gradient'])>1e-10 for v in ctkd['trace'])
            aux_logits=np.load(root/f'{tid}_predictions.npz')['auxiliary']
            expected_alpha=p['kd_alpha']*np.mean(softmax(aux_logits/p['kd_temperature'],axis=1)
                                                @np.asarray(t['health']['class_health']))
            assert abs(expected_alpha-candidates['kd_meanmatched']['info']['configured_kd_alpha'])<1e-7
            cal=t['metrics']['calibration']; h=t['health']['global_health']
            choose_student=max(['hard','kd','hakd'],key=lambda n:candidates[n]['metrics']['selection']['bal_acc'])
            row=dict(seed=seed,sigma=sigma,health=h,bal_acc=cal['bal_acc'],macro_f1=cal['macro_f1'],
                dispersion=1-cal['maj_pred'],epsilon=t['info']['accountant']['epsilon'],route=t['route'],
                rc_mode=t['rc_route']['mode'],rc_certified=t['rc_route']['certified'],
                iid_rc_mode=t['iid_rc_route']['mode'],s_bacc_mode=choose_student,
                cluster_se=t['health']['cluster']['se'],iid_se=t['health']['iid_se'],
                teacher_test_bal_acc=t['metrics']['test']['bal_acc'])
            tm=t['metrics']['test'];row['test_health']=(tm['bal_acc']+tm['macro_f1']+1-tm['maj_pred'])/3
            row['test_health_region']=region(row['test_health'])
            lo,hi=t['health']['cluster']['interval95']
            row['cluster_interval_contains_test_estimate']=lo<=row['test_health']<=hi
            for name,r in candidates.items():
                row[name]=r['metrics']['test']['bal_acc'];row[name+'_mf1']=r['metrics']['test']['macro_f1']
            row['kd_gain']=row['kd']-row['hard']
            row['hakd_gain_hard']=row['hakd']-row['hard'];row['hakd_gain_kd']=row['hakd']-row['kd']
            row['tcrd']=row[t['route']];row['rc_tcrd']=row[t['rc_route']['mode']]
            row['t_bacc']=row['hard' if cal['bal_acc']<.45 else 'kd']
            row['s_bacc']=row[choose_student]
            row['tcrd_mf1']=row[t['route']+'_mf1'];row['rc_tcrd_mf1']=row[t['rc_route']['mode']+'_mf1']
            row['t_bacc_mf1']=row[('hard' if cal['bal_acc']<.45 else 'kd')+'_mf1']
            row['s_bacc_mf1']=row[choose_student+'_mf1']
            comps=np.array([row['bal_acc'],row['macro_f1'],row['dispersion']])
            for weights in p['components']:
                key='w_'+'_'.join(map(str,weights));ch=float(comps@np.array(weights)/sum(weights))
                row[key+'_mode']=region(ch);row[key]=row[region(ch)]
                row[key+'_mf1']=row[region(ch)+'_mf1']
            rows.append(row)
    assert teacher_count==80 and student_count==310,(teacher_count,student_count)
    table=pd.DataFrame(rows);table.to_csv(args.out/'paired_rows.csv',index=False)
    pd.DataFrame(baselines).to_csv(args.out/'teacher_rows.csv',index=False)
    pd.DataFrame(split_counts).to_csv(args.out/'split_counts.csv',index=False)
    methods=['hard','kd','hakd','ctkd_global','hakd_class','hakd_confidence','kd_meanmatched','tcrd','rc_tcrd','t_bacc','s_bacc']
    table.groupby('sigma')[['health','epsilon','teacher_test_bal_acc']+methods+[m+'_mf1' for m in methods]].agg(['mean','std']).to_csv(args.out/'summary.csv')
    contrasts={}
    for a,b in [('kd','hard'),('hakd','hard'),('hakd','kd'),('tcrd','t_bacc'),('tcrd','hard'),
                ('rc_tcrd','tcrd'),('tcrd','s_bacc'),('tcrd','w_1_0_0'),('hakd','kd_meanmatched')]:
        name=a+'-minus-'+b; difference=table[a]-table[b]
        contrasts[name]={'equal_noise_weight_seed_summary':interval(difference.groupby(table.seed).mean()),
                        'per_sigma':{str(s):interval(difference[table.sigma==s]) for s in p['noise_multipliers']}}
        mf1_difference=table[a+'_mf1']-table[b+'_mf1']
        contrasts[name]['macro_f1']={'equal_noise_weight_seed_summary':interval(mf1_difference.groupby(table.seed).mean()),
                        'per_sigma':{str(s):interval(mf1_difference[table.sigma==s]) for s in p['noise_multipliers']}}
    predictive=[]
    for features in [['bal_acc'],['health'],['bal_acc','macro_f1','dispersion']]:
        predicted=np.empty(len(table))
        for seed in p['split_seeds']:
            test=table.seed==seed;train=~test
            model=make_pipeline(StandardScaler(),Ridge(alpha=1.))
            model.fit(table.loc[train,features],table.loc[train,'kd_gain'])
            predicted[test]=model.predict(table.loc[test,features])
        key='+'.join(features);error=(predicted-table.kd_gain.to_numpy())**2
        table[key+'_prediction']=predicted;table[key+'_squared_error']=error
        predictive.append(dict(features=features,mse=float(error.mean())))
    delta=table['bal_acc+macro_f1+dispersion_squared_error']-table['bal_acc_squared_error']
    incremental=interval(delta.groupby(table.seed).mean())
    hdelta=table['health_squared_error']-table['bal_acc_squared_error']
    table.to_csv(args.out/'paired_rows.csv',index=False)
    correlations={name:dict(pearson=float(pearsonr(table[name],table.kd_gain).statistic),
                           spearman=float(spearmanr(table[name],table.kd_gain).statistic))
                  for name in ['health','bal_acc','macro_f1','dispersion']}
    result=dict(teachers=teacher_count,students=student_count,independent_split_seeds=10,
        noise_conditions_per_seed=5,records_are_50_independent_partitions=False,max_epsilon_error=eps_error,
        disjoint_speaker_and_clip_sets=True,predictions_and_accounting_match=True,
        auxiliary_only_normalization_and_checkpoints_verified=True,
        calibration_resampling_and_region_certificates_recomputed=True,
        route_counts=table.route.value_counts().to_dict(),contrasts=contrasts,
        predictive_loso=predictive,all_components_minus_bacc_mse=incremental,
        composite_minus_bacc_mse=interval(hdelta.groupby(table.seed).mean()),
        descriptive_correlations=correlations,
        hakd_better_than_both_seed_conditions=int(((table.hakd>table.hard)&(table.hakd>table.kd)).sum()),
        rc_iid_cluster_disagreements=int((table.rc_mode!=table.iid_rc_mode).sum()),
        finite_test_health_proxy=dict(
            deterministic_disagreements=int((table.route!=table.test_health_region).sum()),
            rc_disagreements=int((table.rc_mode!=table.test_health_region).sum()),
            rc_certified=int(table.rc_certified.sum()),
            rc_certified_disagreements=int(((table.rc_mode!=table.test_health_region)&table.rc_certified).sum()),
            interval_contains_test_estimate=int(table.cluster_interval_contains_test_estimate.sum()),
            comparisons=50,population_coverage_claim=False),
        positive_kd_rejected_conditions=int(((table.kd_gain>0)&(table.route=='hard')).sum()),
        uncertainty_interpretation='seed bootstrap conditional on fixed corpus/evaluation split; no utility guarantee')
    (args.out/'evidence.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
