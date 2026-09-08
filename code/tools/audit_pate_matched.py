"""Recompute matched PATE metrics, pairing and the stated Gaussian RDP bound."""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True)
    ap.add_argument('--bundle',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();p=json.loads((args.results/'protocol.json').read_text())
    orders=np.arange(2.,257.)
    bounds=p['queries']*orders/p['noise_sigma']**2+np.log(1/p['delta'])/(orders-1)
    epsilon=float(bounds.min());assert abs(epsilon-p['target_epsilon'])<1e-9
    rows=[]
    for seed in range(3):
        bundle=np.load(args.bundle/f'seed{seed}.npz');queries=np.load(args.results/f'seed{seed}_noisy_queries.npz')
        assert np.array_equal(queries['noisy_labels'],queries['noisy_counts'].argmax(1))
        assert queries['noisy_counts'].shape==(120,4) and np.isfinite(queries['noisy_counts']).all()
        initial=set()
        for mode in ['hard','pate','pate_plus_labels']:
            r=json.loads((args.results/f'seed{seed}_{mode}.json').read_text())
            pred=np.load(args.results/f'seed{seed}_{mode}_predictions.npz')['test'].argmax(1)
            y=bundle['ytest'];c=confusion_matrix(y,pred,labels=[0,1,2,3])
            m=dict(bal_acc=float(balanced_accuracy_score(y,pred)),macro_f1=float(f1_score(y,pred,average='macro')),
                   maj_pred=float(np.bincount(pred,minlength=4).max()/len(pred)))
            assert np.array_equal(c,r['student_test']['confusion'])
            for name,v in m.items():assert abs(v-r['student_test'][name])<1e-12
            if mode=='hard':assert np.array_equal(c,r['hard_reference']['confusion'])
            assert 1<=r['best_epoch']<=5;initial.add(r['initial_state'])
            rows.append(dict(seed=seed,mode=mode,**m))
        assert len(initial)==1
    result=dict(student_models=9,epsilon=epsilon,noise_sigma=p['noise_sigma'],delta=p['delta'],
        pairing_and_raw_prediction_metrics_match=True,hard_reproduces_strict_vctk=True,
        models={mode:{metric:dict(mean=float(np.mean([r[metric] for r in rows if r['mode']==mode])),
            sd=float(np.std([r[metric] for r in rows if r['mode']==mode],ddof=1)))
            for metric in ['bal_acc','macro_f1','maj_pred']}
            for mode in ['hard','pate','pate_plus_labels']},rows=rows,
        budget_interpretation='matched reported epsilon upper bounds; PATE Gaussian RDP and DP-SGD PRV differ')
    args.out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
