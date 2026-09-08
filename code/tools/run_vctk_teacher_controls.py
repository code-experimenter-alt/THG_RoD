"""Replace the old one-seed/eight-epoch VCTK non-private control."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.dataset import set_seed
from src.models import SmallCNNMel
from src.train_teacher import train_teacher
from src.metrics import eval_all
from src.main import _state_dict_sha256


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--bundle',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);device=torch.device('cuda');rows=[]
    protocol=dict(seeds=[0,1,2],epochs=4,private_rows=360,dev_rows=120,test_rows=120,
        architecture='SmallCNNMel 32x80 with DSAF features from strict VCTK bundle',
        optimizer='AdamW lr0.001 wd0.0001 batch64',losses=['ce','balanced_softmax'],
        counts='private, non-private controls; same fixed-count BS convention as strict DP teacher',
        checkpoint='maximum development Macro-F1, earliest ties',
        purpose='replace old single seed eight epoch contextual teacher row; no DP claim for these controls')
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    for seed in protocol['seeds']:
        z=dict(np.load(args.bundle/f'seed{seed}.npz'))
        dev=DataLoader(TensorDataset(torch.tensor(z['xdev']),torch.tensor(z['ydev'])),batch_size=64)
        test=DataLoader(TensorDataset(torch.tensor(z['xtest']),torch.tensor(z['ytest'])),batch_size=64)
        for loss in protocol['losses']:
            set_seed(seed);model=SmallCNNMel(32,80,4)
            initial=_state_dict_sha256(model)
            train=DataLoader(TensorDataset(torch.tensor(z['xpriv']),torch.tensor(z['ypriv'])),
                batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(seed+222222))
            model,info=train_teacher(model,train,dev,device,4,4,.001,.0001,False,0.,8.,1e-5,
                loss,np.bincount(z['ypriv'],minlength=4),record_grad_stats=False)
            metrics=eval_all(model,test,device,4)
            predictions={}
            with torch.no_grad():
                for split in ['dev','test']:
                    predictions[split]=model(torch.tensor(z['x'+split],device=device)).cpu().numpy()
            pred=predictions['test'].argmax(1);y=z['ytest']
            independent=dict(bal_acc=float(balanced_accuracy_score(y,pred)),
                macro_f1=float(f1_score(y,pred,average='macro')),
                maj_pred=float(np.bincount(pred,minlength=4).max()/len(pred)))
            assert np.array_equal(confusion_matrix(y,pred,labels=[0,1,2,3]),metrics['confusion'])
            assert all(abs(metrics[k]-v)<1e-12 for k,v in independent.items())
            name=f'seed{seed}_{loss}'
            torch.save(model.state_dict(),args.out/f'{name}.pt')
            np.savez_compressed(args.out/f'{name}_predictions.npz',**predictions)
            r=dict(seed=seed,loss=loss,initial_state=initial,info=info,test=metrics)
            (args.out/f'{name}.json').write_text(json.dumps(r,indent=2))
            rows.append(dict(seed=seed,loss=loss,**independent));print('CONTROL',rows[-1],flush=True)
    result=dict(teachers=6,raw_prediction_metrics_verified=True,rows=rows,
        controls={loss:{metric:dict(mean=float(np.mean([r[metric] for r in rows if r['loss']==loss])),
            sd=float(np.std([r[metric] for r in rows if r['loss']==loss],ddof=1)))
            for metric in ['bal_acc','macro_f1','maj_pred']} for loss in protocol['losses']})
    (args.out/'evidence.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
