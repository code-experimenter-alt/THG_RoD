"""Matched VCTK PATE with stable shards, saved noisy queries and student logits."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.optimize import brentq
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.main import RunCfg, _state_dict_sha256
from src.dataset import build_features_and_splits, set_seed, normalize_accent_3way
from src.models import SmallCNNMel
from src.metrics import eval_all
from tools.run_pate_vctk import gaussian_rdp_epsilon


def shard_for(identifier):
    # Inserting/removing a record never reassigns another record.
    return int.from_bytes(hashlib.sha256(('pate-20260908:'+identifier).encode()).digest()[:8],'big')%3


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path)
    ap.add_argument('--bundle',type=Path,required=True)
    ap.add_argument('--out',type=Path)
    ap.add_argument('--prepare',action='store_true')
    ap.add_argument('--data-root',type=Path,help='Relocated VCTK manifests for --prepare')
    args=ap.parse_args();torch.set_num_threads(4)
    args.bundle.mkdir(parents=True,exist_ok=True)
    if args.prepare:
        for seed in range(3):
            original=args.source/f'CV_MEL_BS_WEAK_RELEASE_seed{seed}_sigma1_tagstrictpair_bs_s{seed}_hard.json'
            record=json.loads(original.read_text());cfg=RunCfg(**record['cfg'])
            if args.data_root:
                cfg.data_root=str(args.data_root.resolve())
                cfg.cache_dir=str((args.bundle/'features').resolve())
            feat,ys,names=build_features_and_splits(cfg,torch.device('cpu'))
            xpriv,xaux,xdev,xtest=feat['mel'];ypriv,yaux,ydev,ytest=ys
            frame=pd.read_csv(Path(cfg.data_root)/cfg.train_csv)
            priv,aux=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=seed).split(frame,groups=frame.speaker_id))
            encoded=frame.accent.map(lambda s:normalize_accent_3way(s) or str(s).strip().lower()).map(
                {name:i for i,name in enumerate(names)}).to_numpy()
            assert np.array_equal(encoded[priv],ypriv) and np.array_equal(encoded[aux],yaux)
            assert xpriv.shape==(360,32,80) and xaux.shape==(120,32,80)
            id_column='record_id' if 'record_id' in frame else 'filename'
            ids=frame.iloc[priv][id_column].astype(str).to_numpy(dtype=str)
            np.savez_compressed(args.bundle/f'seed{seed}.npz',xpriv=xpriv,xaux=xaux,xdev=xdev,xtest=xtest,
                ypriv=ypriv,yaux=yaux,ydev=ydev,ytest=ytest,private_ids=ids)
            (args.bundle/f'seed{seed}_reference.json').write_text(json.dumps(dict(class_names=names,
                original_hard=record['student_test'],student_init=record['student_info']['student_init_hash'])))
        print('PATE_BUNDLE_COMPLETE',flush=True);return
    args.out.mkdir(parents=True,exist_ok=True)
    target=6.0829038771;delta=1e-5
    sigma=brentq(lambda s:gaussian_rdp_epsilon(120,s,delta)-target,10.,20.,xtol=1e-12)
    protocol=dict(target_epsilon=target,epsilon=gaussian_rdp_epsilon(120,sigma,delta),noise_sigma=sigma,delta=delta,
        queries=120,teachers=3,sharding='stable per-record identifier hash, independent of dataset length',
        adjacency='add_remove_one_private_record, with private/aux/dev/test assignment fixed',
        count_vector_l2_sensitivity=float(np.sqrt(2)),accounting='Gaussian RDP over 120 queries, integer orders2..256',
        teacher_epochs=4,student_epochs=5,n_mels=32,max_time=80,dsaf=True,
        student_modes={'hard':0.,'pate':1.,'pate_plus_labels':.7},
        model='SmallCNNMel with all4 public classes',optimizer='AdamW lr0.001 wd0.0001 batch64',
        checkpoint='maximum development Macro-F1, earliest ties',
        release_boundary='student checkpoints and noisy counts; no raw teacher weights/votes/development statistics')
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    device=torch.device('cuda')
    for seed in range(3):
        z=dict(np.load(args.bundle/f'seed{seed}.npz'))
        reference=json.loads((args.bundle/f'seed{seed}_reference.json').read_text())
        dev=DataLoader(TensorDataset(torch.tensor(z['xdev']),torch.tensor(z['ydev'])),batch_size=64)
        test=DataLoader(TensorDataset(torch.tensor(z['xtest']),torch.tensor(z['ytest'])),batch_size=64)
        assignment=np.array([shard_for(s) for s in z['private_ids']])
        def train(x,y,pseudo,alpha,epochs,initial_seed,order_seed):
            set_seed(initial_seed);model=SmallCNNMel(32,80,4).to(device);initial=_state_dict_sha256(model)
            loader=DataLoader(TensorDataset(torch.tensor(x),torch.tensor(y),torch.tensor(pseudo)),
                batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(order_seed))
            opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
            best=-1.;state=None;best_epoch=-1
            for ep in range(epochs):
                model.train()
                for xb,yb,pb in loader:
                    logits=model(xb.to(device));yb,pb=yb.to(device),pb.to(device)
                    loss=(1-alpha)*torch.nn.functional.cross_entropy(logits,yb)+alpha*torch.nn.functional.cross_entropy(logits,pb)
                    opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                value=eval_all(model,dev,device,4)['macro_f1']
                if value>best:
                    best=value;best_epoch=ep+1;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            model.load_state_dict(state)
            return model,initial,best_epoch
        votes=[]
        for shard in range(3):
            ix=np.flatnonzero(assignment==shard);assert len(ix)>0
            teacher,_,_=train(z['xpriv'][ix],z['ypriv'][ix],z['ypriv'][ix],0.,4,seed*100+shard,seed*100+shard+10000)
            teacher.eval()
            with torch.no_grad():votes.append(teacher(torch.tensor(z['xaux'],device=device)).argmax(1).cpu().numpy())
            del teacher
        counts=np.eye(4)[np.stack(votes)].sum(0)
        noisy=counts+np.random.default_rng(seed+9000).normal(0,sigma,size=counts.shape)
        pseudo=noisy.argmax(1).astype(np.int64)
        np.savez_compressed(args.out/f'seed{seed}_noisy_queries.npz',noisy_counts=noisy,noisy_labels=pseudo)
        for mode,alpha in protocol['student_modes'].items():
            student,initial,best_epoch=train(z['xaux'],z['yaux'],pseudo,alpha,5,seed+1000003,seed+1000004)
            assert initial==reference['student_init']
            metrics=eval_all(student,test,device,4)
            preds={}
            with torch.no_grad():
                for name in ['dev','test']:
                    preds[name]=student(torch.tensor(z['x'+name],device=device)).cpu().numpy()
            np.savez_compressed(args.out/f'seed{seed}_{mode}_predictions.npz',**preds)
            torch.save(student.state_dict(),args.out/f'seed{seed}_{mode}.pt')
            r=dict(seed=seed,mode=mode,alpha_noisy_label=alpha,initial_state=initial,best_epoch=best_epoch,
                student_test=metrics,hard_reference=reference['original_hard'] if mode=='hard' else None)
            (args.out/f'seed{seed}_{mode}.json').write_text(json.dumps(r,indent=2))
            print('PATE_MATCHED',seed,mode,metrics['bal_acc'],flush=True)
    print('PATE_MATCHED_COMPLETE',flush=True)

if __name__=='__main__':main()
