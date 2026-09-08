"""Prespecified CV17 study; disjoint selection/calibration and paired students."""
import argparse
import contextlib
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from opacus import PrivacyEngine
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models import LinearHead
from src.metrics import compute_teacher_health, estimate_health_se
from src.train_student import train_student
from src.main import _state_dict_sha256
from src.routing import resolve_student_mode, certified_rc_tcrd_mode


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def from_confusion(c):
    c=np.asarray(c)
    tp=np.diagonal(c, axis1=-2, axis2=-1)
    row=c.sum(-1); col=c.sum(-2)
    recall=np.divide(tp,row,out=np.zeros_like(tp,dtype=float),where=row>0)
    f1=np.divide(2*tp,row+col,out=np.zeros_like(tp,dtype=float),where=(row+col)>0)
    n=c.sum(axis=(-2,-1))
    return recall.mean(-1),f1.mean(-1),1-col.max(-1)/n


def metrics(y, logits):
    pred=logits.argmax(-1)
    c=np.bincount(np.asarray(y)*3+pred, minlength=9).reshape(3,3)
    b,f,d=from_confusion(c)
    return dict(acc=float(np.trace(c)/c.sum()),bal_acc=float(b),macro_f1=float(f),
        maj_pred=float(1-d),confusion=c.tolist(),
        per_class_recall=(np.diag(c)/c.sum(1)).tolist())


def cluster_health(y, logits, groups, seed, draws=2000):
    _,g=np.unique(groups,return_inverse=True)
    conf=np.zeros((g.max()+1,3,3),dtype=np.int64)
    np.add.at(conf,(g,y,logits.argmax(-1)),1)
    rng=np.random.default_rng(seed)
    counts=rng.multinomial(len(conf),np.ones(len(conf))/len(conf),size=draws)
    sample=np.einsum('bs,sij->bij',counts,conf)
    health=np.stack(from_confusion(sample),axis=-1).mean(-1)
    return dict(se=float(health.std(ddof=1)),interval95=np.quantile(health,[.025,.975]).tolist(),
                clusters=len(conf),draws=draws)


@torch.no_grad()
def predict(model,x):
    model.eval()
    return np.concatenate([model(t.cuda()).cpu().numpy() for t in x.split(1024)])


def teacher_train(x,y,selection_x,selection_y,counts,protocol,seed,sigma,loss_name):
    seed_all(seed)
    model=LinearHead(768,3).cuda()
    initial=_state_dict_sha256(model)
    train=DataLoader(TensorDataset(x,y), batch_size=protocol['batch_size'],shuffle=True,
                     generator=torch.Generator().manual_seed(seed+222222))
    opt=torch.optim.AdamW(model.parameters(),lr=protocol['learning_rate'],weight_decay=protocol['weight_decay'])
    engine=None
    if sigma>0:
        engine=PrivacyEngine(accountant='prv')
        model,opt,train=engine.make_private(module=model,optimizer=opt,data_loader=train,
            noise_multiplier=sigma,max_grad_norm=protocol['max_grad_norm'])
    adjust=torch.tensor(np.log(counts),dtype=torch.float32,device='cuda')
    best=-1.; state=None; trace=[]; selected_epoch=-1
    t0=time.perf_counter()
    for ep in range(protocol['teacher_epochs']):
        model.train()
        for xb,yb in train:
            xb,yb=xb.cuda(),yb.cuda()
            logits=model(xb)
            if loss_name=='balanced_softmax': logits=logits+adjust
            objective=torch.nn.functional.cross_entropy(logits,yb)
            opt.zero_grad(set_to_none=True); objective.backward(); opt.step()
        m=metrics(selection_y,predict(model,selection_x))
        trace.append(dict(epoch=ep+1,selection_bal_acc=m['bal_acc'],selection_macro_f1=m['macro_f1']))
        if m['macro_f1']>best:
            best=m['macro_f1'];selected_epoch=ep+1
            plain=model._module if hasattr(model,'_module') else model
            state={k:v.detach().cpu().clone() for k,v in plain.state_dict().items()}
    final=LinearHead(768,3).cuda();final.load_state_dict(state)
    info=dict(seed=seed,sigma=sigma,loss=loss_name,initial_state=initial,best_epoch=selected_epoch,
              trace=trace,seconds=time.perf_counter()-t0,class_counts=counts.tolist(),class_counts_source='auxiliary')
    if engine:
        info['accountant']=dict(epsilon=float(engine.get_epsilon(protocol['delta'])),delta=protocol['delta'],
            history=list(engine.accountant.history),steps=sum(t[2] for t in engine.accountant.history),
            sample_rate=float(train.sample_rate),dataset_size=len(x),nominal_batch_size=protocol['batch_size'],
            sampling='poisson',adjacency='add_remove_one_training_record',max_grad_norm=protocol['max_grad_norm'])
    else: info['accountant']=None
    return final,info


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--bundle',type=Path,required=True)
    ap.add_argument('--protocol',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--seeds',type=int,nargs='+',default=list(range(10)))
    args=ap.parse_args();torch.set_num_threads(4)
    p=json.loads(args.protocol.read_text());args.out.mkdir(parents=True,exist_ok=True)
    target_protocol=args.out/'protocol.json'
    if target_protocol.exists(): assert json.loads(target_protocol.read_text())==p
    else: target_protocol.write_text(json.dumps(p,indent=2))
    frames={s:pd.read_csv(args.bundle/f'{s}_manifest.csv') for s in ['train','selection','calibration','test']}
    source=json.loads((args.bundle/'source_manifest.json').read_text())
    assert source['sampling_revision']=='quote_none_20260908'
    expected={'train':p['training_rows'],'selection':p['selection_rows'],
              'calibration':p['calibration_rows'],'test':p['test_rows']}
    assert {s:len(f) for s,f in frames.items()}==expected
    assert frames['train'].groupby('client_id').size().max()<=20
    x={s:np.load(args.bundle/f'{s}_ssl.npy') for s in frames}
    y={s:np.load(args.bundle/f'{s}_labels.npy') for s in frames}
    for a in frames:
        assert len(x[a])==len(y[a])==len(frames[a]) and np.isfinite(x[a]).all()
        for b in frames:
            if a!=b:
                assert set(frames[a].client_id).isdisjoint(frames[b].client_id)
                assert set(frames[a].path).isdisjoint(frames[b].path)
    for seed in args.seeds:
        out=args.out/f'seed{seed}';out.mkdir(exist_ok=True)
        private,aux=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=seed).split(
                        frames['train'],groups=frames['train'].client_id))
        assert set(frames['train'].iloc[private].client_id).isdisjoint(frames['train'].iloc[aux].client_id)
        scaler=StandardScaler().fit(x['train'][aux])
        np.savez(out/'normalization.npz',mean=scaler.mean_,scale=scaler.scale_)
        (out/'split.json').write_text(json.dumps(dict(private=private.tolist(),auxiliary=aux.tolist())))
        def transform(a):
            z=scaler.transform(a).astype(np.float32)
            z/=np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
            assert np.isfinite(z).all() and np.linalg.norm(z,axis=1).max()<1.00001
            return torch.from_numpy(z)
        z={s:transform(a) for s,a in x.items()}
        auxx=z['train'][aux];auxy=torch.tensor(y['train'][aux])
        dev_loader=DataLoader(TensorDataset(z['selection'],torch.tensor(y['selection'])),batch_size=256)
        counts=np.bincount(y['train'][aux],minlength=3);assert (counts>0).all()

        def student(mode,teacher_id,soft=None,health=None,weight_mode='instance'):
            name='hard' if mode=='hard' else f'{teacher_id}_{mode}_{weight_mode}'
            record=out/f'{name}.json'
            if record.exists():return json.loads(record.read_text())
            seed_all(seed+1000003);model=LinearHead(768,3).cuda()
            initial=_state_dict_sha256(model)
            alpha=p['kd_alpha']
            if mode=='kd_meanmatched':
                prob=torch.softmax(torch.tensor(soft)/p['kd_temperature'],dim=1)
                alpha*=float((prob*torch.tensor(health)).sum(1).mean())
            train=DataLoader(TensorDataset(auxx,auxy) if mode=='hard' else TensorDataset(auxx,auxy,torch.tensor(soft)),
                batch_size=p['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(seed+1000004))
            t0=time.perf_counter()
            with (out/f'{name}.log').open('w') as log, contextlib.redirect_stdout(log):
                info=train_student(model,train,dev_loader,torch.device('cuda'),3,epochs=p['student_epochs'],
                    lr=p['learning_rate'],weight_decay=p['weight_decay'],
                    student_mode='kd' if mode=='kd_meanmatched' else mode,kd_alpha=alpha,
                    kd_temperature=p['kd_temperature'],class_health=health,hakd_weight_mode=weight_mode)
            info.update(initial_state=initial,order_seed=seed+1000004,seconds=time.perf_counter()-t0,
                        configured_kd_alpha=alpha)
            predictions={s:predict(model,z[s]) for s in ['selection','calibration','test']}
            np.savez_compressed(out/f'{name}_predictions.npz',**predictions)
            torch.save(model.state_dict(),out/f'{name}.pt')
            result=dict(kind='student',seed=seed,teacher_id=None if mode=='hard' else teacher_id,mode=mode,
                weight_mode=weight_mode,info=info,metrics={s:metrics(y[s],v) for s,v in predictions.items()})
            record.write_text(json.dumps(result,indent=2))
            print('STUDENT',seed,name,result['metrics']['test']['bal_acc'],flush=True)
            return result

        student('hard',None)
        conditions=[(s,'balanced_softmax',True) for s in p['noise_multipliers']]
        conditions += [(0.,'ce',False),(0.,'balanced_softmax',False),(1.,'ce',False)]
        for sigma,loss,branches in conditions:
            name=f'teacher_sigma{sigma:g}_{loss}'
            record=out/f'{name}.json'
            if record.exists():
                result=json.loads(record.read_text());teacher=LinearHead(768,3).cuda()
                teacher.load_state_dict(torch.load(out/f'{name}.pt',map_location='cpu',weights_only=True))
                predictions=dict(np.load(out/f'{name}_predictions.npz'))
            else:
                teacher,info=teacher_train(z['train'][private],torch.tensor(y['train'][private]),
                    z['selection'],y['selection'],counts,p,seed,sigma,loss)
                predictions={s:predict(teacher,z[s]) for s in ['selection','calibration','test']}
                predictions['auxiliary']=predict(teacher,auxx)
                np.savez_compressed(out/f'{name}_predictions.npz',**predictions)
                torch.save(teacher.state_dict(),out/f'{name}.pt')
                m={s:metrics(y[s],predictions[s]) for s in ['selection','calibration','test']}
                h=compute_teacher_health(m['calibration'])
                cluster=cluster_health(y['calibration'],predictions['calibration'],
                    frames['calibration'].client_id.to_numpy(),seed+7919)
                iid_se=estimate_health_se(m['calibration'],seed=seed+7919)
                result=dict(kind='teacher',seed=seed,teacher_id=name,info=info,metrics=m,
                    health=dict(global_health=h['global_health'],class_health=h['class_health'].tolist(),
                                cluster=cluster,iid_se=iid_se),
                    route=resolve_student_mode('tcrd',h['global_health']),
                    rc_route=certified_rc_tcrd_mode(h['global_health'],cluster['se']),
                    iid_rc_route=certified_rc_tcrd_mode(h['global_health'],iid_se))
                record.write_text(json.dumps(result,indent=2))
                print('TEACHER',seed,sigma,loss,result['health']['global_health'],result['route'],flush=True)
            if branches:
                for mode,weight in [('kd','instance'),('hakd','instance'),('hakd','class'),
                                    ('hakd','confidence'),('ctkd_global','instance'),('kd_meanmatched','instance')]:
                    student(mode,name,predictions['auxiliary'],result['health']['class_health'],weight)
            del teacher
        (out/'complete.json').write_text(json.dumps(dict(teachers=8,students=31,seed=seed)))
    print('STUDY_COMPLETE',args.seeds,flush=True)


if __name__=='__main__':
    main()
