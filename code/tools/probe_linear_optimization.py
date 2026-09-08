"""Development-only convergence check on the earlier, distinct CV archive."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models import LinearHead


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--bundle', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args=ap.parse_args()
    torch.set_num_threads(4)
    x=np.load(args.bundle/'train_ssl.npy'); y=np.load(args.bundle/'train_labels.npy')
    dx=np.load(args.bundle/'dev_ssl.npy'); dy=np.load(args.bundle/'dev_labels.npy')
    sp=json.loads((args.bundle/'splits.json').read_text())['seeds']['0']
    # No test file is opened. Permitted auxiliary features alone fit scaling.
    scaler=StandardScaler().fit(x[sp['auxiliary']])
    def convert(a):
        z=scaler.transform(a).astype(np.float32)
        z=z/np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
        return torch.from_numpy(z)
    train=convert(x[sp['private']]); dev=convert(dx)
    target=torch.tensor(y[sp['private']])
    counts=np.bincount(y[sp['auxiliary']], minlength=3)
    rows=[]
    for loss in ['ce', 'balanced_softmax']:
        for lr in [.003, .01, .03]:
            torch.manual_seed(0)
            model=LinearHead(768,3)
            optimizer=torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            gen=torch.Generator().manual_seed(4444)
            trace=[]
            for epoch in range(80):
                for idx in torch.randperm(len(train), generator=gen).split(256):
                    logits=model(train[idx])
                    if loss=='balanced_softmax':
                        logits=logits+torch.tensor(np.log(counts),dtype=torch.float32)
                    objective=torch.nn.functional.cross_entropy(logits, target[idx])
                    optimizer.zero_grad(set_to_none=True); objective.backward(); optimizer.step()
                with torch.no_grad():
                    pred=model(dev).argmax(1).numpy()
                trace.append(dict(epoch=epoch+1, dev_bal_acc=balanced_accuracy_score(dy,pred),
                    dev_macro_f1=f1_score(dy,pred,average='macro')))
            row=dict(loss=loss,lr=lr,trace=trace,best=max(trace,key=lambda q:q['dev_macro_f1']))
            rows.append(row)
            args.out.write_text(json.dumps(dict(source='earlier CV archive, not CV17',
                test_accessed=False, private_data_protection='non-DP diagnostic only; not a release',
                seed=0, normalization='auxiliary StandardScaler then per-record unit L2',
                batch_size=256, max_epochs=80, rows=rows),indent=2))
            print(loss,lr,'best',row['best'],'20/40/80',[trace[i-1] for i in [20,40,80]],flush=True)

if __name__=='__main__':
    main()
