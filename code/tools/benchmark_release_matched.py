"""Matched warm-cache release benchmark with real training and CUDA fences."""
import argparse
import contextlib
import gc
import io
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models import LinearHead, SmallCNNMel
from src.train_student import train_student
from src.train_teacher import compute_soft_logits
from src.metrics import eval_all, compute_teacher_health, estimate_health_se
from src.routing import resolve_student_mode, certified_rc_tcrd_mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', type=Path, required=True)
    ap.add_argument('--teachers', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--repeats', type=int, default=3)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device('cuda')
    splits = json.loads((args.bundle/'splits.json').read_text())
    modes = ['hard', 'kd', 'hakd', 'tcrd', 'rc_tcrd', 's_bacc']
    records = []
    def clock():
        torch.cuda.synchronize()
        return time.perf_counter()

    for backend in ('ssl', 'mel'):
        x = np.load(args.bundle/f'train_{backend}.npy')
        dx = np.load(args.bundle/f'dev_{backend}.npy')
        y = np.load(args.bundle/'train_labels.npy')
        dy = np.load(args.bundle/'dev_labels.npy')
        make_model = (lambda: LinearHead(768, 3)) if backend == 'ssl' else (lambda: SmallCNNMel(40, 100, 3))
        for seed in range(3):
            indices = splits['seeds'][str(seed)]['auxiliary']
            ax, ay = torch.tensor(x[indices]), torch.tensor(y[indices])
            aux_eval = DataLoader(TensorDataset(ax, ay), batch_size=128, shuffle=False)
            dev = DataLoader(TensorDataset(torch.tensor(dx), torch.tensor(dy)), batch_size=128)
            loss = 'balanced_softmax' if backend == 'ssl' else 'ce'
            checkpoint = list((args.teachers/backend/f'seed{seed}').glob(f'seed{seed}_sigma1_{backend}_*dp1_*_{loss}_prioraux_*_teacher.pt'))
            assert len(checkpoint) == 1, checkpoint
            teacher = make_model().to(device)
            teacher.load_state_dict(torch.load(checkpoint[0], map_location='cpu', weights_only=True))
            teacher.eval()

            def run(mode, repeat):
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                baseline = torch.cuda.memory_allocated()/1024**2
                start = clock()
                hsec = rsec = qsec = tsec = selectsec = 0.
                health = None
                if mode in ('hakd', 'tcrd', 'rc_tcrd', 's_bacc'):
                    t = clock()
                    metrics = eval_all(teacher, dev, device, 3)
                    health = compute_teacher_health(metrics)
                    se = estimate_health_se(metrics, seed=seed+7919) if mode == 'rc_tcrd' else None
                    hsec = clock()-t
                t = clock()
                if mode == 'tcrd':
                    selected = resolve_student_mode('tcrd', health['global_health'])
                elif mode == 'rc_tcrd':
                    selected = certified_rc_tcrd_mode(health['global_health'], se)['mode']
                else:
                    selected = mode
                rsec = clock()-t
                candidates = ['hard', 'kd', 'hakd'] if mode == 's_bacc' else [selected]
                soft = None
                if any(c != 'hard' for c in candidates):
                    t = clock()
                    soft = torch.tensor(compute_soft_logits(teacher, aux_eval, device))
                    qsec = clock()-t
                summaries = []
                for candidate in candidates:
                    t = clock()
                    torch.manual_seed(seed+1000003)
                    student = make_model().to(device)
                    train = DataLoader(TensorDataset(ax, ay) if candidate == 'hard' else TensorDataset(ax, ay, soft),
                        batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(seed+1000004))
                    with contextlib.redirect_stdout(io.StringIO()):
                        info = train_student(student, train, dev, device, 3, epochs=15,
                            lr=1e-3, weight_decay=1e-4, student_mode=candidate, kd_alpha=.7,
                            kd_temperature=2., class_health=None if health is None else health['class_health'])
                    tsec += clock()-t
                    assert 1 <= info['best_epoch'] <= 15
                    t = clock()
                    utility = eval_all(student, dev, device, 3)['bal_acc'] if mode == 's_bacc' else None
                    selectsec += clock()-t
                    summaries.append(dict(mode=candidate, best_epoch=info['best_epoch'], dev_bal_acc=utility))
                    del student, train
                if mode == 's_bacc':
                    selected = max(summaries, key=lambda z:z['dev_bal_acc'])['mode']
                total = clock()-start
                return dict(backend=backend, seed=seed, repeat=repeat, mode=mode, selected=selected,
                    health_seconds=hsec, routing_seconds=rsec, query_seconds=qsec, student_seconds=tsec,
                    selection_seconds=selectsec, total_seconds=total,
                    peak_cuda_mb=torch.cuda.max_memory_allocated()/1024**2,
                    incremental_peak_cuda_mb=torch.cuda.max_memory_allocated()/1024**2-baseline,
                    student_epochs=15, candidates=summaries)

            # Unreported warmup covers every objective and kernels before measured repeats.
            for mode in modes:
                run(mode, -1)
            for repeat in range(args.repeats):
                for mode in np.random.default_rng(9000+100*seed+repeat).permutation(modes):
                    record = run(str(mode), repeat)
                    records.append(record)
                    (args.out/'runs.json').write_text(json.dumps(records, indent=2))
                    print(backend, seed, repeat, mode, round(record['total_seconds'], 4), flush=True)
            del teacher
    frame = pd.DataFrame([{k:v for k,v in r.items() if k != 'candidates'} for r in records])
    cols = ['health_seconds', 'routing_seconds', 'query_seconds', 'student_seconds', 'selection_seconds',
            'total_seconds', 'peak_cuda_mb', 'incremental_peak_cuda_mb']
    frame.groupby(['backend', 'mode'])[cols].agg(['mean', 'std', 'count']).to_csv(args.out/'summary.csv')
    (args.out/'protocol.json').write_text(json.dumps(dict(
        protocol='matched cached-feature release, excludes common teacher training and feature extraction',
        device=torch.cuda.get_device_name(), torch=torch.__version__, seeds=[0,1,2],
        repeats=args.repeats, student_epochs=15, batch_size=128,
        student_initialization_and_order='same within seed for all candidates/repeats',
        warmup='one unreported full run of every mode per seed',
        scheduling='fixed randomized mode order per repeat', cuda_synchronize='every stage boundary',
        health='teacher development inference included; uncertainty resampling only for RC',
        selection='S-BAcc actually trains all three students; first-max tie',
        samples_are_independent_datasets=False), indent=2))


if __name__ == '__main__':
    main()
