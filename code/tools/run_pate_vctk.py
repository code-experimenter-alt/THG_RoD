# -*- coding: utf-8 -*-
"""Minimal, auditable PATE baseline on the fixed VCTK split.

Each teacher receives a disjoint private shard.  Only noisy vote counts on
permitted auxiliary examples are exposed to the student.  The Gaussian RDP
accounting below covers the released noisy-count queries; private teacher
weights and labels are never released.
"""

import argparse
import json
import math
import os
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experiments.presets import PRESETS
from src.dataset import NumpyDataset, build_features_and_splits, set_seed
from src.metrics import eval_all
from src.models import build_model


def make_cfg(args, seed):
    values = dict(PRESETS["G0a_NonDP_cnn_vanilla"])
    values.update(
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        train_csv="train.csv",
        dev_csv="dev.csv",
        test_csv="test.csv",
        train_folder=".",
        dev_folder=".",
        test_folder=".",
        label_col="accent",
        group_col="speaker_id",
        split_mode="official",
        ratios=(0.6, 0.2, 0.1, 0.1),
        target_sr=16000,
        num_workers=0,
        pin_memory=False,
        analysis_infer_sweep=False,
        infer_counts_source="aux",
        record_grad_stats=False,
        use_dsaf=False,
        eta0=1e-5,
        seed=seed,
        exp_id="PATE_VCTK",
        sigma=0.0,
        run_tag="pate",
        student_backend="mel",
        teacher_backend="mel",
        model_student="cnn_mel",
        dp_teacher=False,
        train_teacher=True,
        student_mode="hard",
    )
    return SimpleNamespace(**values)


def train_cnn(X, y, cfg, seed, epochs):
    set_seed(seed)
    model = build_model(cfg, "cnn_mel", X, num_classes=int(np.max(y)) + 1)
    loader = DataLoader(NumpyDataset(X, y), batch_size=64, shuffle=True, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    for _ in range(int(epochs)):
        for xb, yb in loader:
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model


def gaussian_rdp_epsilon(queries, noise_sigma, delta):
    # One vote change has L2 sensitivity sqrt(2) for a one-hot count vector.
    sensitivity_sq = 2.0
    candidates = []
    for order in range(2, 257):
        rdp = float(queries) * order * sensitivity_sq / (2.0 * noise_sigma**2)
        candidates.append(rdp + math.log(1.0 / delta) / (order - 1.0))
    return float(min(candidates))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--teachers", type=int, default=3)
    parser.add_argument("--teacher_epochs", type=int, default=4)
    parser.add_argument("--student_epochs", type=int, default=5)
    parser.add_argument("--noise_sigma", type=float, default=15.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for seed in args.seeds:
        cfg = make_cfg(args, seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        feat, labels, class_names = build_features_and_splits(cfg, device)
        X_priv, X_aux, X_dev, X_test = feat["mel"]
        y_priv, y_aux, y_dev, y_test = labels
        K = len(class_names)
        if len(y_priv) % args.teachers:
            raise ValueError("private shard count must divide the private split")
        shard_ids = np.array_split(np.arange(len(y_priv)), args.teachers)
        teachers = []
        for t, ids in enumerate(shard_ids):
            teachers.append(train_cnn(X_priv[ids], y_priv[ids], cfg, seed * 100 + t, args.teacher_epochs))
        with torch.no_grad():
            vote_matrix = []
            dev_votes = []
            for teacher in teachers:
                teacher.eval()
                vote_matrix.append(teacher(torch.from_numpy(X_aux)).argmax(1).cpu().numpy())
                dev_votes.append(teacher(torch.from_numpy(X_dev)).argmax(1).cpu().numpy())
        votes = np.stack(vote_matrix, axis=1)
        rng = np.random.default_rng(seed + 9000)
        noisy_labels = np.empty(len(y_aux), dtype=np.int64)
        margins = []
        for i, row in enumerate(votes):
            counts = np.bincount(row, minlength=K).astype(np.float64)
            noisy = counts + rng.normal(0.0, args.noise_sigma, size=K)
            order = np.argsort(noisy)[::-1]
            noisy_labels[i] = int(order[0])
            margins.append(float(noisy[order[0]] - noisy[order[1]]))
        student = train_cnn(X_aux, noisy_labels, cfg, seed + 5000, args.student_epochs)
        test_loader = DataLoader(NumpyDataset(X_test, y_test), batch_size=64, shuffle=False)
        student_test = eval_all(student, test_loader, device=torch.device("cpu"), num_classes=K)
        dev_vote_arr = np.stack(dev_votes, axis=1)
        dev_vote_pred = np.eye(K, dtype=np.int64)[dev_vote_arr].sum(axis=1).argmax(axis=1)
        dev_acc = float(np.mean(dev_vote_pred == y_dev))
        payload = {
            "exp_id": "PATE_VCTK",
            "seed": int(seed),
            "class_names": class_names,
            "teachers": int(args.teachers),
            "private_shard_sizes": [int(len(x)) for x in shard_ids],
            "query_budget": int(len(y_aux)),
            "aggregation": "Gaussian noisy argmax over disjoint-teacher votes",
            "noise_sigma": float(args.noise_sigma),
            "delta": float(args.delta),
            "epsilon_rdp": gaussian_rdp_epsilon(len(y_aux), args.noise_sigma, args.delta),
            "dev_ensemble_acc": dev_acc,
            "mean_noisy_margin": float(np.mean(margins)),
            "student_test": student_test,
            "notes": "Teacher parameters and private labels are not released; only noisy auxiliary votes supervise the student.",
        }
        path = os.path.join(args.out_dir, f"PATE_VCTK_seed{seed}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(json.dumps({"seed": seed, "epsilon": payload["epsilon_rdp"], "student_bal_acc": student_test["bal_acc"]}))


if __name__ == "__main__":
    main()
