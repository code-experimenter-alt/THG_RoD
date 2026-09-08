"""Run a compact, feature-level IEMOCAP route comparison.

This is intentionally separate from the audio/CSV release entry point: the
Zenodo record contains WavLM features, not the USC raw audio release.  The
script consumes pooled NPZ files produced by prepare_iemocap_precomputed.py
and writes one auditable JSON per seed.
"""

import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import NumpyDataset, set_seed
from src.metrics import compute_teacher_health, estimate_health_se, eval_all
from src.models import LinearHead
from src.routing import certified_rc_tcrd_mode, resolve_student_mode
from src.train_student import train_student
from src.train_teacher import compute_soft_logits, train_teacher, unwrap_opacus_model


def loader(x, y, batch_size, shuffle, seed, soft=None):
    ds = NumpyDataset(x, y, soft_logits=soft)
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=gen, num_workers=0)


def clone_model(model):
    out = LinearHead(model.fc.in_features, model.fc.out_features)
    out.load_state_dict(copy.deepcopy(model.state_dict()))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs_teacher", type=int, default=4)
    parser.add_argument("--epochs_student", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train = np.load(args.data_dir / "train.npz", allow_pickle=False)
    dev = np.load(args.data_dir / "dev.npz", allow_pickle=False)
    test = np.load(args.data_dir / "test.npz", allow_pickle=False)
    manifest = np.genfromtxt(args.data_dir / "manifest.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
    train_rows = manifest[manifest["source_split"] == "train"]
    private_mask = train_rows["route_split"] == "private"
    aux_mask = train_rows["route_split"] == "auxiliary"
    private_idx = train_rows["row_index"][private_mask].astype(np.int64)
    aux_idx = train_rows["row_index"][aux_mask].astype(np.int64)
    x_priv = train["X"][private_idx]
    y_priv = train["y"][private_idx]
    x_aux = train["X"][aux_idx]
    y_aux = train["y"][aux_idx]
    x_dev, y_dev = dev["X"], dev["y"]
    x_test, y_test = test["X"], test["y"]
    if len(x_priv) == 0 or len(x_aux) == 0:
        raise RuntimeError("manifest produced an empty private or auxiliary split")
    num_classes = int(max(y_priv.max(), y_aux.max(), y_dev.max(), y_test.max()) + 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    for seed in args.seeds:
        set_seed(int(seed))
        train_loader = loader(x_priv, y_priv, args.batch_size, True, seed + 11)
        dev_loader = loader(x_dev, y_dev, args.batch_size, False, seed + 12)
        test_loader = loader(x_test, y_test, args.batch_size, False, seed + 13)
        teacher = LinearHead(x_priv.shape[1], num_classes)
        # Class-prior side information comes only from permitted auxiliary
        # labels and remains fixed under private-record adjacency.
        class_counts = np.maximum(np.bincount(y_aux, minlength=num_classes), 1)
        teacher, teacher_info = train_teacher(
            teacher, train_loader, dev_loader, device, num_classes,
            args.epochs_teacher, 1e-3, 1e-4, True, args.sigma,
            args.max_grad_norm, args.delta, "balanced_softmax", class_counts,
        )
        teacher_dev = eval_all(teacher, dev_loader, device, num_classes)
        teacher_test = eval_all(teacher, test_loader, device, num_classes)
        torch.save(unwrap_opacus_model(teacher).state_dict(), args.out_dir / f"teacher_seed{seed}.pt")
        np.savez(args.out_dir / f"teacher_dev_seed{seed}.npz",
                 logits=compute_soft_logits(teacher, dev_loader, device), y=y_dev)
        health = compute_teacher_health(teacher_dev)
        health_se = estimate_health_se(teacher_dev, draws=4000, seed=seed + 100)
        soft_loader = loader(x_aux, y_aux, args.batch_size, False, seed + 14)
        soft_aux = compute_soft_logits(teacher, soft_loader, device)

        set_seed(seed + 1000)
        base = LinearHead(x_aux.shape[1], num_classes)
        base_init = copy.deepcopy(base.state_dict())
        branch = {}
        for mode in ("hard", "kd", "hakd"):
            set_seed(seed + 2000)
            student = clone_model(base)
            student.load_state_dict(copy.deepcopy(base_init))
            train_loader_s = loader(x_aux, y_aux, args.batch_size, True, seed + 3000)
            if mode != "hard":
                train_loader_s = loader(x_aux, y_aux, args.batch_size, True, seed + 3000, soft=soft_aux)
            info = train_student(
                student, train_loader_s, dev_loader, device, num_classes,
                args.epochs_student, 1e-3, 1e-4, mode, 0.5, 2.0,
                class_health=np.asarray(health["class_health"], dtype=np.float32),
                teacher_global_health=float(health["global_health"]),
            )
            branch[mode] = {"info": info,
                            "dev": eval_all(student, dev_loader, device, num_classes),
                            "test": eval_all(student, test_loader, device, num_classes)}
            torch.save(student.state_dict(), args.out_dir / f"student_{mode}_seed{seed}.pt")

        tcrd_mode = resolve_student_mode("tcrd", health["global_health"], 0.45, 0.60)
        rc = certified_rc_tcrd_mode(health["global_health"], health_se, 0.45, 0.60, 0.10, "hard")
        row = {
            "seed": int(seed),
            "device": str(device),
            "source": "zenodo:10.5281/zenodo.17803295",
            "raw_derivative_not_usc_release": True,
            "num_classes": num_classes,
            "record_unit": "one unique utterance",
            "teacher_prior_source": "permitted_auxiliary_labels",
            "teacher_class_counts": class_counts.tolist(),
            "split_sizes": {"private": int(len(x_priv)), "auxiliary": int(len(x_aux)), "dev": int(len(x_dev)), "test": int(len(x_test))},
            "teacher": {"info": teacher_info, "dev": teacher_dev, "test": teacher_test, "health": health, "health_se": float(health_se)},
            "branches": branch,
            "tcrd": {"mode": tcrd_mode, "test": branch[tcrd_mode]["test"]},
            "rc_tcrd": {"decision": rc, "test": branch[rc["mode"]]["test"]},
            "config": dict(vars(args)),
        }
        row["config"]["data_dir"] = str(args.data_dir)
        row["config"]["out_dir"] = str(args.out_dir)
        out = args.out_dir / f"iemocap_precomputed_seed{seed}.json"
        out.write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
        rows.append(row)
        print(json.dumps({"seed": seed, "health": health, "tcrd_mode": tcrd_mode, "rc_mode": rc["mode"]}, indent=2, default=str))

    (args.out_dir / "summary.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
