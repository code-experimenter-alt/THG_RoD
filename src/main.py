# -*- coding: utf-8 -*-

import os
import json
import time
import argparse
import warnings
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from opacus.validators import ModuleValidator

from experiments.presets import PRESETS, MINIMAL_SET
from .dataset import set_seed, CLASS_NAMES, NumpyDataset, build_features_and_splits, make_loaders
from .models import build_model
from .losses import compute_class_counts, make_infer_adjust_from_counts
from .metrics import eval_all, compute_teacher_health
from .routing import resolve_student_mode, mode_needs_teacher_logits
from .train_teacher import train_teacher, compute_soft_logits, teacher_cache_id, unwrap_opacus_model, _safe_tag
from .train_student import train_student

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

warnings.filterwarnings(
    "ignore",
    message="Full backward hook is firing when gradients are computed with respect to module outputs*",
)


@dataclass
class RunCfg:
    data_root: str
    cache_dir: str = "outputs/cache_features"

    train_csv: str = "cv-valid-train.csv"
    dev_csv: str = "cv-valid-dev.csv"
    test_csv: str = "cv-valid-test.csv"
    train_folder: str = "cv-valid-train"
    dev_folder: str = "cv-valid-dev"
    test_folder: str = "cv-valid-test"

    label_col: str = "accent"
    group_col: Optional[str] = None

    split_mode: str = "official"
    ratios: Tuple[float, float, float, float] = (0.6, 0.2, 0.1, 0.1)

    feature_backend: str = "mel"
    target_sr: int = 16000

    teacher_backend: str = "mel"
    student_backend: str = "mel"

    analysis_infer_sweep: bool = False
    infer_counts_source: str = "aux"
    record_grad_stats: bool = True

    n_mels: int = 40
    max_time: int = 100
    use_dsaf: bool = False
    eta0: float = 1e-5

    ssl_model_name: str = "facebook/wav2vec2-base"
    ssl_pooling: str = "mean"
    ssl_max_seconds: float = 12.0

    model_teacher: str = "cnn_mel"
    model_student: Optional[str] = None

    batch_size: int = 128
    num_workers: int = 0
    pin_memory: bool = False

    epochs_teacher: int = 8
    epochs_student: int = 15
    lr_teacher: float = 1e-3
    lr_student: float = 1e-3
    weight_decay: float = 1e-4

    dp_teacher: bool = True
    max_grad_norm: float = 1.0
    delta: float = 1e-5

    loss_name: str = "ce"
    logit_adjust_tau: float = 1.0

    train_teacher: bool = True
    student_mode: str = "kd"
    kd_alpha: float = 0.7
    kd_temperature: float = 2.0

    health_power: float = 1.0
    health_floor: float = 0.0
    tcrd_tau_low: float = 0.45
    tcrd_tau_high: float = 0.60

    out_dir: str = "outputs/runs"
    exp_id: str = "unknown"
    seed: int = 0
    sigma: float = 1.0
    run_tag: str = ""


def _apply_if_not_none(cfg: RunCfg, attr: str, value):
    if value is not None:
        setattr(cfg, attr, value)


def run_once(cfg: RunCfg, device: torch.device) -> Dict[str, Any]:
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    num_classes = 3
    feat, (y_priv, y_aux, y_dev, y_test) = build_features_and_splits(cfg, device)

    X_priv_t, X_aux_t, X_dev_t, X_test_t = feat[cfg.teacher_backend]

    priv_loader_t, _, dev_loader_t, test_loader_t = make_loaders(
        cfg, device,
        X_priv_t, y_priv,
        X_aux_t, y_aux,
        X_dev_t, y_dev,
        X_test_t, y_test,
        soft_aux=None,
    )

    class_counts_priv = compute_class_counts(y_priv, num_classes=num_classes)
    teacher = build_model(cfg, cfg.model_teacher, X_priv_t, num_classes=num_classes)

    tcache = teacher_cache_id(cfg)
    teacher_ckpt = os.path.join(cfg.out_dir, f"{tcache}_teacher.pt")
    teacher_info_path = os.path.join(cfg.out_dir, f"{tcache}_teacher_info.json")
    teacher_info: Dict[str, Any] = {}

    if cfg.train_teacher:
        teacher, teacher_info = train_teacher(
            model=teacher,
            train_loader=priv_loader_t,
            dev_loader=dev_loader_t,
            device=device,
            num_classes=num_classes,
            epochs=cfg.epochs_teacher,
            lr=cfg.lr_teacher,
            weight_decay=cfg.weight_decay,
            dp=cfg.dp_teacher,
            sigma=cfg.sigma,
            max_grad_norm=cfg.max_grad_norm,
            delta=cfg.delta,
            loss_name=cfg.loss_name,
            class_counts_np=class_counts_priv,
            logit_adjust_tau=cfg.logit_adjust_tau,
            record_grad_stats=bool(cfg.record_grad_stats),
        )

        plain_teacher = unwrap_opacus_model(teacher)
        torch.save(plain_teacher.state_dict(), teacher_ckpt)
        with open(teacher_info_path, "w", encoding="utf-8") as f:
            json.dump(teacher_info, f, ensure_ascii=False, indent=2)
        print("[teacher-ckpt saved]", teacher_ckpt)
    else:
        if not os.path.exists(teacher_ckpt):
            raise FileNotFoundError(f"Teacher ckpt not found: {teacher_ckpt}")

        if cfg.dp_teacher:
            teacher = ModuleValidator.fix(teacher)
            ModuleValidator.validate(teacher, strict=False)

        sd = torch.load(teacher_ckpt, map_location="cpu")
        teacher.load_state_dict(sd)
        teacher = teacher.to(device)

        if os.path.exists(teacher_info_path):
            with open(teacher_info_path, "r", encoding="utf-8") as f:
                teacher_info = json.load(f)
        teacher_info["loaded_ckpt"] = teacher_ckpt
        print("[teacher-ckpt loaded]", teacher_ckpt)

    m_teacher_dev = eval_all(teacher, dev_loader_t, device, num_classes=num_classes, infer_adjust=None)
    m_teacher_test = eval_all(teacher, test_loader_t, device, num_classes=num_classes, infer_adjust=None)

    teacher_health = compute_teacher_health(
        m_teacher_dev,
        power=float(cfg.health_power),
        floor=float(cfg.health_floor),
    )
    teacher_global_health = float(teacher_health["global_health"])
    teacher_class_health = teacher_health["class_health"]

    teacher_info["teacher_health_dev"] = {
        "global_health": teacher_global_health,
        "class_health": teacher_class_health.tolist(),
    }

    print(
        f"[teacher-health] global={teacher_global_health:.4f} "
        f"class_health={[round(float(x), 4) for x in teacher_class_health.tolist()]}"
    )

    if getattr(cfg, "analysis_infer_sweep", False):
        infer_taus_teacher = [0.0, 0.5, 1.0, 1.5, 2.0]
        counts_for_infer = compute_class_counts(y_aux if cfg.infer_counts_source == "aux" else y_priv, num_classes)
        sweep = []
        for tau_infer in infer_taus_teacher:
            infer_adjust = make_infer_adjust_from_counts(counts_for_infer, tau=tau_infer, device=device)
            m = eval_all(teacher, dev_loader_t, device, num_classes=num_classes, infer_adjust=infer_adjust)
            sweep.append({"tau_infer": float(tau_infer), **m})
        teacher_info["analysis_infer_sweep"] = sweep

    student_info: Dict[str, Any] = {}
    m_student_dev = None
    m_student_test = None

    tag = _safe_tag(cfg.run_tag)
    tag_suffix = f"_tag{tag}" if tag else ""

    if cfg.model_student is not None:
        X_priv_s, X_aux_s, X_dev_s, X_test_s = feat[cfg.student_backend]

        effective_mode = resolve_student_mode(
            requested_mode=cfg.student_mode,
            teacher_global_health=teacher_global_health,
            tau_low=float(cfg.tcrd_tau_low),
            tau_high=float(cfg.tcrd_tau_high),
        )
        needs_teacher_logits = mode_needs_teacher_logits(effective_mode)
        soft_path = os.path.join(cfg.out_dir, f"{tcache}_soft_logits_aux.npy")

        if needs_teacher_logits:
            if os.path.exists(soft_path):
                soft_aux = np.load(soft_path)
                print("[soft-logits loaded]", soft_path)
            else:
                aux_loader_for_logits = DataLoader(
                    NumpyDataset(X_aux_t, y_aux),
                    batch_size=cfg.batch_size,
                    shuffle=False,
                    num_workers=cfg.num_workers,
                    pin_memory=(cfg.pin_memory and device.type == "cuda"),
                )
                soft_aux = compute_soft_logits(teacher, aux_loader_for_logits, device)
                np.save(soft_path, soft_aux)
                print("[soft-logits saved]", soft_path)
        else:
            soft_aux = None

        _, _, dev_loader_s, test_loader_s = make_loaders(
            cfg, device,
            X_priv_s, y_priv,
            X_aux_s, y_aux,
            X_dev_s, y_dev,
            X_test_s, y_test,
            soft_aux=None,
        )

        student = build_model(cfg, cfg.model_student, X_aux_s, num_classes=num_classes)

        if needs_teacher_logits:
            student_train_loader = DataLoader(
                NumpyDataset(X_aux_s, y_aux, soft_logits=soft_aux),
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                pin_memory=(cfg.pin_memory and device.type == "cuda"),
            )
        else:
            student_train_loader = DataLoader(
                NumpyDataset(X_aux_s, y_aux),
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                pin_memory=(cfg.pin_memory and device.type == "cuda"),
            )

        student_info = train_student(
            student=student,
            train_loader=student_train_loader,
            dev_loader=dev_loader_s,
            device=device,
            num_classes=num_classes,
            epochs=cfg.epochs_student,
            lr=cfg.lr_student,
            weight_decay=cfg.weight_decay,
            student_mode=cfg.student_mode,
            kd_alpha=cfg.kd_alpha,
            kd_temperature=cfg.kd_temperature,
            class_health=teacher_class_health,
            teacher_global_health=teacher_global_health,
            tcrd_tau_low=cfg.tcrd_tau_low,
            tcrd_tau_high=cfg.tcrd_tau_high,
        )

        m_student_dev = eval_all(student, dev_loader_s, device, num_classes=num_classes, infer_adjust=None)
        m_student_test = eval_all(student, test_loader_s, device, num_classes=num_classes, infer_adjust=None)

        student_info["requested_mode"] = str(cfg.student_mode)
        student_info["effective_mode"] = str(effective_mode)
        student_info["needs_teacher_logits"] = bool(needs_teacher_logits)

    out = dict(
        exp_id=cfg.exp_id,
        seed=int(cfg.seed),
        sigma=float(cfg.sigma),
        run_tag=str(cfg.run_tag),
        teacher_cache_id=str(tcache),
        cfg=asdict(cfg),
        class_names=CLASS_NAMES,
        teacher_info=teacher_info,
        teacher_dev=m_teacher_dev,
        teacher_test=m_teacher_test,
        student_info=student_info,
        student_dev=m_student_dev,
        student_test=m_student_test,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    out_json = os.path.join(cfg.out_dir, f"{cfg.exp_id}_seed{cfg.seed}_sigma{cfg.sigma:g}{tag_suffix}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("[saved]", out_json)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/runs")
    parser.add_argument("--cache_dir", type=str, default="outputs/cache_features")

    parser.add_argument("--exp_id", type=str, default=None)
    parser.add_argument("--run_all_minimal", action="store_true")

    parser.add_argument("--split_mode", type=str, default="official", choices=["custom", "official"])
    parser.add_argument("--group_col", type=str, default=None)

    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--sigmas", type=float, nargs="+", default=[1.0])

    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--train_teacher", type=int, choices=[0, 1], default=None)

    parser.add_argument("--lr_teacher", type=float, default=None)
    parser.add_argument("--epochs_teacher", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)

    parser.add_argument("--dp_teacher", type=int, choices=[0, 1], default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--delta", type=float, default=None)

    parser.add_argument("--loss_name", type=str, choices=["ce", "balanced_softmax", "logit_adjust"], default=None)
    parser.add_argument("--logit_adjust_tau", type=float, default=None)

    parser.add_argument("--n_mels", type=int, default=None)
    parser.add_argument("--max_time", type=int, default=None)
    parser.add_argument("--use_dsaf", type=int, choices=[0, 1], default=None)
    parser.add_argument("--eta0", type=float, default=None)
    parser.add_argument("--target_sr", type=int, default=None)

    parser.add_argument("--model_student", type=str, choices=["linear_head", "cnn_mel"], default=None)
    parser.add_argument("--student_backend", type=str, choices=["ssl", "mel"], default=None)

    parser.add_argument("--kd_alpha", type=float, default=None)
    parser.add_argument("--kd_temperature", type=float, default=None)
    parser.add_argument("--epochs_student", type=int, default=None)
    parser.add_argument("--lr_student", type=float, default=None)

    parser.add_argument("--student_mode", type=str, choices=["hard", "kd", "hakd", "tcrd"], default=None)
    parser.add_argument("--health_power", type=float, default=None)
    parser.add_argument("--health_floor", type=float, default=None)
    parser.add_argument("--tcrd_tau_low", type=float, default=None)
    parser.add_argument("--tcrd_tau_high", type=float, default=None)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.run_all_minimal:
        exp_ids = MINIMAL_SET
    else:
        if args.exp_id is None:
            raise ValueError("Provide --exp_id or use --run_all_minimal")
        exp_ids = [args.exp_id]

    for exp_id in exp_ids:
        preset = PRESETS.get(exp_id)
        if preset is None:
            avail = ", ".join(sorted(PRESETS.keys()))
            raise ValueError(f"Unknown exp_id={exp_id}. Available: {avail}")

        for seed in args.seeds:
            for sigma in args.sigmas:
                cfg = RunCfg(
                    data_root=args.data_root,
                    out_dir=args.out_dir,
                    cache_dir=args.cache_dir,
                    exp_id=exp_id,
                    seed=int(seed),
                    sigma=float(sigma),
                    split_mode=args.split_mode,
                    group_col=args.group_col,
                    run_tag=str(args.tag or ""),
                )

                for k, v in preset.items():
                    setattr(cfg, k, v)

                if args.train_teacher is not None:
                    cfg.train_teacher = bool(int(args.train_teacher))
                if args.dp_teacher is not None:
                    cfg.dp_teacher = bool(int(args.dp_teacher))
                if args.use_dsaf is not None:
                    cfg.use_dsaf = bool(int(args.use_dsaf))

                _apply_if_not_none(cfg, "lr_teacher", args.lr_teacher)
                _apply_if_not_none(cfg, "epochs_teacher", args.epochs_teacher)
                _apply_if_not_none(cfg, "batch_size", args.batch_size)
                _apply_if_not_none(cfg, "weight_decay", args.weight_decay)
                _apply_if_not_none(cfg, "max_grad_norm", args.max_grad_norm)
                _apply_if_not_none(cfg, "delta", args.delta)

                _apply_if_not_none(cfg, "loss_name", args.loss_name)
                _apply_if_not_none(cfg, "logit_adjust_tau", args.logit_adjust_tau)

                _apply_if_not_none(cfg, "n_mels", args.n_mels)
                _apply_if_not_none(cfg, "max_time", args.max_time)
                _apply_if_not_none(cfg, "eta0", args.eta0)
                _apply_if_not_none(cfg, "target_sr", args.target_sr)

                _apply_if_not_none(cfg, "model_student", args.model_student)
                _apply_if_not_none(cfg, "student_backend", args.student_backend)

                _apply_if_not_none(cfg, "kd_alpha", args.kd_alpha)
                _apply_if_not_none(cfg, "kd_temperature", args.kd_temperature)
                _apply_if_not_none(cfg, "epochs_student", args.epochs_student)
                _apply_if_not_none(cfg, "lr_student", args.lr_student)

                _apply_if_not_none(cfg, "student_mode", args.student_mode)
                _apply_if_not_none(cfg, "health_power", args.health_power)
                _apply_if_not_none(cfg, "health_floor", args.health_floor)
                _apply_if_not_none(cfg, "tcrd_tau_low", args.tcrd_tau_low)
                _apply_if_not_none(cfg, "tcrd_tau_high", args.tcrd_tau_high)

                print("\n" + "=" * 80)
                print(
                    f"[RUN] exp_id={exp_id} seed={seed} sigma={sigma} tag='{cfg.run_tag}' "
                    f"teacher_backend={cfg.teacher_backend} student_backend={cfg.student_backend} "
                    f"dp_teacher={cfg.dp_teacher} loss={cfg.loss_name} "
                    f"student_mode={cfg.student_mode} lr={cfg.lr_teacher:g} "
                    f"C={cfg.max_grad_norm:g} train_teacher={cfg.train_teacher}"
                )
                print("=" * 80)

                run_once(cfg, device)


if __name__ == "__main__":
    main()