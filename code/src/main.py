# -*- coding: utf-8 -*-

import os
import json
import hashlib
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
from .metrics import eval_all, compute_teacher_health, estimate_health_se
from .routing import (resolve_student_mode, mode_needs_teacher_logits,
                      rc_tcrd_masses,
                      certified_rc_tcrd_mode)
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
    precomputed_bundle: Optional[str] = None
    teacher_counts_source: str = "private"
    save_eval_predictions: bool = False
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
    hakd_weight_mode: str = "instance"

    health_weights: Tuple[float, float, float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

    health_power: float = 1.0
    health_floor: float = 0.0
    tcrd_tau_low: float = 0.45
    tcrd_tau_high: float = 0.60
    rc_rho: float = 0.10
    rc_fallback: str = "hard"

    out_dir: str = "outputs/runs"
    exp_id: str = "unknown"
    seed: int = 0
    sigma: float = 1.0
    run_tag: str = ""
    dev_limit: Optional[int] = None
    paired_hard_ckpt: Optional[str] = None


def _apply_if_not_none(cfg: RunCfg, attr: str, value):
    if value is not None:
        setattr(cfg, attr, value)


def _subsample_dev(feat, y_dev: np.ndarray, limit: Optional[int], seed: int):
    """Optionally use a deterministic class-stratified development subset."""
    if limit is None or int(limit) >= int(len(y_dev)):
        return feat, y_dev
    limit = max(int(np.unique(y_dev).size), int(limit))
    limit = min(limit, int(len(y_dev)))
    rng = np.random.default_rng(int(seed) + 4241)
    classes = sorted(int(c) for c in np.unique(y_dev))
    pools = {c: rng.permutation(np.flatnonzero(y_dev == c)).tolist() for c in classes}
    selected = []
    while len(selected) < limit:
        progressed = False
        for c in classes:
            if pools[c] and len(selected) < limit:
                selected.append(pools[c].pop())
                progressed = True
        if not progressed:
            break
    idx = np.asarray(sorted(selected), dtype=np.int64)
    feat = dict(feat)
    for backend, arrays in feat.items():
        X_priv, X_aux, X_dev, X_test = arrays
        feat[backend] = (X_priv, X_aux, X_dev[idx], X_test)
    return feat, y_dev[idx]


def _state_dict_sha256(model: torch.nn.Module) -> str:
    """Hash a model state dict for reproducible student-pair audits."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def run_once(cfg: RunCfg, device: torch.device) -> Dict[str, Any]:
    wall_start = time.perf_counter()
    set_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    os.makedirs(cfg.out_dir, exist_ok=True)

    feat_start = time.perf_counter()
    feat, (y_priv, y_aux, y_dev, y_test), class_names = build_features_and_splits(cfg, device)
    feat, y_dev = _subsample_dev(feat, y_dev, cfg.dev_limit, cfg.seed)
    feature_seconds = time.perf_counter() - feat_start
    num_classes = int(max(y_priv.max(initial=0), y_aux.max(initial=0), y_dev.max(initial=0), y_test.max(initial=0)) + 1)

    X_priv_t, X_aux_t, X_dev_t, X_test_t = feat[cfg.teacher_backend]

    priv_loader_t, _, dev_loader_t, test_loader_t = make_loaders(
        cfg, device,
        X_priv_t, y_priv,
        X_aux_t, y_aux,
        X_dev_t, y_dev,
        X_test_t, y_test,
        soft_aux=None,
    )

    if cfg.teacher_counts_source not in {"private", "aux"}:
        raise ValueError("teacher_counts_source must be private or aux")
    class_counts_priv = compute_class_counts(
        y_aux if cfg.teacher_counts_source == "aux" else y_priv,
        num_classes=num_classes)
    teacher = build_model(cfg, cfg.model_teacher, X_priv_t, num_classes=num_classes)

    tcache = teacher_cache_id(cfg)
    teacher_ckpt = os.path.join(cfg.out_dir, f"{tcache}_teacher.pt")
    teacher_info_path = os.path.join(cfg.out_dir, f"{tcache}_teacher_info.json")
    teacher_info: Dict[str, Any] = {}

    teacher_start = time.perf_counter()
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

        teacher_info["class_counts_source"] = cfg.teacher_counts_source
        teacher_info["class_counts"] = class_counts_priv.tolist()
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
        # Preserve the accountant attached to the trained teacher.  In
        # particular, Opacus uses 1 / len(loader), which is not generally
        # nominal_batch_size / dataset_size.  Student options cannot rewrite
        # the privacy metadata of a cached teacher.
        teacher_info["loaded_ckpt"] = teacher_ckpt
        print("[teacher-ckpt loaded]", teacher_ckpt)

    teacher_seconds = time.perf_counter() - teacher_start

    m_teacher_dev = eval_all(teacher, dev_loader_t, device, num_classes=num_classes, infer_adjust=None)
    m_teacher_test = eval_all(teacher, test_loader_t, device, num_classes=num_classes, infer_adjust=None)

    health_start = time.perf_counter()
    teacher_health = compute_teacher_health(
        m_teacher_dev,
        power=float(cfg.health_power),
        floor=float(cfg.health_floor),
        weights=cfg.health_weights,
    )
    teacher_global_health = float(teacher_health["global_health"])
    teacher_class_health = teacher_health["class_health"]
    health_se = estimate_health_se(
        m_teacher_dev,
        weights=cfg.health_weights,
        draws=2000,
        seed=int(cfg.seed) + 7919,
    )
    health_seconds = time.perf_counter() - health_start
    routing_masses = None
    sampled_mode = None
    routing_certified = False
    routing_fallback = False
    routing_cert_threshold = None
    routing_start = time.perf_counter()
    if str(cfg.student_mode).lower() in {"tcrd", "rc_tcrd"}:
        routing_masses = rc_tcrd_masses(
            teacher_global_health,
            health_se,
            tau_low=float(cfg.tcrd_tau_low),
            tau_high=float(cfg.tcrd_tau_high),
        )
        # TCRD uses deterministic thresholds. RC-TCRD applies the region
        # certificate and Hard fallback; it does not sample an action.
        if str(cfg.student_mode).lower() == "rc_tcrd":
            decision = certified_rc_tcrd_mode(
                teacher_global_health,
                health_se,
                tau_low=float(cfg.tcrd_tau_low),
                tau_high=float(cfg.tcrd_tau_high),
                rho=float(cfg.rc_rho),
                fallback=str(cfg.rc_fallback),
            )
            routing_masses = decision["masses"]
            sampled_mode = decision["mode"]
            routing_certified = bool(decision["certified"])
            routing_fallback = bool(decision["fallback"])
            routing_cert_threshold = float(decision["threshold"])
    routing_seconds = time.perf_counter() - routing_start

    teacher_info["teacher_health_dev"] = {
        "global_health": teacher_global_health,
        "class_health": teacher_class_health.tolist(),
        "standard_error": float(health_se),
        "weights": [float(x) for x in cfg.health_weights],
        "routing_masses": routing_masses,
        "sampled_mode": sampled_mode,
        "routing_certified": bool(routing_certified),
        "routing_fallback": bool(routing_fallback),
        "routing_cert_threshold": routing_cert_threshold,
        "routing_rho": float(cfg.rc_rho),
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

    student_start = time.perf_counter()
    if cfg.model_student is not None:
        X_priv_s, X_aux_s, X_dev_s, X_test_s = feat[cfg.student_backend]

        if sampled_mode is not None:
            effective_mode = sampled_mode
        else:
            effective_mode = resolve_student_mode(
                requested_mode=cfg.student_mode,
                teacher_global_health=teacher_global_health,
                tau_low=float(cfg.tcrd_tau_low),
                tau_high=float(cfg.tcrd_tau_high),
            )
        needs_teacher_logits = mode_needs_teacher_logits(effective_mode)
        soft_tag = _safe_tag(cfg.run_tag)
        soft_suffix = f"_tag{soft_tag}" if soft_tag else ""
        soft_path = os.path.join(cfg.out_dir, f"{tcache}{soft_suffix}_soft_logits_aux.npy")

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

        # Reset the student RNG independently of teacher training so that
        # Hard/KD/HAKD runs can share an auditable initialization and shuffle.
        student_seed = int(cfg.seed) + 1000003
        set_seed(student_seed)
        student_order_generator = torch.Generator(device="cpu")
        student_order_generator.manual_seed(student_seed + 1)
        student = build_model(cfg, cfg.model_student, X_aux_s, num_classes=num_classes)
        student_init_hash = _state_dict_sha256(student)

        if needs_teacher_logits:
            student_train_loader = DataLoader(
                NumpyDataset(X_aux_s, y_aux, soft_logits=soft_aux),
                batch_size=cfg.batch_size,
                shuffle=True,
                generator=student_order_generator,
                num_workers=cfg.num_workers,
                pin_memory=(cfg.pin_memory and device.type == "cuda"),
            )
        else:
            student_train_loader = DataLoader(
                NumpyDataset(X_aux_s, y_aux),
                batch_size=cfg.batch_size,
                shuffle=True,
                generator=student_order_generator,
                num_workers=cfg.num_workers,
                pin_memory=(cfg.pin_memory and device.type == "cuda"),
            )

        paired_reused = False
        if (
            str(cfg.student_mode).lower() in {"tcrd", "rc_tcrd"}
            and effective_mode == "hard"
            and cfg.paired_hard_ckpt
        ):
            if not os.path.exists(cfg.paired_hard_ckpt):
                raise FileNotFoundError(f"Paired hard student checkpoint not found: {cfg.paired_hard_ckpt}")
            student.load_state_dict(torch.load(cfg.paired_hard_ckpt, map_location="cpu"))
            student = student.to(device)
            paired_reused = True
            student_info = {
                "best_epoch": None,
                "best_dev_macro_f1": None,
                "requested_mode": str(cfg.student_mode),
                "effective_mode": "hard",
                "teacher_global_health": float(teacher_global_health),
                "class_health": [float(x) for x in teacher_class_health],
                "hakd_weight_mode": str(cfg.hakd_weight_mode),
                "student_init_seed": int(student_seed),
                "student_init_hash": student_init_hash,
                "student_loaded_hash": _state_dict_sha256(student),
            }
        else:
            student_info = train_student(
                student=student,
                train_loader=student_train_loader,
                dev_loader=dev_loader_s,
                device=device,
                num_classes=num_classes,
                epochs=cfg.epochs_student,
                lr=cfg.lr_student,
                weight_decay=cfg.weight_decay,
                student_mode=effective_mode,
                kd_alpha=cfg.kd_alpha,
                kd_temperature=cfg.kd_temperature,
                class_health=teacher_class_health,
                hakd_weight_mode=cfg.hakd_weight_mode,
                teacher_global_health=teacher_global_health,
                tcrd_tau_low=cfg.tcrd_tau_low,
                tcrd_tau_high=cfg.tcrd_tau_high,
            )

        m_student_dev = eval_all(student, dev_loader_s, device, num_classes=num_classes, infer_adjust=None)
        m_student_test = eval_all(student, test_loader_s, device, num_classes=num_classes, infer_adjust=None)

        student_info["requested_mode"] = str(cfg.student_mode)
        student_info["effective_mode"] = str(effective_mode)
        student_info["needs_teacher_logits"] = bool(needs_teacher_logits)
        student_info["paired_reused"] = bool(paired_reused)
        student_info.setdefault("student_init_seed", int(student_seed))
        student_info.setdefault("student_init_hash", student_init_hash)
        student_ckpt = os.path.join(cfg.out_dir, f"{tcache}{tag_suffix}_student_{effective_mode}.pt")
        torch.save(student.state_dict(), student_ckpt)
        student_info["student_ckpt"] = student_ckpt

    student_seconds = time.perf_counter() - student_start

    peak_cuda_mb = None
    if device.type == "cuda":
        peak_cuda_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2))

    out = dict(
        exp_id=cfg.exp_id,
        seed=int(cfg.seed),
        sigma=float(cfg.sigma),
        run_tag=str(cfg.run_tag),
        teacher_cache_id=str(tcache),
        cfg=asdict(cfg),
        class_names=class_names,
        teacher_info=teacher_info,
        teacher_dev=m_teacher_dev,
        teacher_test=m_teacher_test,
        student_info=student_info,
        student_dev=m_student_dev,
        student_test=m_student_test,
        routing={
            "requested_mode": str(cfg.student_mode),
            "health_se": float(health_se),
            "masses": routing_masses,
            "sampled_mode": sampled_mode,
            "certified": bool(routing_certified),
            "fallback": bool(routing_fallback),
            "cert_threshold": routing_cert_threshold,
            "rho": float(cfg.rc_rho),
        },
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        timing={
            "feature_seconds": float(feature_seconds),
            "teacher_seconds": float(teacher_seconds),
            "health_seconds": float(health_seconds),
            "routing_seconds": float(routing_seconds),
            "student_seconds": float(student_seconds),
            "total_seconds": float(time.perf_counter() - wall_start),
            "peak_cuda_allocated_mb": peak_cuda_mb,
        },
    )

    out_json = os.path.join(cfg.out_dir, f"{cfg.exp_id}_seed{cfg.seed}_sigma{cfg.sigma:g}{tag_suffix}.json")
    if cfg.save_eval_predictions:
        predictions = dict(y_dev=y_dev, y_test=y_test,
            teacher_dev=compute_soft_logits(teacher, dev_loader_t, device),
            teacher_test=compute_soft_logits(teacher, test_loader_t, device))
        if cfg.model_student is not None:
            predictions.update(student_dev=compute_soft_logits(student, dev_loader_s, device),
                student_test=compute_soft_logits(student, test_loader_s, device))
        prediction_path = out_json[:-5] + "_predictions.npz"
        np.savez_compressed(prediction_path, **predictions)
        out["predictions_file"] = os.path.basename(prediction_path)
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
    parser.add_argument("--label_col", type=str, default=None)
    parser.add_argument("--train_csv", type=str, default=None)
    parser.add_argument("--dev_csv", type=str, default=None)
    parser.add_argument("--test_csv", type=str, default=None)
    parser.add_argument("--train_folder", type=str, default=None)
    parser.add_argument("--dev_folder", type=str, default=None)
    parser.add_argument("--test_folder", type=str, default=None)
    parser.add_argument("--dev_limit", type=int, default=None)

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
    parser.add_argument("--ssl_model_name", type=str, default=None)

    parser.add_argument("--kd_alpha", type=float, default=None)
    parser.add_argument("--kd_temperature", type=float, default=None)
    parser.add_argument("--epochs_student", type=int, default=None)
    parser.add_argument("--lr_student", type=float, default=None)

    parser.add_argument("--student_mode", type=str, choices=["hard", "kd", "hakd", "difficulty_kd", "ctkd", "ctkd_global", "tcrd", "rc_tcrd"], default=None)
    parser.add_argument("--hakd_weight_mode", type=str, choices=["instance", "class", "uniform", "confidence"], default=None)
    parser.add_argument("--health_weights", type=float, nargs=3, default=None)
    parser.add_argument("--health_power", type=float, default=None)
    parser.add_argument("--health_floor", type=float, default=None)
    parser.add_argument("--tcrd_tau_low", type=float, default=None)
    parser.add_argument("--tcrd_tau_high", type=float, default=None)
    parser.add_argument("--rc_rho", type=float, default=None)
    parser.add_argument("--rc_fallback", type=str, choices=["hard"], default=None)
    parser.add_argument("--paired_hard_ckpt", type=str, default=None)

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
                    dev_limit=args.dev_limit,
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
                _apply_if_not_none(cfg, "label_col", args.label_col)
                _apply_if_not_none(cfg, "train_csv", args.train_csv)
                _apply_if_not_none(cfg, "dev_csv", args.dev_csv)
                _apply_if_not_none(cfg, "test_csv", args.test_csv)
                _apply_if_not_none(cfg, "train_folder", args.train_folder)
                _apply_if_not_none(cfg, "dev_folder", args.dev_folder)
                _apply_if_not_none(cfg, "test_folder", args.test_folder)
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
                _apply_if_not_none(cfg, "ssl_model_name", args.ssl_model_name)

                _apply_if_not_none(cfg, "kd_alpha", args.kd_alpha)
                _apply_if_not_none(cfg, "kd_temperature", args.kd_temperature)
                _apply_if_not_none(cfg, "epochs_student", args.epochs_student)
                _apply_if_not_none(cfg, "lr_student", args.lr_student)

                _apply_if_not_none(cfg, "student_mode", args.student_mode)
                _apply_if_not_none(cfg, "hakd_weight_mode", args.hakd_weight_mode)
                if args.health_weights is not None:
                    cfg.health_weights = tuple(float(x) for x in args.health_weights)
                _apply_if_not_none(cfg, "health_power", args.health_power)
                _apply_if_not_none(cfg, "health_floor", args.health_floor)
                _apply_if_not_none(cfg, "tcrd_tau_low", args.tcrd_tau_low)
                _apply_if_not_none(cfg, "tcrd_tau_high", args.tcrd_tau_high)
                _apply_if_not_none(cfg, "rc_rho", args.rc_rho)
                _apply_if_not_none(cfg, "rc_fallback", args.rc_fallback)
                _apply_if_not_none(cfg, "paired_hard_ckpt", args.paired_hard_ckpt)

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
