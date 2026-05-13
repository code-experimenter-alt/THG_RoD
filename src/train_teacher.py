# -*- coding: utf-8 -*-

import json
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

from .losses import loss_balanced_softmax, loss_logit_adjust
from .metrics import eval_all


def _safe_tag(tag: str) -> str:
    tag = (tag or "").strip()
    if not tag:
        return ""
    keep = []
    for ch in tag:
        if ch.isalnum() or ch in ("-", "_", ".", "=", "+"):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def unwrap_opacus_model(m: nn.Module) -> nn.Module:
    return m._module if hasattr(m, "_module") else m


def teacher_cache_id(cfg) -> str:
    if cfg.teacher_backend == "ssl":
        feat_id = f"ssl_{cfg.ssl_model_name}_{cfg.ssl_pooling}_sr{cfg.target_sr}_mx{cfg.ssl_max_seconds}"
    else:
        feat_id = f"mel_sr{cfg.target_sr}_m{cfg.n_mels}_L{cfg.max_time}_dsaf{int(cfg.use_dsaf)}_eta{cfg.eta0:g}"

    group = (cfg.group_col or "auto")
    ratios = ",".join([f"{x:g}" for x in cfg.ratios])

    parts = [
        f"seed{cfg.seed}",
        f"sigma{cfg.sigma:g}",
        f"split{cfg.split_mode}",
        f"group{group}",
        f"ratios{ratios}",
        f"tback{cfg.teacher_backend}",
        f"tmodel{cfg.model_teacher}",
        f"feat{feat_id}",
        f"dp{int(cfg.dp_teacher)}",
        f"C{cfg.max_grad_norm:g}",
        f"loss{cfg.loss_name}",
        f"tau{cfg.logit_adjust_tau:g}",
        f"ept{cfg.epochs_teacher}",
        f"lrt{cfg.lr_teacher:g}",
        f"wd{cfg.weight_decay:g}",
        f"bs{cfg.batch_size}",
    ]
    return _safe_tag("_".join(parts))


def train_teacher(
    model: nn.Module,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    dp: bool,
    sigma: float,
    max_grad_norm: float,
    delta: float,
    loss_name: str,
    class_counts_np: np.ndarray,
    logit_adjust_tau: float = 1.0,
    record_grad_stats: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    model = model.to(device)
    model = ModuleValidator.fix(model)
    ModuleValidator.validate(model, strict=False)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    privacy_engine = None

    if dp:
        privacy_engine = PrivacyEngine()
        model, opt, dp_train_loader = privacy_engine.make_private(
            module=model,
            optimizer=opt,
            data_loader=train_loader,
            noise_multiplier=float(sigma),
            max_grad_norm=float(max_grad_norm),
        )
        train_loader_use = dp_train_loader
    else:
        train_loader_use = train_loader

    class_counts = torch.tensor(class_counts_np, dtype=torch.float32)
    priors = class_counts / class_counts.sum()

    def per_sample_grad_norm(m: nn.Module) -> Optional[torch.Tensor]:
        norms2 = None
        for p in m.parameters():
            gs = getattr(p, "grad_sample", None)
            if gs is None:
                continue
            gs = gs.reshape(gs.shape[0], -1)
            n2 = (gs.norm(2, dim=1) ** 2)
            norms2 = n2 if norms2 is None else (norms2 + n2)
        return None if norms2 is None else norms2.sqrt()

    grad_stats = {"clip_rate": [], "gnorm_p90": [], "gnorm_p99": []}
    best = {"dev_macro_f1": -1.0, "state": None, "epoch": -1}

    for ep in range(epochs):
        model.train()
        epoch_clip = []
        epoch_p90 = []
        epoch_p99 = []

        for xb, yb in train_loader_use:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)

            if loss_name == "ce":
                loss = F.cross_entropy(logits, yb)
            elif loss_name == "balanced_softmax":
                loss = loss_balanced_softmax(logits, yb, class_counts)
            elif loss_name == "logit_adjust":
                loss = loss_logit_adjust(logits, yb, priors, tau=float(logit_adjust_tau))
            else:
                raise ValueError(f"Unknown loss_name: {loss_name}")

            loss.backward()

            if dp and record_grad_stats:
                g = per_sample_grad_norm(model)
                if g is not None:
                    epoch_clip.append((g > float(max_grad_norm)).float().mean().item())
                    epoch_p90.append(torch.quantile(g, 0.90).item())
                    epoch_p99.append(torch.quantile(g, 0.99).item())

            opt.step()

        m_dev = eval_all(model, dev_loader, device, num_classes=num_classes)
        eps = float(privacy_engine.get_epsilon(delta)) if privacy_engine is not None else 0.0
        print(f"[teacher] ep {ep+1}/{epochs} dev_macro_f1={m_dev['macro_f1']:.4f} dev_bal_acc={m_dev['bal_acc']:.4f} eps={eps:.4f}")

        if dp and record_grad_stats and len(epoch_clip) > 0:
            grad_stats["clip_rate"].append(float(np.mean(epoch_clip)))
            grad_stats["gnorm_p90"].append(float(np.mean(epoch_p90)))
            grad_stats["gnorm_p99"].append(float(np.mean(epoch_p99)))

        if m_dev["macro_f1"] > best["dev_macro_f1"]:
            best["dev_macro_f1"] = m_dev["macro_f1"]
            best["epoch"] = ep + 1
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best["state"] is not None:
        model.load_state_dict(best["state"])

    final_eps = float(privacy_engine.get_epsilon(delta)) if privacy_engine is not None else 0.0
    info = dict(
        dp=bool(dp),
        sigma=float(sigma),
        epsilon=float(final_eps),
        delta=float(delta),
        best_epoch=int(best["epoch"]),
        best_dev_macro_f1=float(best["dev_macro_f1"]),
        grad_stats=(grad_stats if (dp and record_grad_stats) else None),
    )
    return model, info


@torch.no_grad()
def compute_soft_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    outs = []
    for xb, _yb in loader:
        xb = xb.to(device)
        logits = model(xb).detach().cpu().numpy().astype(np.float32)
        outs.append(logits)
    return np.concatenate(outs, axis=0) if outs else np.zeros((0, 0), dtype=np.float32)