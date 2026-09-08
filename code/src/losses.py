# -*- coding: utf-8 -*-

from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F


def compute_class_counts(y: np.ndarray, num_classes: int) -> np.ndarray:
    cnt = np.bincount(y.astype(np.int64), minlength=num_classes).astype(np.float64)
    cnt = np.maximum(cnt, 1.0)
    return cnt


def loss_balanced_softmax(logits: torch.Tensor, y: torch.Tensor, class_counts: torch.Tensor):
    """Balanced Softmax training loss, using empirical class counts."""
    class_counts = class_counts.to(logits.device).view(1, -1)
    log_counts = torch.log(class_counts)
    log_denom = torch.logsumexp(logits + log_counts, dim=1)
    z_y = logits.gather(1, y.view(-1, 1)).squeeze(1)
    z_y = z_y + log_counts.expand_as(logits).gather(1, y.view(-1, 1)).squeeze(1)
    return (log_denom - z_y).mean()


def loss_logit_adjust(
    logits: torch.Tensor,
    y: torch.Tensor,
    class_priors: torch.Tensor,
    tau: float,
    return_adjust: bool = False,
):
    """Standard *training-time* logit adjustment.

    The adjusted logits are ``z + tau * log(pi)``.  With ``tau=1`` and
    ``pi`` proportional to the class counts, this differs from Balanced
    Softmax only by the sample-independent constant ``-log(sum(counts))``.
    Post-hoc inference adjustment is kept separate in
    :func:`make_infer_adjust_from_counts`.
    """
    pri = class_priors.to(logits.device).view(1, -1).clamp_min(1e-12)
    adjust = float(tau) * torch.log(pri)
    loss = F.cross_entropy(logits + adjust, y)
    if return_adjust:
        return loss, adjust.detach()
    return loss


def make_infer_adjust_from_counts(
    class_counts_np: np.ndarray,
    tau: float,
    device: torch.device
) -> Optional[torch.Tensor]:
    tau = float(tau)
    if tau <= 0.0:
        return None

    cc = torch.tensor(class_counts_np, dtype=torch.float32, device=device).view(1, -1).clamp_min(1.0)
    priors = cc / cc.sum(dim=1, keepdim=True)
    adjust = -tau * torch.log(priors.clamp_min(1e-12))
    return adjust


def kd_loss(student_logits, teacher_logits, y, alpha: float, T: float):
    hard = F.cross_entropy(student_logits, y)
    p_t = F.softmax(teacher_logits / T, dim=1)
    log_p_s = F.log_softmax(student_logits / T, dim=1)
    soft = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)
    return alpha * soft + (1.0 - alpha) * hard


def difficulty_kd_loss(student_logits, teacher_logits, y, alpha: float,
                       T: float, gamma: float = 1.0):
    """Uncertainty-weighted CE/KD probe, not published DLKD reproduction.

    Published DLKD uses original-versus-pruned teacher predictions. This
    historical probe instead weights CE by teacher predictive uncertainty.
    """
    hard = F.cross_entropy(student_logits, y, reduction="none")
    p_t = F.softmax(teacher_logits / T, dim=1)
    log_p_s = F.log_softmax(student_logits / T, dim=1)
    soft = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=1) * (T * T)
    difficulty = (1.0 - p_t.max(dim=1).values).detach()
    weights = 1.0 + float(gamma) * difficulty
    hard_weighted = (weights * hard).mean() / weights.mean().clamp_min(1e-12)
    return alpha * soft.mean() + (1.0 - alpha) * hard_weighted, {
        "difficulty_mean": float(difficulty.mean().item()),
        "difficulty_weight_mean": float(weights.mean().item()),
    }


def curriculum_temperature_kd_loss(student_logits, teacher_logits, y,
                                    alpha: float, base_T: float,
                                    progress: float):
    """Historical linear-temperature probe; not learned-temperature CTKD.

    The original ``ctkd`` mode is preserved for old run compatibility. Use
    ``ctkd_global`` for the adversarial global-temperature CTKD mechanism.
    """
    progress = float(np.clip(progress, 0.0, 1.0))
    T_eff = float(base_T) * (1.0 + progress)
    return kd_loss(student_logits, teacher_logits, y, alpha=alpha, T=T_eff), T_eff


def hakd_y_loss(student_logits, teacher_logits, y, alpha: float, T: float,
                class_health: torch.Tensor, weight_mode: str = "instance"):
    hard = F.cross_entropy(student_logits, y, reduction="none")

    p_t = F.softmax(teacher_logits / T, dim=1)
    log_p_s = F.log_softmax(student_logits / T, dim=1)
    soft = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=1) * (T * T)

    health = class_health.to(student_logits.device).view(1, -1)
    mode = str(weight_mode).lower()
    if mode == "instance":
        reliability = (p_t * health).sum(dim=1)
    elif mode == "class":
        reliability = health.expand(student_logits.shape[0], -1).gather(1, y.view(-1, 1)).squeeze(1)
    elif mode in {"uniform", "confidence"}:
        reliability = torch.ones_like(hard) if mode == "uniform" else p_t.max(dim=1).values
    else:
        raise ValueError(f"Unknown HAKD weight_mode: {weight_mode}")
    lam = torch.clamp(alpha * reliability, 0.0, 1.0)

    loss = ((1.0 - lam) * hard + lam * soft).mean()
    info = {
        "lambda_mean": float(lam.mean().item()),
        "lambda_min": float(lam.min().item()),
        "lambda_max": float(lam.max().item()),
    }
    return loss, info
