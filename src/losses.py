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
    class_counts = class_counts.to(logits.device).view(1, -1)
    log_denom = torch.logsumexp(logits + torch.log(class_counts), dim=1)
    z_y = logits.gather(1, y.view(-1, 1)).squeeze(1)
    return (log_denom - z_y).mean()


def loss_logit_adjust(
    logits: torch.Tensor,
    y: torch.Tensor,
    class_priors: torch.Tensor,
    tau: float,
    return_adjust: bool = False,
):
    pri = class_priors.to(logits.device).view(1, -1).clamp_min(1e-12)
    adjust = -float(tau) * torch.log(pri)
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


def hakd_y_loss(student_logits, teacher_logits, y, alpha: float, T: float, class_health: torch.Tensor):
    hard = F.cross_entropy(student_logits, y, reduction="none")

    p_t = F.softmax(teacher_logits / T, dim=1)
    log_p_s = F.log_softmax(student_logits / T, dim=1)
    soft = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=1) * (T * T)

    h_y = class_health.to(student_logits.device)[y]
    lam = torch.clamp(alpha * h_y, 0.0, 1.0)

    loss = ((1.0 - lam) * hard + lam * soft).mean()
    info = {
        "lambda_mean": float(lam.mean().item()),
        "lambda_min": float(lam.min().item()),
        "lambda_max": float(lam.max().item()),
    }
    return loss, info