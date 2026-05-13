# -*- coding: utf-8 -*-

from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F

from .losses import kd_loss, hakd_y_loss
from .metrics import eval_all
from .routing import resolve_student_mode


def train_student(
    student,
    train_loader,
    dev_loader,
    device: torch.device,
    num_classes: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    student_mode: str,
    kd_alpha: float,
    kd_temperature: float,
    class_health: Optional[np.ndarray] = None,
    teacher_global_health: Optional[float] = None,
    tcrd_tau_low: float = 0.45,
    tcrd_tau_high: float = 0.60,
) -> Dict[str, Any]:
    student = student.to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)

    effective_mode = resolve_student_mode(
        requested_mode=student_mode,
        teacher_global_health=float(teacher_global_health if teacher_global_health is not None else 1.0),
        tau_low=float(tcrd_tau_low),
        tau_high=float(tcrd_tau_high),
    )

    class_health_t = None
    if class_health is not None:
        class_health_t = torch.tensor(class_health, dtype=torch.float32, device=device)

    print(
        f"[student] requested_mode={student_mode} effective_mode={effective_mode} "
        f"teacher_global_health={float(teacher_global_health if teacher_global_health is not None else -1.0):.4f}"
    )

    best = {"dev_macro_f1": -1.0, "state": None, "epoch": -1}
    hakd_stats = []

    for ep in range(epochs):
        student.train()

        for batch in train_loader:
            if effective_mode == "hard":
                xb, yb = batch
                xb, yb = xb.to(device), yb.to(device)
                slogits = student(xb)
                loss = F.cross_entropy(slogits, yb)

            elif effective_mode == "kd":
                xb, yb, tlogits = batch
                xb, yb, tlogits = xb.to(device), yb.to(device), tlogits.to(device)
                slogits = student(xb)
                loss = kd_loss(slogits, tlogits, yb, alpha=float(kd_alpha), T=float(kd_temperature))

            elif effective_mode == "hakd":
                if class_health_t is None:
                    raise ValueError("class_health is required for HAKD.")

                xb, yb, tlogits = batch
                xb, yb, tlogits = xb.to(device), yb.to(device), tlogits.to(device)
                slogits = student(xb)
                loss, hakd_info = hakd_y_loss(
                    slogits, tlogits, yb,
                    alpha=float(kd_alpha),
                    T=float(kd_temperature),
                    class_health=class_health_t,
                )
                hakd_stats.append(hakd_info["lambda_mean"])
            else:
                raise ValueError(f"Unknown effective student mode: {effective_mode}")

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        m_dev = eval_all(student, dev_loader, device, num_classes=num_classes)
        print(f"[student] ep {ep+1}/{epochs} mode={effective_mode} dev_macro_f1={m_dev['macro_f1']:.4f} dev_bal_acc={m_dev['bal_acc']:.4f}")

        if m_dev["macro_f1"] > best["dev_macro_f1"]:
            best["dev_macro_f1"] = m_dev["macro_f1"]
            best["epoch"] = ep + 1
            best["state"] = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}

    if best["state"] is not None:
        student.load_state_dict(best["state"])

    out = dict(
        best_epoch=int(best["epoch"]),
        best_dev_macro_f1=float(best["dev_macro_f1"]),
        requested_mode=str(student_mode),
        effective_mode=str(effective_mode),
        teacher_global_health=float(teacher_global_health if teacher_global_health is not None else -1.0),
        class_health=None if class_health is None else [float(x) for x in class_health],
    )

    if len(hakd_stats) > 0:
        out["hakd_lambda_mean_train"] = float(np.mean(hakd_stats))

    return out