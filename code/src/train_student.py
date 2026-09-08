# -*- coding: utf-8 -*-

from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F

from .losses import (kd_loss, hakd_y_loss, difficulty_kd_loss,
                     curriculum_temperature_kd_loss)
from .metrics import eval_all
from .routing import resolve_student_mode
from .ctkd import GlobalTemperature, curriculum_magnitude


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
    hakd_weight_mode: str = "instance",
    teacher_global_health: Optional[float] = None,
    tcrd_tau_low: float = 0.45,
    tcrd_tau_high: float = 0.60,
) -> Dict[str, Any]:
    student = student.to(device)

    effective_mode = resolve_student_mode(
        requested_mode=student_mode,
        teacher_global_health=float(teacher_global_health if teacher_global_health is not None else 1.0),
        tau_low=float(tcrd_tau_low),
        tau_high=float(tcrd_tau_high),
    )

    temperature_model = GlobalTemperature().to(device) if effective_mode == "ctkd_global" else None
    parameters = list(student.parameters())
    if temperature_model is not None:
        parameters += list(temperature_model.parameters())
    opt = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    temperature_trace = []
    selection_trace = []

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

            elif effective_mode == "difficulty_kd":
                xb, yb, tlogits = batch
                xb, yb, tlogits = xb.to(device), yb.to(device), tlogits.to(device)
                slogits = student(xb)
                loss, difficulty_info = difficulty_kd_loss(
                    slogits, tlogits, yb, alpha=float(kd_alpha),
                    T=float(kd_temperature), gamma=1.0,
                )

            elif effective_mode == "ctkd":
                xb, yb, tlogits = batch
                xb, yb, tlogits = xb.to(device), yb.to(device), tlogits.to(device)
                slogits = student(xb)
                loss, curriculum_temperature = curriculum_temperature_kd_loss(
                    slogits, tlogits, yb, alpha=float(kd_alpha),
                    base_T=float(kd_temperature),
                    progress=float(ep + 1) / max(int(epochs), 1),
                )

            elif effective_mode == "ctkd_global":
                xb, yb, tlogits = batch
                xb, yb, tlogits = xb.to(device), yb.to(device), tlogits.to(device)
                slogits = student(xb)
                learned_temperature = temperature_model(ep + 1)
                loss = kd_loss(slogits, tlogits, yb, alpha=float(kd_alpha), T=learned_temperature)
                temperature_trace.append(dict(epoch=ep + 1,
                    temperature=float(learned_temperature.detach().item()),
                    gradient_reversal_magnitude=curriculum_magnitude(ep + 1)))

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
                    weight_mode=str(hakd_weight_mode),
                )
                hakd_stats.append(hakd_info["lambda_mean"])
            else:
                raise ValueError(f"Unknown effective student mode: {effective_mode}")

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if temperature_model is not None:
                temperature_trace[-1]["raw_gradient"] = float(temperature_model.raw.grad.item())
            opt.step()

        m_dev = eval_all(student, dev_loader, device, num_classes=num_classes)
        selection_trace.append(dict(epoch=ep + 1, macro_f1=float(m_dev["macro_f1"]),
                                    bal_acc=float(m_dev["bal_acc"])))
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
        hakd_weight_mode=str(hakd_weight_mode),
        selection_trace=selection_trace,
    )

    if len(hakd_stats) > 0:
        out["hakd_lambda_mean_train"] = float(np.mean(hakd_stats))

    if temperature_model is not None:
        out["ctkd"] = dict(variant="global", source_commit="56112892d5aca069bd56bb057feee9f7ff9e4141",
            initial_raw=1.0, temperature_offset=1.0, temperature_range=20.0,
            cosine_ramp_epochs=10, optimizer="AdamW, joint with student; matched speech protocol",
            raw_final=float(temperature_model.raw.detach().item()), trace=temperature_trace)

    return out
