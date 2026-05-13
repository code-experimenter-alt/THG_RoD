# -*- coding: utf-8 -*-


def resolve_student_mode(requested_mode: str, teacher_global_health: float,
                         tau_low: float = 0.45, tau_high: float = 0.60) -> str:
    requested_mode = str(requested_mode).lower()
    if requested_mode != "tcrd":
        return requested_mode

    h = float(teacher_global_health)
    if h < float(tau_low):
        return "hard"
    elif h < float(tau_high):
        return "hakd"
    else:
        return "kd"


def mode_needs_teacher_logits(effective_mode: str) -> bool:
    return str(effective_mode).lower() in {"kd", "hakd"}