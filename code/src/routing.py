# -*- coding: utf-8 -*-

import numpy as np


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
    return str(effective_mode).lower() in {"kd", "hakd", "difficulty_kd", "ctkd", "ctkd_global"}


def rc_tcrd_masses(health: float, health_se: float,
                   tau_low: float = 0.45, tau_high: float = 0.60):
    """Posterior routing masses under the normal health approximation."""
    from math import erf, sqrt

    sd = max(float(health_se), 1e-12)

    def cdf(x):
        return 0.5 * (1.0 + erf((float(x) - float(health)) / (sd * sqrt(2.0))))

    q_hard = cdf(tau_low)
    q_hakd = max(0.0, cdf(tau_high) - q_hard)
    q_kd = max(0.0, 1.0 - cdf(tau_high))
    q = np.asarray([q_hard, q_hakd, q_kd], dtype=np.float64)
    q /= max(float(q.sum()), 1e-12)
    return {"hard": float(q[0]), "hakd": float(q[1]), "kd": float(q[2])}


def sample_rc_tcrd_mode(masses, seed: int):
    names = ("hard", "hakd", "kd")
    p = np.asarray([float(masses[n]) for n in names], dtype=np.float64)
    p = np.clip(p, 0.0, None)
    p /= max(float(p.sum()), 1e-12)
    rng = np.random.default_rng(int(seed))
    return names[int(rng.choice(len(names), p=p))]


def certified_rc_tcrd_mode(
    health: float,
    health_se: float,
    tau_low: float = 0.45,
    tau_high: float = 0.60,
    rho: float = 0.10,
    fallback: str = "hard",
):
    """Choose a mode only when its health-region mass is certified.

    ``q_m`` is the normal-approximation mass that the teacher health lies in
    mode ``m``'s interval.  The returned ``certified`` flag is true only when
    the selected mode satisfies ``q_m >= 1-rho``.  If no mode qualifies, the
    declared fallback is used and remains uncertified.
    """
    rho = float(rho)
    if not 0.0 < rho < 1.0:
        raise ValueError(f"rho must be in (0, 1), got {rho}")
    fallback = str(fallback).lower()
    if fallback != "hard":
        raise ValueError("Only the explicit hard fallback is supported")

    masses = rc_tcrd_masses(health, health_se, tau_low=tau_low, tau_high=tau_high)
    threshold = 1.0 - rho
    eligible = [name for name in ("hard", "hakd", "kd") if masses[name] >= threshold]
    if not eligible:
        return {
            "mode": fallback,
            "certified": False,
            "fallback": True,
            "threshold": threshold,
            "masses": masses,
        }

    mode = max(eligible, key=lambda name: (masses[name], -("hard", "hakd", "kd").index(name)))
    return {
        "mode": mode,
        "certified": True,
        "fallback": False,
        "threshold": threshold,
        "masses": masses,
    }
