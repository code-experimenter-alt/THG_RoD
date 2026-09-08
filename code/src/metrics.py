# -*- coding: utf-8 -*-

from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def eval_all(model, loader, device, num_classes: int, infer_adjust: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    from sklearn.metrics import f1_score, balanced_accuracy_score, recall_score

    model.eval()
    ys, ps = [], []
    entropies = []
    maxps = []

    if infer_adjust is not None:
        infer_adjust = infer_adjust.to(device)

    for batch in loader:
        xb, yb = batch[0].to(device), batch[1]
        logits = model(xb)

        if infer_adjust is not None:
            logits = logits + infer_adjust

        probs = F.softmax(logits, dim=1).cpu().numpy()
        pred = probs.argmax(axis=1)

        ys.append(yb.numpy())
        ps.append(pred)

        pmax = probs.max(axis=1)
        maxps.append(pmax)
        ent = -(probs * (np.log(probs + 1e-12))).sum(axis=1)
        entropies.append(ent)

    y = np.concatenate(ys) if ys else np.array([], dtype=np.int64)
    p = np.concatenate(ps) if ps else np.array([], dtype=np.int64)
    ent = np.concatenate(entropies) if entropies else np.array([], dtype=np.float32)
    pmax = np.concatenate(maxps) if maxps else np.array([], dtype=np.float32)

    if y.size == 0:
        return dict(
            acc=0.0, bal_acc=0.0, macro_f1=0.0, maj_pred=0.0,
            per_class_recall=[0.0] * num_classes,
            pred_counts=[0] * num_classes,
            confusion=[[0] * num_classes for _ in range(num_classes)],
            entropy_mean=0.0, pmax_mean=0.0
        )

    acc = float((p == y).mean())
    bal_acc = float(balanced_accuracy_score(y, p))
    macro_f1 = float(f1_score(y, p, average="macro"))
    counts = np.bincount(p, minlength=num_classes)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (y, p), 1)
    maj_pred = float(counts.max() / len(p))
    per_class_recall = recall_score(y, p, average=None, labels=list(range(num_classes))).astype(float).tolist()

    return dict(
        acc=acc,
        bal_acc=bal_acc,
        macro_f1=macro_f1,
        maj_pred=maj_pred,
        per_class_recall=per_class_recall,
        pred_counts=counts.astype(int).tolist(),
        confusion=confusion.tolist(),
        entropy_mean=float(ent.mean()),
        pmax_mean=float(pmax.mean()),
    )


def compute_teacher_health(metrics: Dict[str, Any], power: float = 1.0, floor: float = 0.0,
                           weights=(1.0 / 3.0,) * 3):
    recalls = np.asarray(metrics["per_class_recall"], dtype=np.float32)
    recalls = np.clip(recalls, 0.0, 1.0)

    power = float(power)
    floor = float(floor)

    if power != 1.0:
        recalls = np.power(recalls, power)
    if floor > 0.0:
        recalls = np.maximum(recalls, floor)

    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (3,) or np.any(w < 0.0) or float(w.sum()) <= 0.0:
        raise ValueError("health weights must be three non-negative values with positive sum")
    w = w / float(w.sum())
    components = np.asarray([
        float(metrics["bal_acc"]),
        float(metrics["macro_f1"]),
        1.0 - float(metrics["maj_pred"]),
    ], dtype=np.float64)
    H_global = float(np.dot(w, components))

    return {
        "global_health": float(H_global),
        "class_health": recalls.astype(np.float32),
    }


def estimate_health_se(metrics: Dict[str, Any], weights=(1.0 / 3.0,) * 3,
                       draws: int = 2000, seed: int = 0) -> float:
    """Parametric multinomial standard error for the composite health score."""
    conf = np.asarray(metrics.get("confusion", []), dtype=np.int64)
    if conf.ndim != 2 or conf.size == 0 or conf.sum() <= 1:
        return 0.0
    k = conf.shape[0]
    n = int(conf.sum())
    probs = (conf.reshape(-1) / float(n)).astype(np.float64)
    rng = np.random.default_rng(int(seed))
    samples = rng.multinomial(n, probs, size=max(100, int(draws))).reshape(-1, k, k)
    hs = []
    w = np.asarray(weights, dtype=np.float64)
    w = w / max(w.sum(), 1e-12)
    for c in samples:
        row = c.sum(axis=1)
        col = c.sum(axis=0)
        rec = np.divide(np.diag(c), row, out=np.zeros(k), where=row > 0)
        prec = np.divide(np.diag(c), col, out=np.zeros(k), where=col > 0)
        f1 = np.divide(2 * prec * rec, prec + rec,
                       out=np.zeros(k), where=(prec + rec) > 0)
        bal = float(rec.mean())
        mf1 = float(f1.mean())
        maj = float(col.max() / max(1, n))
        hs.append(w[0] * bal + w[1] * mf1 + w[2] * (1.0 - maj))
    return float(np.std(np.asarray(hs), ddof=1))
