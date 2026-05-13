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
            entropy_mean=0.0, pmax_mean=0.0
        )

    acc = float((p == y).mean())
    bal_acc = float(balanced_accuracy_score(y, p))
    macro_f1 = float(f1_score(y, p, average="macro"))
    counts = np.bincount(p, minlength=num_classes)
    maj_pred = float(counts.max() / len(p))
    per_class_recall = recall_score(y, p, average=None, labels=list(range(num_classes))).astype(float).tolist()

    return dict(
        acc=acc,
        bal_acc=bal_acc,
        macro_f1=macro_f1,
        maj_pred=maj_pred,
        per_class_recall=per_class_recall,
        pred_counts=counts.astype(int).tolist(),
        entropy_mean=float(ent.mean()),
        pmax_mean=float(pmax.mean()),
    )


def compute_teacher_health(metrics: Dict[str, Any], power: float = 1.0, floor: float = 0.0):
    recalls = np.asarray(metrics["per_class_recall"], dtype=np.float32)
    recalls = np.clip(recalls, 0.0, 1.0)

    power = float(power)
    floor = float(floor)

    if power != 1.0:
        recalls = np.power(recalls, power)
    if floor > 0.0:
        recalls = np.maximum(recalls, floor)

    H_global = (
        float(metrics["bal_acc"])
        + float(metrics["macro_f1"])
        + (1.0 - float(metrics["maj_pred"]))
    ) / 3.0

    return {
        "global_health": float(H_global),
        "class_health": recalls.astype(np.float32),
    }