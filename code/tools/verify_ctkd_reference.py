"""Compare temperature, loss and gradients against the downloaded official code."""
import argparse
import ast
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ctkd import GlobalTemperature, curriculum_magnitude
from src.losses import kd_loss


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    official_t = module_from_file("official_ctkd_global", args.reference / "models/temp_global.py")
    official_kd = module_from_file("official_ctkd_kl", args.reference / "distiller_zoo/KD.py")
    # Load only the published schedule, not the benchmark's CLI/training imports.
    tree = ast.parse((args.reference / "train_student.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CosineDecay")
    scope = {"math": math}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "official_CosineDecay", "exec"), scope)
    schedule = scope["CosineDecay"](max_value=0, min_value=-1, num_loops=10)
    reference = official_t.Global_T().double().cuda()
    ours = GlobalTemperature().double().cuda()
    torch.manual_seed(734)
    errors = []
    for epoch in (1, 5, 10, 15):
        reference.zero_grad()
        ours.zero_grad()
        logits = torch.randn(7, 4, dtype=torch.float64, device="cuda")
        a, b = logits.clone().requires_grad_(), logits.clone().requires_grad_()
        teacher = torch.randn_like(a)
        target = torch.arange(7, device="cuda") % 4
        ref_temp = 1 + 20 * torch.sigmoid(reference(teacher, a, schedule.get_value(epoch)))
        temp = ours(epoch)
        ref_loss = .7 * official_kd.DistillKL()(a, teacher, ref_temp) + .3 * torch.nn.functional.cross_entropy(a, target)
        loss = kd_loss(b, teacher, target, .7, temp)
        ref_loss.backward()
        loss.backward()
        differences = dict(temperature=float((temp-ref_temp).abs().max().item()),
            loss=float((loss-ref_loss).abs().max().item()),
            student_gradient=float((a.grad-b.grad).abs().max().item()),
            temperature_gradient=float((reference.global_T.grad-ours.raw.grad).abs().max().item()),
            schedule=abs(curriculum_magnitude(epoch)+schedule.get_value(epoch)))
        assert max(differences.values()) < 1e-10, differences
        errors.append(dict(epoch=epoch, **differences))
    commit = subprocess.check_output(["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True).strip()
    report = dict(source="https://github.com/zhengli97/CTKD", commit=commit,
        checks=errors, precision="float64, CUDA", passed=True,
        scope="global-T temperature, cosine gradient reversal, CE/KL objective and gradients; not benchmark accuracy reproduction")
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
