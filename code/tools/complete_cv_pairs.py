"""Complete paired student branches for the ten existing CV mel-BS teachers."""
import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.main import RunCfg, run_once


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teachers", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    valid = {f.name for f in fields(RunCfg)}
    for source in sorted(args.teachers.glob("CV_MEL_BS_WEAK_RELEASE_seed*_sigma*.json")):
        record = json.loads(source.read_text())
        for suffix in ("_teacher.pt", "_teacher_info.json"):
            original = args.teachers / (record["teacher_cache_id"] + suffix)
            target = args.out / original.name
            if not target.exists():
                target.symlink_to(original)
        for mode in ("hard", "kd", "hakd"):
            cfg = RunCfg(**{k: v for k, v in record["cfg"].items() if k in valid})
            cfg.train_teacher = False
            cfg.out_dir = str(args.out)
            cfg.student_mode = mode
            cfg.run_tag = "paired_20260907_" + mode
            dest = args.out / f"{cfg.exp_id}_seed{cfg.seed}_sigma{cfg.sigma:g}_tag{cfg.run_tag}.json"
            if dest.exists():
                print("Existing completed row", dest, flush=True)
                continue
            print("Paired branch", cfg.seed, cfg.sigma, mode, flush=True)
            run_once(cfg, torch.device("cuda"))


if __name__ == "__main__":
    main()
