"""Add the published global-T mechanism under cached speech teachers."""
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
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--pattern", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", default="ctg")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    valid = {f.name for f in fields(RunCfg)}
    sources = sorted(args.source.glob(args.pattern))
    if not sources:
        raise ValueError("No source runs match the requested pattern")
    for source in sources:
        record = json.loads(source.read_text())
        for suffix in ("_teacher.pt", "_teacher_info.json"):
            original = args.source / (record["teacher_cache_id"] + suffix)
            if not original.exists():
                raise FileNotFoundError(original)
            dest = args.out / original.name
            if not dest.exists():
                dest.symlink_to(original.resolve())
        cfg = RunCfg(**{k:v for k,v in record["cfg"].items() if k in valid})
        cfg.train_teacher = False
        cfg.out_dir = str(args.out)
        cfg.student_mode = "ctkd_global"
        cfg.run_tag = args.tag
        cfg.paired_hard_ckpt = None
        result = args.out / f"{cfg.exp_id}_seed{cfg.seed}_sigma{cfg.sigma:g}_tag{cfg.run_tag}.json"
        if not result.exists():
            run_once(cfg, torch.device("cuda"))
        actual = json.loads(result.read_text())
        assert actual["teacher_cache_id"] == record["teacher_cache_id"]
        assert actual["student_info"]["student_init_hash"] == record["student_info"]["student_init_hash"]
        assert actual["teacher_info"]["epsilon"] == record["teacher_info"]["epsilon"]
        trace = actual["student_info"]["ctkd"]["trace"]
        assert any(abs(v["raw_gradient"]) > 1e-10 for v in trace)
        assert abs(actual["student_info"]["ctkd"]["raw_final"] - 1) > 1e-7
        print("Paired CTKD complete", cfg.seed, cfg.sigma, actual["student_test"]["bal_acc"], flush=True)


if __name__ == "__main__":
    main()
