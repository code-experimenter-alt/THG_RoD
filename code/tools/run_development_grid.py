"""Paired TCRD/RC-TCRD grid with the complete development budget varied."""
import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.main import RunCfg, run_once
from src.routing import certified_rc_tcrd_mode, resolve_student_mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    valid = {f.name for f in fields(RunCfg)}
    records = []
    for n in (30, 60, 120):
        for seed in range(3):
            source = args.source / f"CV_MEL_BS_WEAK_RELEASE_seed{seed}_sigma1_tagstrictpair_bs_s{seed}_hard.json"
            base_cfg = json.loads(source.read_text())["cfg"]
            branches = {}
            for mode in ("hard", "kd", "hakd"):
                cfg = RunCfg(**{k:v for k,v in base_cfg.items() if k in valid})
                cfg.dev_limit = n
                cfg.out_dir = str(args.out / f"dev{n}")
                cfg.student_mode = mode
                cfg.train_teacher = mode == "hard"
                cfg.run_tag = "devgrid_" + mode
                cfg.paired_hard_ckpt = None
                dest = Path(cfg.out_dir) / f"{cfg.exp_id}_seed{seed}_sigma1_tag{cfg.run_tag}.json"
                if not dest.exists():
                    print("Development grid", n, seed, mode, flush=True)
                    run_once(cfg, torch.device("cuda"))
                branches[mode] = json.loads(dest.read_text())
            assert len({r["student_info"]["student_init_hash"] for r in branches.values()}) == 1
            h = branches["hard"]["teacher_info"]["teacher_health_dev"]["global_health"]
            se = branches["hard"]["routing"]["health_se"]
            for lo,hi in ((0.45,0.60),(0.15,0.25),(0.25,0.35),(0.30,0.40)):
                mode = resolve_student_mode("tcrd", h, lo,hi)
                rc = certified_rc_tcrd_mode(h,se,lo,hi,0.1,"hard")
                records.append(dict(n_dev=n,seed=seed,tau_low=lo,tau_high=hi,health=h,se=se,
                                    tcrd_mode=mode,rc=rc,
                                    tcrd_test=branches[mode]["student_test"],
                                    rc_test=branches[rc["mode"]]["student_test"],
                                    hard_test=branches["hard"]["student_test"]))
    summaries=[]
    for n in (30,60,120):
        for lo,hi in ((0.45,0.60),(0.15,0.25),(0.25,0.35),(0.30,0.40)):
            rows=[r for r in records if r["n_dev"]==n and r["tau_low"]==lo]
            summaries.append(dict(n_dev=n,tau_low=lo,tau_high=hi,
                tcrd_balacc=float(np.mean([r["tcrd_test"]["bal_acc"] for r in rows])),
                rc_balacc=float(np.mean([r["rc_test"]["bal_acc"] for r in rows])),
                tcrd_modes=[r["tcrd_mode"] for r in rows],rc_modes=[r["rc"]["mode"] for r in rows],
                certified=sum(r["rc"]["certified"] for r in rows)))
    (args.out/"routing_grid.json").write_text(json.dumps({"rows":records,"summary":summaries},indent=2))
    print(json.dumps(summaries,indent=2))


if __name__=="__main__":
    main()
