# -*- coding: utf-8 -*-
"""Repeatable VCTK timing benchmark for KD/HAKD/TCRD/RC-TCRD."""

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = "/home/fu/tmmrr_vctk_subset"
CACHE_DIR = "/home/fu/tmmrr_cache/vctk_timing_bench"
OUT_DIR = "/home/fu/tmmrr_results/vctk_timing_bench"
SEEDS = (0, 1, 2)
MODES = ("kd", "hakd", "tcrd", "rc_tcrd")


def invoke(mode, seed, tag, train_teacher, epochs_student):
    cmd = [
        sys.executable,
        "-m",
        "src.main",
        "--exp_id",
        "CV_MEL_BS_WEAK_RELEASE",
        "--data_root",
        DATA_ROOT,
        "--cache_dir",
        CACHE_DIR,
        "--out_dir",
        OUT_DIR,
        "--train_csv",
        "train.csv",
        "--dev_csv",
        "dev.csv",
        "--test_csv",
        "test.csv",
        "--train_folder",
        ".",
        "--dev_folder",
        ".",
        "--test_folder",
        ".",
        "--label_col",
        "accent",
        "--group_col",
        "speaker_id",
        "--seeds",
        str(seed),
        "--sigmas",
        "1.0",
        "--tag",
        tag,
        "--train_teacher",
        str(int(train_teacher)),
        "--epochs_teacher",
        "4",
        "--epochs_student",
        str(epochs_student),
        "--batch_size",
        "64",
        "--n_mels",
        "32",
        "--max_time",
        "80",
        "--use_dsaf",
        "1",
        "--student_mode",
        mode,
        "--dev_limit",
        "120",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    output = Path(OUT_DIR)
    return output / f"CV_MEL_BS_WEAK_RELEASE_seed{seed}_sigma1_tag{tag}.json"


def main():
    records = []
    seen_seed = set()
    for mode in MODES:
        for seed in SEEDS:
            tag = f"timing_{mode}_s{seed}"
            path = invoke(mode, seed, tag, True, 5)
            payload = json.loads(path.read_text(encoding="utf-8"))
            row = {"mode": mode, "seed": seed, "phase": "cold_initial" if seed not in seen_seed else "warm_training"}
            row.update(payload["timing"])
            records.append(row)
            seen_seed.add(seed)

    for mode in MODES:
        for repeat in range(10):
            tag = f"timing_{mode}_cheap{repeat:02d}"
            path = invoke(mode, 0, tag, False, 0)
            payload = json.loads(path.read_text(encoding="utf-8"))
            row = {"mode": mode, "seed": 0, "phase": f"warm_cheap_{repeat + 1:02d}"}
            row.update(payload["timing"])
            records.append(row)

    out_csv = Path(OUT_DIR) / "timing_benchmark.csv"
    fields = [
        "mode",
        "seed",
        "phase",
        "feature_seconds",
        "teacher_seconds",
        "health_seconds",
        "routing_seconds",
        "student_seconds",
        "total_seconds",
        "peak_cuda_allocated_mb",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({"rows": len(records), "output": str(out_csv)}))


if __name__ == "__main__":
    main()
