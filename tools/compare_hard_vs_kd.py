#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import json
import argparse
import pandas as pd


def load_rows(input_dir: str) -> pd.DataFrame:
    rows = []
    for fp in glob.glob(os.path.join(input_dir, "*.json")):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                j = json.load(f)
        except Exception:
            continue

        cfg = j.get("cfg", {})
        rows.append({
            "file": os.path.basename(fp),
            "timestamp": j.get("timestamp"),
            "exp_id": j.get("exp_id"),
            "seed": j.get("seed"),
            "sigma": j.get("sigma"),
            "run_tag": j.get("run_tag"),
            "teacher_cache_id": j.get("teacher_cache_id"),

            "train_teacher": cfg.get("train_teacher"),
            "dp_teacher": cfg.get("dp_teacher"),
            "loss_name": cfg.get("loss_name"),
            "max_grad_norm": cfg.get("max_grad_norm"),

            "teacher_backend": cfg.get("teacher_backend"),
            "student_backend": cfg.get("student_backend"),
            "model_student": cfg.get("model_student"),
            "student_mode": cfg.get("student_mode"),
            "kd_alpha": cfg.get("kd_alpha"),
            "kd_temperature": cfg.get("kd_temperature"),

            "teacher_dev_macro_f1": (j.get("teacher_dev") or {}).get("macro_f1"),
            "teacher_test_macro_f1": (j.get("teacher_test") or {}).get("macro_f1"),
            "student_dev_macro_f1": (j.get("student_dev") or {}).get("macro_f1"),
            "student_test_macro_f1": (j.get("student_test") or {}).get("macro_f1"),

            "teacher_dev_bal_acc": (j.get("teacher_dev") or {}).get("bal_acc"),
            "teacher_test_bal_acc": (j.get("teacher_test") or {}).get("bal_acc"),
            "student_dev_bal_acc": (j.get("student_dev") or {}).get("bal_acc"),
            "student_test_bal_acc": (j.get("student_test") or {}).get("bal_acc"),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--hard_tag", type=str, required=True)
    parser.add_argument("--kd_tag", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    args = parser.parse_args()

    df = load_rows(args.input_dir)
    print("all json runs:", len(df))

    d2 = df[df["run_tag"].isin([args.hard_tag, args.kd_tag])].copy()
    print("selected runs:", len(d2))

    cov = (
        d2.groupby(["exp_id", "seed", "sigma"])["run_tag"]
        .apply(lambda s: set(s.tolist()))
        .reset_index(name="tags_present")
    )
    cov["has_hard"] = cov["tags_present"].apply(lambda s: args.hard_tag in s)
    cov["has_kd"] = cov["tags_present"].apply(lambda s: args.kd_tag in s)
    cov["complete_pair"] = cov["has_hard"] & cov["has_kd"]

    missing = cov[~cov["complete_pair"]]
    print("missing pairs:", len(missing))
    if len(missing):
        print(missing)

    hard = d2[d2["run_tag"] == args.hard_tag].copy()
    kd = d2[d2["run_tag"] == args.kd_tag].copy()

    key = ["exp_id", "seed", "sigma"]
    hard = hard.sort_values("timestamp").drop_duplicates(key, keep="last")
    kd = kd.sort_values("timestamp").drop_duplicates(key, keep="last")

    pair = hard.merge(kd, on=key, suffixes=("_hard", "_kd"), how="inner")
    pair["teacher_fixed"] = pair["teacher_cache_id_hard"] == pair["teacher_cache_id_kd"]

    for m in [
        "student_dev_macro_f1",
        "student_test_macro_f1",
        "student_dev_bal_acc",
        "student_test_bal_acc",
    ]:
        pair[f"delta_{m}"] = pair[f"{m}_kd"] - pair[f"{m}_hard"]

    cols = key + [
        "teacher_fixed",
        "teacher_cache_id_hard", "teacher_cache_id_kd",
        "teacher_dev_macro_f1_hard", "teacher_test_macro_f1_hard",
        "student_dev_macro_f1_hard", "student_dev_macro_f1_kd", "delta_student_dev_macro_f1",
        "student_test_macro_f1_hard", "student_test_macro_f1_kd", "delta_student_test_macro_f1",
        "student_dev_bal_acc_hard", "student_dev_bal_acc_kd", "delta_student_dev_bal_acc",
        "student_test_bal_acc_hard", "student_test_bal_acc_kd", "delta_student_test_bal_acc",
    ]
    pair_view = pair[cols].sort_values(key)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    pair_view.to_csv(args.out_csv, index=False)
    print("saved:", args.out_csv)

    bad = pair_view[~pair_view["teacher_fixed"]]
    print("teacher not fixed pairs:", len(bad))
    if len(bad):
        print(bad)


if __name__ == "__main__":
    main()