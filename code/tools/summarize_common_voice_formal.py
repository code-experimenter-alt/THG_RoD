"""Summarize and validate Common Voice formal weak-teacher JSON runs."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


def load_rows(root: Path):
    rows = []
    for path in sorted(root.glob("CV_MEL_BS_WEAK_RELEASE_seed*_sigma*.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        classes = obj.get("class_names")
        if classes != ["england", "indian", "us"]:
            raise ValueError(f"{path.name}: unexpected class_names={classes}")
        cfg = obj.get("cfg", {})
        rows.append({
            "file": path.name,
            "seed": int(obj["seed"]),
            "sigma": float(obj["sigma"]),
            "teacher_dev_bal_acc": float(obj["teacher_dev"]["bal_acc"]),
            "teacher_dev_macro_f1": float(obj["teacher_dev"]["macro_f1"]),
            "teacher_dev_maj_pred": float(obj["teacher_dev"]["maj_pred"]),
            "student_test_bal_acc": float(obj["student_test"]["bal_acc"]),
            "student_test_macro_f1": float(obj["student_test"]["macro_f1"]),
            "student_test_maj_pred": float(obj["student_test"]["maj_pred"]),
            "requested_mode": str(obj["routing"]["requested_mode"]),
            "certified": bool(obj["routing"]["certified"]),
            "data_root": str(cfg.get("data_root", "")),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--out_json", type=Path, required=True)
    args = parser.parse_args()
    rows = load_rows(args.input_dir)
    expected = {(sigma, seed) for sigma in (1.0, 1.5) for seed in range(5)}
    observed = {(row["sigma"], row["seed"]) for row in rows}
    missing = sorted(expected - observed)
    if missing:
        raise RuntimeError(f"missing formal runs: {missing}")
    if len(observed) != len(rows):
        raise RuntimeError("duplicate seed/sigma result files")
    summaries = []
    metrics = ("teacher_dev_bal_acc", "student_test_bal_acc", "student_test_macro_f1", "student_test_maj_pred")
    for sigma in (1.0, 1.5):
        group = [row for row in rows if row["sigma"] == sigma]
        summary = {"sigma": sigma, "n": len(group)}
        for metric in metrics:
            values = [row[metric] for row in group]
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_sd"] = stdev(values) if len(values) > 1 else 0.0
        summary["certified_count"] = sum(row["certified"] for row in group)
        summary["class_names"] = ["england", "indian", "us"]
        summaries.append(summary)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    args.out_json.write_text(json.dumps({"runs": rows, "summaries": summaries}, indent=2), encoding="utf-8")
    print(json.dumps({"run_count": len(rows), "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
