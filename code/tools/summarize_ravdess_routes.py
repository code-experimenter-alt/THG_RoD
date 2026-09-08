"""Summarize the completed three-seed RAVDESS route batch without retraining."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


MODES = ("hard", "kd", "hakd", "tcrd", "rc_tcrd")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    records = []
    for path in sorted(args.input.glob("*.json")):
        if path.name == "summary.json":
            continue
        obj = json.loads(path.read_text(encoding="utf-8"))
        run_tag = str(obj.get("run_tag", ""))
        mode = run_tag.removeprefix("ravdess_full_")
        if mode not in MODES:
            continue
        teacher = obj["teacher_info"]
        student = obj["student_test"]
        records.append({
            "seed": int(obj["seed"]),
            "mode": mode,
            "requested_mode": str(obj["student_info"]["requested_mode"]),
            "effective_mode": str(obj["student_info"]["effective_mode"]),
            "teacher_health_dev": float(teacher["teacher_health_dev"]["global_health"]),
            "teacher_epsilon": float(teacher["epsilon"]),
            "test_bal_acc": float(student["bal_acc"]),
            "test_macro_f1": float(student["macro_f1"]),
            "test_maj_pred": float(student["maj_pred"]),
            "test_acc": float(student["acc"]),
            "paired_reused": bool(obj["student_info"].get("paired_reused", False)),
            "class_names": "|".join(obj.get("class_names", [])),
            "source_root": str(obj["cfg"]["data_root"]),
            "split_mode": str(obj["cfg"]["split_mode"]),
            "sigma": float(obj["sigma"]),
            "json": path.name,
        })

    expected = {(seed, mode) for seed in (0, 1, 2) for mode in MODES}
    observed = {(r["seed"], r["mode"]) for r in records}
    if observed != expected:
        raise RuntimeError(f"expected 15 seed/mode rows, observed missing={sorted(expected-observed)} extra={sorted(observed-expected)}")

    hard_bal = {r["seed"]: r["test_bal_acc"] for r in records if r["mode"] == "hard"}
    for row in records:
        row["delta_vs_hard_bal_acc"] = row["test_bal_acc"] - hard_bal[row["seed"]]

    by_seed = defaultdict(list)
    for row in records:
        by_seed[row["seed"]].append(row)
    for seed, rows in by_seed.items():
        roots = {r["source_root"] for r in rows}
        classes = {r["class_names"] for r in rows}
        if len(roots) != 1 or len(classes) != 1 or any(r["split_mode"] != "official" for r in rows):
            raise RuntimeError(f"protocol mismatch in seed {seed}")
        for r in rows:
            if r["mode"] in {"tcrd", "rc_tcrd"} and r["effective_mode"] == "hard" and not r["paired_reused"]:
                raise RuntimeError(f"hard route is not paired in seed {seed}")

    fields = list(records[0].keys())
    csv_path = args.out / "ravdess_full_routes_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda r: (r["seed"], MODES.index(r["mode"]))))

    aggregate = []
    for mode in MODES:
        rows = [r for r in records if r["mode"] == mode]
        aggregate.append({
            "mode": mode,
            "effective_modes": dict(Counter(r["effective_mode"] for r in rows)),
            "bal_acc_mean": sum(r["test_bal_acc"] for r in rows) / len(rows),
            "bal_acc_sd": (sum((r["test_bal_acc"] - sum(x["test_bal_acc"] for x in rows) / len(rows)) ** 2 for r in rows) / (len(rows) - 1)) ** 0.5,
            "macro_f1_mean": sum(r["test_macro_f1"] for r in rows) / len(rows),
            "maj_pred_mean": sum(r["test_maj_pred"] for r in rows) / len(rows),
        })
    metadata = {
        "rows": len(records),
        "seeds": sorted(by_seed),
        "modes": list(MODES),
        "aggregate": aggregate,
        "protocol": {
            "dataset": "RAVDESS speech",
            "source_root": records[0]["source_root"],
            "split_mode": records[0]["split_mode"],
            "class_names": records[0]["class_names"].split("|"),
            "hard_selected_route_rows_paired_reused": all(
                r["paired_reused"] for r in records
                if r["mode"] in {"tcrd", "rc_tcrd"} and r["effective_mode"] == "hard"
            ),
            "not_iemocap": True,
        },
    }
    (args.out / "ravdess_full_routes_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    lines = [
        "# RAVDESS 三 seed 完整路由汇总",
        "",
        "该报告只汇总已完成的 RAVDESS speech 运行，不把结果称为 IEMOCAP。来源为",
        "官方 Zenodo speech release（CC BY-NC-SA 4.0）；四类标签顺序为 angry/happy/neutral/sad。",
        "所有路由均使用相同 official speaker split；当 TCRD/RC-TCRD 选择 hard 时复用同 seed 的 Hard checkpoint。",
        "",
        "## 逐 seed 结果",
        "",
        "| seed | mode | effective | teacher H(dev) | test BalAcc | Δ vs Hard | test Macro-F1 | test MajPred | ε | paired |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in sorted(records, key=lambda x: (x["seed"], MODES.index(x["mode"]))):
        lines.append(
            f"| {r['seed']} | {r['mode']} | {r['effective_mode']} | {r['teacher_health_dev']:.4f} | "
            f"{r['test_bal_acc']:.4f} | {r['delta_vs_hard_bal_acc']:+.4f} | {r['test_macro_f1']:.4f} | {r['test_maj_pred']:.4f} | "
            f"{r['teacher_epsilon']:.4f} | {str(r['paired_reused']).lower()} |"
        )
    lines.extend([
        "",
        "## 三 seed 均值",
        "",
        "| mode | effective modes | BalAcc mean±SD | Macro-F1 mean | MajPred mean |",
        "|---|---|---:|---:|---:|",
    ])
    for a in aggregate:
        lines.append(
            f"| {a['mode']} | {', '.join(f'{k}:{v}' for k, v in a['effective_modes'].items())} | "
            f"{a['bal_acc_mean']:.4f} ± {a['bal_acc_sd']:.4f} | {a['macro_f1_mean']:.4f} | {a['maj_pred_mean']:.4f} |"
        )
    lines.extend([
        "",
        "## 解释边界",
        "",
        "三 seed 中 RC-TCRD 均回退/选择 Hard；TCRD 两次选择 Hard、一次选择 HAKD。该批次的"
        " RAVDESS 结果用于补充跨数据集路由可运行性与风险对照，不替代 USC 授权的 IEMOCAP"
        " 原始数据，也不支持 TCRD 普遍优于简单路由的结论。",
    ])
    (args.out / "ravdess_full_routes_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
