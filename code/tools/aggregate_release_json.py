"""Aggregate release-stage JSON records into a deterministic summary."""

import argparse
import json
from pathlib import Path
from statistics import mean, stdev


def _numeric(values):
    return [float(value) for value in values if isinstance(value, (int, float))]


def aggregate(input_dir: Path):
    records = []
    for path in sorted(input_dir.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            payload = {**payload, "_file": path.name}
            records.append(payload)
    summary = {"record_count": len(records), "files": [r["_file"] for r in records]}
    for key in ("bal_acc", "macro_f1", "health"):
        values = _numeric([record.get(key) for record in records])
        if values:
            summary[f"{key}_mean"] = mean(values)
            summary[f"{key}_sd"] = stdev(values) if len(values) > 1 else 0.0
    return {"summary": summary, "records": records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.input_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "release_summary.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
