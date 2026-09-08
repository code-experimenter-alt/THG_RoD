"""Pool Zenodo IEMOCAP WavLM pickles as input to prepare_iemocap_disjoint.py.

The source files are feature-level derivatives, not the USC raw release.  The
script preserves row order, labels, and utterance IDs. Its intermediate speaker
metadata uses the source filename prefix; the paper's suffix-based speaker
split and per-utterance variant pooling are applied by prepare_iemocap_disjoint.py.
It never changes or overwrites the downloaded pickle files.
"""

import argparse
import csv
import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


LABELS = (0, 1, 2, 3)
SPEAKER_RE = re.compile(r"^(Ses\d+[FM])")


def read_pool(path: Path):
    frame = pd.read_pickle(path)
    required = {"utterance_id", "label", "features", "length"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    rows = []
    ids = []
    labels = []
    speakers = []
    lengths = []
    skipped_missing_id = []
    for idx, row in frame.iterrows():
        if pd.isna(row["utterance_id"]):
            skipped_missing_id.append(int(idx))
            continue
        arr = np.asarray(row["features"], dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 1024:
            raise ValueError(f"{path} row {idx} has feature shape {arr.shape}")
        n = int(row["length"])
        n = max(1, min(n, arr.shape[0]))
        rows.append(arr[:n].mean(axis=0, dtype=np.float32))
        uid = str(row["utterance_id"])
        match = SPEAKER_RE.match(uid)
        if match is None:
            raise ValueError(f"Cannot derive speaker from utterance_id={uid}")
        ids.append(uid)
        labels.append(int(row["label"]))
        speakers.append(match.group(1))
        lengths.append(n)
    x = np.stack(rows).astype(np.float32, copy=False)
    y = np.asarray(labels, dtype=np.int64)
    return x, y, ids, speakers, lengths, skipped_missing_id


def choose_private_speakers(speakers, labels, private_ratio=0.6):
    unique = sorted(set(speakers))
    if len(unique) < 2:
        raise ValueError("Need at least two speakers for a split")
    best = None
    for size in range(1, len(unique)):
        for combo in combinations(unique, size):
            priv = set(combo)
            priv_labels = {int(y) for s, y in zip(speakers, labels) if s in priv}
            aux_labels = {int(y) for s, y in zip(speakers, labels) if s not in priv}
            if set(LABELS).difference(priv_labels) or set(LABELS).difference(aux_labels):
                continue
            frac = sum(s in priv for s in speakers) / max(1, len(speakers))
            score = (abs(frac - private_ratio), abs(size - len(unique) * private_ratio), combo)
            if best is None or score < best[0]:
                best = (score, priv)
    if best is None:
        raise ValueError("No speaker split gives all four labels in both private and auxiliary sets")
    return sorted(best[1])


def write_npz(path: Path, x, y, ids, speakers, lengths):
    np.savez(path, X=x, y=y, utterance_id=np.asarray(ids), speaker_id=np.asarray(speakers), length=np.asarray(lengths, dtype=np.int64))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pooled = {}
    skipped = {}
    for split, path in (("train", args.train), ("dev", args.dev), ("test", args.test)):
        x, y, ids, speakers, lengths, skipped_ids = read_pool(path)
        pooled[split] = (x, y, ids, speakers, lengths)
        skipped[split] = skipped_ids
        write_npz(args.out / f"{split}.npz", x, y, ids, speakers, lengths)

    train_x, train_y, train_ids, train_speakers, train_lengths = pooled["train"]
    private_speakers = set(choose_private_speakers(train_speakers, train_y))
    rows = []
    fields = ["source_split", "row_index", "utterance_id", "speaker_id", "label", "route_split"]
    for split, (_x, y, ids, speakers, _lengths) in pooled.items():
        for idx, (uid, speaker, label) in enumerate(zip(ids, speakers, y)):
            route_split = "private" if split == "train" and speaker in private_speakers else "auxiliary" if split == "train" else split
            rows.append({"source_split": split, "row_index": idx, "utterance_id": uid, "speaker_id": speaker, "label": int(label), "route_split": route_split})
    with (args.out / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "source": "https://doi.org/10.5281/zenodo.17803295",
        "source_license": "CC BY 4.0",
        "feature": "WavLM-large, mean pooling over stored length, 1024 dimensions",
        "raw_derivative_not_usc_release": True,
        "split_rows": {k: int(len(v[1])) for k, v in pooled.items()},
        "split_label_counts": {k: dict(sorted(Counter(map(int, v[1])).items())) for k, v in pooled.items()},
        "train_speakers": sorted(set(train_speakers)),
        "private_speakers": sorted(private_speakers),
        "auxiliary_speakers": sorted(set(train_speakers).difference(private_speakers)),
        "speaker_disjoint_private_auxiliary": True,
        "label_values": sorted(set(map(int, train_y))),
        "skipped_missing_utterance_id": skipped,
    }
    (args.out / "manifest_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
