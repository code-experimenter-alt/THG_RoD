"""Repair the feature-release protocol using one record per utterance.

Input: the existing, row-traceable pooled Zenodo feature files.  Training
utterance variants are averaged; dev/test source utterances occur once.
Splits use the actual utterance speaker suffix, not the dialogue prefix.
"""
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np


def speaker_of(uid):
    match = re.fullmatch(r"(Ses\d\d)[FM]_.+_([FM])\d+", uid)
    if match is None:
        raise ValueError(f"Unrecognized IEMOCAP utterance ID: {uid}")
    return match.group(1) + match.group(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pooled", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    grouped = defaultdict(list)
    for source in ("train", "dev", "test"):
        with np.load(args.pooled / f"{source}.npz", allow_pickle=False) as data:
            features, labels, ids = data["X"], data["y"], data["utterance_id"]
            for idx, uid in enumerate(ids):
                grouped[str(uid)].append((features[idx], int(labels[idx]), source, idx))
    route_speakers = {
        "private": [f"Ses{s:02d}{sex}" for s in (1, 2, 3) for sex in "FM"],
        "auxiliary": ["Ses04F", "Ses04M"],
        "dev": ["Ses05F"],
        "test": ["Ses05M"],
    }
    speaker_routes = {s: route for route, ss in route_speakers.items() for s in ss}
    pools = defaultdict(list)
    provenance = []
    for uid, variants in sorted(grouped.items()):
        labels = {v[1] for v in variants}
        sources = {v[2] for v in variants}
        if len(labels) != 1 or len(sources) != 1:
            raise ValueError(f"Inconsistent source/label for {uid}")
        speaker = speaker_of(uid)
        route = speaker_routes[speaker]
        feature = np.mean(np.stack([v[0] for v in variants]), axis=0, dtype=np.float64).astype(np.float32)
        pools[route].append((feature, labels.pop(), uid, speaker))
        provenance.append({"utterance_id": uid, "speaker_id": speaker,
                           "route_split": route, "source_split": variants[0][2],
                           "source_row_indices": ";".join(str(v[3]) for v in variants),
                           "variants_averaged": len(variants)})
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = []
    for output, routes in (("train", ("private", "auxiliary")), ("dev", ("dev",)), ("test", ("test",))):
        items = [(row, route) for route in routes for row in pools[route]]
        np.savez(args.out / f"{output}.npz", X=np.stack([x[0][0] for x in items]),
                 y=np.asarray([x[0][1] for x in items]),
                 utterance_id=np.asarray([x[0][2] for x in items]),
                 speaker_id=np.asarray([x[0][3] for x in items]))
        for idx, (row, route) in enumerate(items):
            manifest.append(dict(source_split=output, row_index=idx, utterance_id=row[2],
                                 speaker_id=row[3], label=row[1], route_split=route))
    for name, rows in (("manifest.csv", manifest), ("utterance_provenance.csv", provenance)):
        with (args.out / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    sets = {route: {r[2] for r in rows} for route, rows in pools.items()}
    assert all(not sets[a] & sets[b] for a, b in combinations(sets, 2))
    assert all(set(r[1] for r in rows) == {0, 1, 2, 3} for rows in pools.values())
    metadata = {
        "source": "https://doi.org/10.5281/zenodo.17803295",
        "source_license": "CC-BY-4.0", "pooled_source": str(args.pooled),
        "record_unit": "one unique utterance; mean of available pooled variants",
        "source_training_rows_without_id_excluded": 4424,
        "retained_source_rows": len([v for vv in grouped.values() for v in vv]),
        "unique_utterances": len(grouped),
        "variant_multiplicity": dict(Counter(len(v) for v in grouped.values())),
        "split_speakers": route_speakers,
        "split_sizes": {k: len(v) for k, v in pools.items()},
        "split_class_counts": {k: dict(Counter(r[1] for r in v)) for k, v in pools.items()},
        "pairwise_speaker_and_utterance_disjoint": True,
        "selection": "fixed sessions 1-3 private, session 4 auxiliary, session 5 F dev/M test; no utility-based selection",
    }
    (args.out / "manifest_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
