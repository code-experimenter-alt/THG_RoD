"""Prepare a speaker-disjoint RAVDESS speech manifest for the TMM pipeline.

The source archive is the official Zenodo speech-only release.  This script
extracts only the 1,440 audio-only speech WAV files, preserves an all-label
manifest, and writes a four-label subset matching the legacy IEMOCAP diagnostic.
"""

import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path


LABELS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprise",
}
TARGET_LABELS = {"angry", "happy", "sad", "neutral"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_archive(archive: Path, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        root_resolved = root.resolve()
        for member in zf.infolist():
            target = (root / member.filename).resolve()
            if root_resolved not in target.parents and target != root_resolved:
                raise ValueError(f"unsafe archive member: {member.filename}")
        zf.extractall(root)


def build_rows(root: Path):
    rows = []
    for wav in sorted(root.glob("Actor_*/*.wav")):
        fields = wav.stem.split("-")
        if len(fields) != 7 or fields[0] != "03" or fields[1] != "01":
            continue
        emotion = LABELS.get(fields[2])
        actor = fields[6]
        if emotion is None or not actor.isdigit():
            continue
        rows.append({
            "filename": wav.relative_to(root).as_posix(),
            "emotion": emotion,
            "actor": f"Actor_{int(actor):02d}",
            "emotion_code": fields[2],
            "intensity_code": fields[3],
            "statement_code": fields[4],
            "repetition_code": fields[5],
        })
    if len(rows) != 1440:
        raise RuntimeError(f"expected 1440 speech WAVs, found {len(rows)}")
    return rows


def write_csv(path: Path, rows) -> None:
    fields = ["filename", "emotion", "actor", "emotion_code", "intensity_code", "statement_code", "repetition_code"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    archive = args.archive.resolve()
    root = args.root.resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    marker = root / ".ravdess_extracted"
    if not marker.exists():
        extract_archive(archive, root)
        marker.write_text("official Zenodo speech archive extracted\n", encoding="utf-8")

    rows = build_rows(root)
    selected = [r for r in rows if r["emotion"] in TARGET_LABELS]
    train = [r for r in selected if int(r["actor"].split("_")[1]) <= 16]
    dev = [r for r in selected if 17 <= int(r["actor"].split("_")[1]) <= 20]
    test = [r for r in selected if int(r["actor"].split("_")[1]) >= 21]
    for name, subset in (("all_manifest.csv", rows), ("four_class_manifest.csv", selected), ("train.csv", train), ("dev.csv", dev), ("test.csv", test)):
        write_csv(root / name, subset)

    metadata = {
        "source": "https://doi.org/10.5281/zenodo.1188976",
        "license": "CC BY-NC-SA 4.0",
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "rows": len(rows),
        "emotion_counts": dict(sorted(Counter(r["emotion"] for r in rows).items())),
        "target_labels": sorted(TARGET_LABELS),
        "target_rows": len(selected),
        "target_emotion_counts": dict(sorted(Counter(r["emotion"] for r in selected).items())),
        "split_actors": {"train": "Actor_01-Actor_16", "dev": "Actor_17-Actor_20", "test": "Actor_21-Actor_24"},
        "split_rows": {"train": len(train), "dev": len(dev), "test": len(test)},
        "audio_only": True,
        "speaker_disjoint": True,
    }
    (root / "manifest_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
