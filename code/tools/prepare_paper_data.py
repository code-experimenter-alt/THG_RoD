"""Locate the exact recorded audio files and write portable experiment manifests."""
import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd


def locate_audio(root, value, dataset):
    value = Path(str(value))
    name = value.name
    if dataset == "small-cv" and not value.suffix:
        name += ".mp3"
    candidates = [root / value, root / name, root / "clips" / name]
    if dataset == "vctk":
        speaker = name.split("_")[0]
        candidates += [root / "audio" / speaker / name, root / speaker / name,
                       root / "wav48_silence_trimmed" / speaker / name]
    found = next((p.resolve() for p in candidates if p.is_file()), None)
    if found is None:
        raise FileNotFoundError(f"Missing recorded audio: {value} under {root}")
    return str(found)


def prepare(dataset, manifests, audio_root, out):
    frames = {}
    for split in ("train", "dev", "test"):
        name = f"{split}.csv" if dataset == "vctk" else f"{split}_manifest.csv"
        frames[split] = pd.read_csv(manifests / name)
    expected = (480, 120, 120) if dataset == "vctk" else (6714, 1061, 459)
    assert tuple(map(len, frames.values())) == expected
    path_key, group_key = (("filename", "speaker_id") if dataset == "vctk"
                           else ("path", "client_id"))
    for frame in frames.values():
        assert frame[path_key].is_unique and frame[group_key].notna().all()
        frame["record_id"] = frame[path_key].astype(str)
        frame[path_key] = [locate_audio(audio_root, p, dataset) for p in frame[path_key]]
        assert frame[path_key].is_unique
    for a, b in combinations(frames.values(), 2):
        assert set(a[path_key]).isdisjoint(b[path_key])
        assert set(a[group_key]).isdisjoint(b[group_key])
    out.mkdir(parents=True, exist_ok=True)
    for split, frame in frames.items():
        name = f"{split}.csv" if dataset == "vctk" else f"{split}_filtered.tsv"
        destination = out / name
        if destination.exists():
            raise FileExistsError(f"Use a new manifest directory: {destination}")
        frame.to_csv(destination, index=False, sep="," if dataset == "vctk" else "\t")
    print(f"{dataset}: {expected}; every audio file found; disjoint speakers and clips.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("vctk", "small-cv"), required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.dataset, args.manifests, args.audio_root, args.out)


if __name__ == "__main__":
    main()
