"""Build the fixed, checked feature bundle for the authorized CV replacement."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import extract_mel_features, extract_ssl_embeddings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--backend", choices=["mel", "ssl"], required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    frames = {s: pd.read_csv(args.manifest / f"{s}_filtered.tsv", sep="\t")
              for s in ("train", "dev", "test")}
    names = ["england", "indian", "us"]
    for s, df in frames.items():
        assert len(df) == dict(train=6714, dev=1061, test=459)[s]
        assert df.path.is_unique and df.client_id.notna().all()
        for t, other in frames.items():
            if s != t:
                assert set(df.path).isdisjoint(other.path)
                assert set(df.client_id).isdisjoint(other.client_id)
        np.save(args.out / f"{s}_labels.npy", df.accent.map(dict(zip(names, range(3)))).to_numpy(np.int64))
        df[["path", "client_id", "accent"]].to_csv(args.out / f"{s}_manifest.csv", index=False)
    seeds = {}
    for seed in range(5):
        tr = frames["train"]
        private, auxiliary = next(GroupShuffleSplit(n_splits=1, test_size=.25,
            random_state=seed).split(tr, groups=tr.client_id))
        assert set(tr.iloc[private].client_id).isdisjoint(tr.iloc[auxiliary].client_id)
        seeds[str(seed)] = dict(private=private.tolist(), auxiliary=auxiliary.tolist())
    split_record = dict(class_names=names, seeds=seeds,
        source="https://zenodo.org/records/12588635", protocol="cv_primary_replacement_20260907")
    (args.out / "splits.json").write_text(json.dumps(split_record, indent=2))
    for s, df in frames.items():
        target = args.out / f"{s}_{args.backend}.npy"
        if target.exists():
            arr = np.load(target)
            assert len(arr) == len(df) and np.isfinite(arr).all()
            assert np.all(np.any(arr.reshape(len(df), -1) != 0, axis=1))
            print("Completed", target, flush=True)
            continue
        paths = []
        for value in df.path:
            path = args.audio / str(value)
            if not path.exists():
                path = args.audio / (str(value) + ".mp3")
            assert path.is_file(), path
            paths.append(str(path))
        if args.backend == "ssl":
            arr = extract_ssl_embeddings(paths, args.model, "mean", 16000,
                torch.device("cuda"), max_seconds=12.0, strict=True)
        else:
            arr = extract_mel_features(paths, 40, 100, True, 1e-5)
        assert len(arr) == len(df) and np.isfinite(arr).all()
        assert np.all(np.any(arr.reshape(len(df), -1) != 0, axis=1))
        np.save(target, arr)
        print("Saved", target, arr.shape, flush=True)
    (args.out / f"{args.backend}_complete.json").write_text(json.dumps(dict(
        backend=args.backend, rows=sum(map(len, frames.values())), errors=0,
        model=(args.model if args.backend == "ssl" else None),
        revision=("0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8" if args.backend == "ssl" else None)), indent=2))


if __name__ == "__main__":
    main()
