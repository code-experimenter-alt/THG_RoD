"""Create an auditable Common Voice accent manifest from official TSV splits."""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


ACCENT_TO_LABEL = {"us": 0, "england": 1, "indian": 2}


def load_split(root: Path, name: str):
    path = root / f"{name}.tsv"
    if not path.exists():
        path = root / f"{name}.csv"
    sep = "\t" if path.suffix == ".tsv" else ","
    frame = pd.read_csv(path, sep=sep)
    required = {"client_id", "path", "accent"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["accent_norm"] = frame["accent"].astype("string").str.strip().str.lower()
    frame["label"] = frame["accent_norm"].map(ACCENT_TO_LABEL)
    frame["source_row"] = range(len(frame))
    frame["source_split"] = name
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--aux_ratio", type=float, default=0.25)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    frames = {name: load_split(args.root, name) for name in ("train", "dev", "test")}
    kept = {name: frame[frame["label"].notna()].copy().reset_index(drop=True) for name, frame in frames.items()}
    for name, frame in kept.items():
        filtered = frame.drop(columns=["accent_norm", "label", "source_row", "source_split"], errors="ignore").copy()
        filtered["accent"] = frame["accent_norm"].astype(str).to_numpy()
        filtered.to_csv(args.out / f"{name}_filtered.tsv", sep="\t", index=False)
    train = kept["train"]
    groups = train["client_id"].astype(str).to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.aux_ratio, random_state=args.seed)
    priv_idx, aux_idx = next(splitter.split(train, groups=groups))
    private_clients = set(train.iloc[priv_idx]["client_id"].astype(str))
    aux_clients = set(train.iloc[aux_idx]["client_id"].astype(str))

    rows = []
    for name, frame in kept.items():
        for _, item in frame.iterrows():
            client = str(item["client_id"])
            route = "private" if name == "train" and client in private_clients else "auxiliary" if name == "train" else name
            rows.append({
                "source_split": name,
                "source_row": int(item["source_row"]),
                "route_split": route,
                "client_id": client,
                "path": str(item["path"]),
                "accent": str(item["accent_norm"]),
                "label": int(item["label"]),
            })
    manifest = pd.DataFrame(rows)
    manifest.to_csv(args.out / "manifest.csv", index=False)
    metadata = {
        "source": "https://zenodo.org/records/12588635",
        "source_license": "CC BY 4.0 (Zenodo record metadata)",
        "root": str(args.root),
        "seed": int(args.seed),
        "aux_ratio": float(args.aux_ratio),
        "label_map": ACCENT_TO_LABEL,
        "source_rows": {name: int(len(frame)) for name, frame in frames.items()},
        "kept_rows": {name: int(len(frame)) for name, frame in kept.items()},
        "dropped_rows": {name: int(len(frames[name]) - len(kept[name])) for name in frames},
        "route_rows": {name: int((manifest["route_split"] == name).sum()) for name in ("private", "auxiliary", "dev", "test")},
        "route_label_counts": {
            name: {str(k): int(v) for k, v in manifest.loc[manifest["route_split"] == name, "label"].value_counts().sort_index().items()}
            for name in ("private", "auxiliary", "dev", "test")
        },
        "private_clients": sorted(private_clients),
        "auxiliary_clients": sorted(aux_clients),
        "speaker_disjoint_private_auxiliary": private_clients.isdisjoint(aux_clients),
        "path_has_extension": bool(manifest["path"].str.contains(r"\\.[A-Za-z0-9]+$", regex=True).all()),
    }
    (args.out / "manifest_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
