# -*- coding: utf-8 -*-

import os
import json
from pathlib import Path
import random
from collections import Counter
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .features import build_features_for_backend


ACCENT_CANONICAL = {
    "us": {"us", "united states english", "united states", "american", "usa"},
    "england": {"england", "england english", "british", "uk", "united kingdom", "english (england)"},
    "indian": {
        "indian",
        "india",
        "india and south asia (india, pakistan, sri lanka)",
        "india and south asia",
        "south asia",
    },
}
LABEL_MAP = {"us": 0, "england": 1, "indian": 2}
CLASS_NAMES = ["us", "england", "indian"]


def normalize_accent_3way(x):
    if not isinstance(x, str):
        return None
    s = x.strip().lower()
    for canon, vocab in ACCENT_CANONICAL.items():
        if s in vocab:
            return canon
    if s in LABEL_MAP:
        return s
    return None


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cv_csv(path: str) -> pd.DataFrame:
    sep = "\t" if path.lower().endswith(".tsv") else ","
    return pd.read_csv(path, sep=sep)


def build_audio_resolver(
    data_root: str,
    subset_folder: str,
    rel_paths: List[str],
    tag: str,
    sample_n: int = 200,
):
    rel_paths = [str(x).strip() for x in rel_paths]
    rel_paths = [x for x in rel_paths if x and x.lower() != "nan"]

    if len(rel_paths) == 0:
        raise ValueError(f"[{tag}] Empty rel_paths; check your CSV filename/path column.")

    cand_bases = [
        os.path.join(data_root, subset_folder),
        os.path.join(data_root, subset_folder, subset_folder),
        os.path.join(data_root, "clips"),
        os.path.join(data_root, "audio"),
        os.path.join(data_root, "wav"),
        os.path.join(data_root, "mp3"),
        data_root,
    ]

    seen = set()
    deduped = []
    for b in cand_bases:
        if b not in seen:
            seen.add(b)
            deduped.append(b)
    cand_bases = deduped

    n = min(int(sample_n), len(rel_paths))
    sample = rel_paths[:n]

    def candidate_paths(base: str, value: str, mode: str):
        rr = value.lstrip("./")
        p = os.path.join(base, rr) if mode == "rel" else os.path.join(base, os.path.basename(rr))
        paths = [p]
        if not os.path.splitext(p)[1]:
            paths.extend(p + ext for ext in (".mp3", ".wav", ".flac", ".ogg"))
        return paths

    def score(base: str, mode: str) -> int:
        hit = 0
        for r in sample:
            if os.path.isabs(r) and os.path.exists(r):
                hit += 1
                continue
            if any(os.path.exists(p) for p in candidate_paths(base, r, mode)):
                hit += 1
        return hit

    best_hits = -1
    best_base = None
    best_mode = None

    for b in cand_bases:
        for mode in ("rel", "base"):
            h = score(b, mode)
            if h > best_hits:
                best_hits = h
                best_base = b
                best_mode = mode

    print(f"[audio-resolve:{tag}] best_base='{best_base}' mode='{best_mode}' sample_hits={best_hits}/{n}")

    def resolver(r: str) -> str:
        r = str(r).strip()
        if os.path.isabs(r) and os.path.exists(r):
            return r
        paths = candidate_paths(best_base, r, best_mode)
        for path in paths:
            if os.path.exists(path):
                return path
        return paths[0]

    return resolver


def pick_group_col(df: pd.DataFrame, preferred: Optional[str]) -> str:
    if preferred and preferred in df.columns:
        return preferred

    for c in ["client_id", "speaker_id", "user_id", "voice_id"]:
        if c in df.columns:
            return c

    for c in ["filename", "path"]:
        if c in df.columns:
            print(f"[WARN] No speaker id column found. Falling back to '{c}' as group.")
            return c

    raise ValueError("No usable group column found.")


def group_split_indices(
    df: pd.DataFrame,
    group_col: str,
    ratios: Tuple[float, float, float, float],
    seed: int,
):
    from sklearn.model_selection import GroupShuffleSplit

    r_priv, r_aux, r_dev, r_test = ratios
    assert abs((r_priv + r_aux + r_dev + r_test) - 1.0) < 1e-6

    groups = df[group_col].astype(str).values
    idx_all = np.arange(len(df))

    gss1 = GroupShuffleSplit(n_splits=1, test_size=r_test, random_state=seed)
    idx_rem, idx_test = next(gss1.split(idx_all, groups=groups))

    df_rem = df.iloc[idx_rem].reset_index(drop=True)
    groups_rem = df_rem[group_col].astype(str).values
    dev_size = r_dev / max(1e-12, (1.0 - r_test))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=dev_size, random_state=seed + 1)
    idx_privaux, idx_dev = next(gss2.split(np.arange(len(df_rem)), groups=groups_rem))

    df_privaux = df_rem.iloc[idx_privaux].reset_index(drop=True)
    groups_pa = df_privaux[group_col].astype(str).values
    aux_size = r_aux / max(1e-12, (r_priv + r_aux))
    gss3 = GroupShuffleSplit(n_splits=1, test_size=aux_size, random_state=seed + 2)
    idx_priv, idx_aux = next(gss3.split(np.arange(len(df_privaux)), groups=groups_pa))

    idx_test_orig = df.iloc[idx_test].index.values
    idx_dev_orig = df.iloc[idx_rem].iloc[idx_dev].index.values
    idx_priv_orig = df.iloc[idx_rem].iloc[idx_privaux].iloc[idx_priv].index.values
    idx_aux_orig = df.iloc[idx_rem].iloc[idx_privaux].iloc[idx_aux].index.values

    return idx_priv_orig, idx_aux_orig, idx_dev_orig, idx_test_orig


class NumpyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, soft_logits: Optional[np.ndarray] = None):
        self.X = torch.from_numpy(np.array(X, copy=True)).float()
        self.y = torch.from_numpy(np.array(y, copy=True)).long()
        self.soft = None if soft_logits is None else torch.from_numpy(np.array(soft_logits, copy=True)).float()

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, i):
        if self.soft is None:
            return self.X[i], self.y[i]
        return self.X[i], self.y[i], self.soft[i]


def make_loaders(
    cfg,
    device: torch.device,
    X_priv, y_priv, X_aux, y_aux, X_dev, y_dev, X_test, y_test,
    soft_aux: Optional[np.ndarray] = None,
):
    def dl(X, y, shuffle, soft=None):
        ds = NumpyDataset(X, y, soft_logits=soft)
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=cfg.num_workers,
            pin_memory=(cfg.pin_memory and device.type == "cuda"),
        )

    priv_loader = dl(X_priv, y_priv, shuffle=True)
    aux_loader = dl(X_aux, y_aux, shuffle=False, soft=soft_aux)
    dev_loader = dl(X_dev, y_dev, shuffle=False)
    test_loader = dl(X_test, y_test, shuffle=False)
    return priv_loader, aux_loader, dev_loader, test_loader


def build_features_and_splits(cfg, device: torch.device):
    if getattr(cfg, "precomputed_bundle", None):
        bundle = Path(cfg.precomputed_bundle)
        meta = json.loads((bundle / "splits.json").read_text())
        indices = meta["seeds"][str(cfg.seed)]
        needed = {cfg.teacher_backend}
        if cfg.model_student is not None:
            needed.add(cfg.student_backend)
        labels = {split: np.load(bundle / f"{split}_labels.npy")
                  for split in ("train", "dev", "test")}
        feat = {}
        for backend in needed:
            arrays = {split: np.load(bundle / f"{split}_{backend}.npy", mmap_mode="r")
                      for split in ("train", "dev", "test")}
            feat[backend] = (arrays["train"][indices["private"]],
                             arrays["train"][indices["auxiliary"]],
                             arrays["dev"], arrays["test"])
        ys = (labels["train"][indices["private"]], labels["train"][indices["auxiliary"]],
              labels["dev"], labels["test"])
        return feat, ys, meta["class_names"]
    os.makedirs(cfg.cache_dir, exist_ok=True)

    df_train = load_cv_csv(os.path.join(cfg.data_root, cfg.train_csv))
    df_dev = load_cv_csv(os.path.join(cfg.data_root, cfg.dev_csv))
    df_test = load_cv_csv(os.path.join(cfg.data_root, cfg.test_csv))

    def get_fname_col(df):
        if "filename" in df.columns:
            return "filename"
        if "path" in df.columns:
            return "path"
        raise ValueError("CSV missing filename/path column.")

    fcol_train = get_fname_col(df_train)
    fcol_dev = get_fname_col(df_dev)
    fcol_test = get_fname_col(df_test)

    # Build one stable label vocabulary across all official CSVs.
    raw_labels = pd.concat([df_train[cfg.label_col], df_dev[cfg.label_col], df_test[cfg.label_col]], ignore_index=True)
    normalized_labels = raw_labels.map(lambda x: normalize_accent_3way(x) or (str(x).strip().lower() if pd.notna(x) else None))
    class_names = sorted({x for x in normalized_labels if isinstance(x, str) and x})
    label_map = {name: i for i, name in enumerate(class_names)}

    def prep(df):
        df = df.copy()
        if cfg.label_col not in df.columns:
            raise ValueError(f"label_col='{cfg.label_col}' not found.")
        df[cfg.label_col] = df[cfg.label_col].apply(lambda x: normalize_accent_3way(x) or (str(x).strip().lower() if pd.notna(x) else None))
        df = df[df[cfg.label_col].notna()].reset_index(drop=True)
        df["y"] = df[cfg.label_col].map(label_map).astype(np.int64)
        return df

    df_train = prep(df_train)
    df_dev = prep(df_dev)
    df_test = prep(df_test)

    group_col_train = pick_group_col(df_train, cfg.group_col)

    train_resolver = build_audio_resolver(cfg.data_root, cfg.train_folder, df_train[fcol_train].tolist(), tag="train")
    dev_resolver = build_audio_resolver(cfg.data_root, cfg.dev_folder, df_dev[fcol_dev].tolist(), tag="dev")
    test_resolver = build_audio_resolver(cfg.data_root, cfg.test_folder, df_test[fcol_test].tolist(), tag="test")

    df_train["audio_path"] = df_train[fcol_train].apply(train_resolver)
    df_dev["audio_path"] = df_dev[fcol_dev].apply(dev_resolver)
    df_test["audio_path"] = df_test[fcol_test].apply(test_resolver)

    def drop_missing(df, tag):
        exists = df["audio_path"].apply(os.path.exists).to_numpy()
        if exists.sum() == 0:
            raise FileNotFoundError(f"No audio found for {tag}. Check data_root and CSV file paths.")
        if exists.sum() != len(df):
            print(f"[WARN] {tag}: dropping {int((~exists).sum())} rows with missing audio files")
            df = df.loc[exists].reset_index(drop=True)
        return df

    df_train = drop_missing(df_train, "train")
    df_dev = drop_missing(df_dev, "dev")
    df_test = drop_missing(df_test, "test")

    if cfg.split_mode == "official":
        from sklearn.model_selection import GroupShuffleSplit

        groups = df_train[group_col_train].astype(str).values
        # Preserve official dev/test CSVs; split the training CSV according to
        # the configured private/auxiliary proportions.
        r_priv, r_aux, _, _ = cfg.ratios
        aux_ratio = r_aux / max(1e-12, r_priv + r_aux)
        gss = GroupShuffleSplit(n_splits=1, test_size=aux_ratio, random_state=cfg.seed)
        idx_priv, idx_aux = next(gss.split(np.arange(len(df_train)), groups=groups))
        df_priv = df_train.iloc[idx_priv].reset_index(drop=True)
        df_aux = df_train.iloc[idx_aux].reset_index(drop=True)
        df_dev2 = df_dev
        df_test2 = df_test

    elif cfg.split_mode == "custom":
        df_all = pd.concat([df_train, df_dev, df_test], axis=0, ignore_index=True)
        group_col = pick_group_col(df_all, cfg.group_col)
        idx_priv, idx_aux, idx_dev, idx_test = group_split_indices(df_all, group_col, ratios=cfg.ratios, seed=cfg.seed)
        df_priv = df_all.loc[idx_priv].reset_index(drop=True)
        df_aux = df_all.loc[idx_aux].reset_index(drop=True)
        df_dev2 = df_all.loc[idx_dev].reset_index(drop=True)
        df_test2 = df_all.loc[idx_test].reset_index(drop=True)
    else:
        raise ValueError("split_mode must be 'custom' or 'official'")

    print("[split] Dpriv:", len(df_priv), Counter(df_priv["y"].tolist()))
    print("[split] Daux :", len(df_aux), Counter(df_aux["y"].tolist()))
    print("[split] Dev  :", len(df_dev2), Counter(df_dev2["y"].tolist()))
    print("[split] Test :", len(df_test2), Counter(df_test2["y"].tolist()))

    needed = {cfg.teacher_backend}
    if cfg.model_student is not None:
        needed.add(cfg.student_backend)

    feat: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for b in needed:
        feat[b] = build_features_for_backend(cfg, device, df_priv, df_aux, df_dev2, df_test2, backend=b)

    y_priv = df_priv["y"].to_numpy(np.int64)
    y_aux = df_aux["y"].to_numpy(np.int64)
    y_dev = df_dev2["y"].to_numpy(np.int64)
    y_test = df_test2["y"].to_numpy(np.int64)

    return feat, (y_priv, y_aux, y_dev, y_test), class_names
