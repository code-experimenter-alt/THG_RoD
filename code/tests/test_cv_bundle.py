import json
from types import SimpleNamespace
import numpy as np
import torch
from src.dataset import build_features_and_splits
from src.main import RunCfg
from src.train_teacher import teacher_cache_id


def test_bundle_preserves_seed_indices_and_label_order(tmp_path):
    (tmp_path / "splits.json").write_text(json.dumps(dict(class_names=["a", "b"],
        seeds={"0": dict(private=[2, 0], auxiliary=[1])})))
    for split, y in [("train", [0, 1, 0]), ("dev", [1, 0]), ("test", [0])]:
        np.save(tmp_path / f"{split}_labels.npy", y)
        np.save(tmp_path / f"{split}_ssl.npy", np.arange(len(y) * 2).reshape(-1, 2))
    cfg = RunCfg(data_root="unused", precomputed_bundle=str(tmp_path),
        teacher_backend="ssl", student_backend="ssl", model_student="linear_head")
    feat, ys, names = build_features_and_splits(cfg, torch.device("cpu"))
    assert names == ["a", "b"]
    assert feat["ssl"][0].tolist() == [[4, 5], [0, 1]]
    assert [y.tolist() for y in ys] == [[0, 0], [1], [1, 0], [0]]


def test_bundle_teacher_cache_names_distinguish_count_source():
    cfg = RunCfg(data_root="unused", precomputed_bundle="bundle")
    old = teacher_cache_id(cfg)
    cfg.teacher_counts_source = "aux"
    assert teacher_cache_id(cfg) != old
    assert len(teacher_cache_id(cfg) + "_student_ctkd_global.pt") < 255
