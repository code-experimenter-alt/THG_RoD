"""Small offline checks for dynamic labels and health-aware loss routing."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import dataset
from src.losses import hakd_y_loss, kd_loss, loss_balanced_softmax, loss_logit_adjust
from src.routing import (
    rc_tcrd_masses,
    sample_rc_tcrd_mode,
    certified_rc_tcrd_mode,
)
from src.main import _state_dict_sha256
from src.metrics import compute_teacher_health, estimate_health_se, eval_all


def test_dynamic_labels():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        rows = []
        labels = ["American", "English", "Irish", "Scottish"]
        for i in range(16):
            name = f"p{i:03d}.flac"
            (root / name).write_bytes(b"")
            rows.append({"filename": name, "accent": labels[i % 4], "speaker_id": f"s{i:02d}"})
        (root / "missing.flac").write_bytes(b"")
        rows.append({"filename": "missing.flac", "accent": None, "speaker_id": "s_missing"})
        frame = pd.DataFrame(rows)
        for n in ("train.csv", "dev.csv", "test.csv"):
            frame.to_csv(root / n, index=False)

        cfg = SimpleNamespace(
            data_root=str(root), cache_dir=str(root / "cache"), train_csv="train.csv",
            dev_csv="dev.csv", test_csv="test.csv", train_folder=".",
            dev_folder=".", test_folder=".", label_col="accent", group_col="speaker_id",
            split_mode="official", ratios=(0.6, 0.2, 0.1, 0.1), seed=0,
            teacher_backend="mel", student_backend="mel", model_student=None,
            n_mels=4, max_time=5, use_dsaf=False, eta0=1e-5, target_sr=16000,
        )

        def fake_features(cfg, device, df_priv, df_aux, df_dev, df_test, backend):
            return tuple(np.zeros((len(df), 4, 5), dtype=np.float32)
                         for df in (df_priv, df_aux, df_dev, df_test))

        old = dataset.build_features_for_backend
        dataset.build_features_for_backend = fake_features
        try:
            _feat, ys, names = dataset.build_features_and_splits(cfg, torch.device("cpu"))
        finally:
            dataset.build_features_for_backend = old
        assert names == ["english", "irish", "scottish", "us"], names
        assert max(int(y.max()) for y in ys) == 3


def test_audio_resolver_appends_common_voice_suffix():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "clips").mkdir()
        audio = root / "clips" / "clip_hash.mp3"
        audio.write_bytes(b"audio")
        resolver = dataset.build_audio_resolver(str(root), "", ["clip_hash"], tag="common_voice")
        assert Path(resolver("clip_hash")) == audio


def test_hakd_modes():
    slogits = torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    tlogits = torch.tensor([[2.0, 0.0, -1.0], [-1.0, 2.0, 0.0]])
    y = torch.tensor([0, 1])
    health = torch.tensor([1.0, 0.5, 0.2])
    for mode in ("instance", "class", "uniform", "confidence"):
        loss, info = hakd_y_loss(slogits, tlogits, y, 0.7, 2.0, health, mode)
        assert torch.isfinite(loss)
        assert 0.0 <= info["lambda_min"] <= info["lambda_max"] <= 1.0


def test_hakd_zero_one_and_instance_weight_regressions():
    """HAKD must reduce to Hard at w=0 and to the soft KD term at w=1."""
    slogits = torch.tensor([[1.0, -0.5, 0.2], [0.1, 0.7, -0.2]], dtype=torch.float64)
    tlogits = torch.tensor([[0.8, 0.0, -0.3], [-0.4, 1.2, 0.1]], dtype=torch.float64)
    y = torch.tensor([0, 1])
    alpha = 1.0
    temperature = 2.0

    hard = torch.nn.functional.cross_entropy(slogits, y)
    zero, zero_info = hakd_y_loss(
        slogits, tlogits, y, alpha, temperature,
        torch.zeros(3, dtype=torch.float64), "class"
    )
    assert torch.allclose(zero, hard, atol=1e-12, rtol=1e-10)
    assert zero_info["lambda_min"] == 0.0 and zero_info["lambda_max"] == 0.0

    kd_only = kd_loss(slogits, tlogits, y, alpha, temperature)
    one, one_info = hakd_y_loss(
        slogits, tlogits, y, alpha, temperature,
        torch.ones(3, dtype=torch.float64), "class"
    )
    assert torch.allclose(one, kd_only, atol=1e-12, rtol=1e-10)
    assert one_info["lambda_min"] == 1.0 and one_info["lambda_max"] == 1.0

    class_health = torch.tensor([0.2, 0.8, 0.4], dtype=torch.float64)
    weighted, info = hakd_y_loss(
        slogits, tlogits, y, 0.7, temperature, class_health, "instance"
    )
    p_t = torch.softmax(tlogits / temperature, dim=1)
    expected_lam = 0.7 * (p_t * class_health.view(1, -1)).sum(dim=1)
    hard_each = torch.nn.functional.cross_entropy(slogits, y, reduction="none")
    soft_each = torch.nn.functional.kl_div(
        torch.log_softmax(slogits / temperature, dim=1), p_t, reduction="none"
    ).sum(dim=1) * temperature * temperature
    expected = ((1.0 - expected_lam) * hard_each + expected_lam * soft_each).mean()
    assert torch.allclose(weighted, expected, atol=1e-12, rtol=1e-10)
    assert abs(info["lambda_mean"] - float(expected_lam.mean())) < 1e-12


def test_student_state_hash_is_deterministic():
    torch.manual_seed(17)
    first = torch.nn.Linear(4, 3, bias=True)
    torch.manual_seed(17)
    second = torch.nn.Linear(4, 3, bias=True)
    assert _state_dict_sha256(first) == _state_dict_sha256(second)


def test_student_shuffle_generator_is_reproducible():
    values = torch.arange(12)
    first_gen = torch.Generator(device="cpu").manual_seed(101)
    second_gen = torch.Generator(device="cpu").manual_seed(101)
    first = list(torch.utils.data.DataLoader(values, batch_size=3, shuffle=True, generator=first_gen))
    second = list(torch.utils.data.DataLoader(values, batch_size=3, shuffle=True, generator=second_gen))
    assert torch.equal(torch.cat(first), torch.cat(second))


def test_metric_health_reference_confusion_matrix():
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    predictions = torch.tensor([0, 1, 1, 1, 2, 0])
    logits = torch.full((len(labels), 3), -2.0)
    logits[torch.arange(len(labels)), predictions] = 2.0
    loader = DataLoader(TensorDataset(logits, labels), batch_size=6, shuffle=False)

    class Identity(torch.nn.Module):
        def forward(self, x):
            return x

    metrics = eval_all(Identity(), loader, torch.device("cpu"), num_classes=3)
    assert metrics["confusion"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert abs(metrics["bal_acc"] - (2.0 / 3.0)) < 1e-12
    assert abs(metrics["macro_f1"] - ((0.5 + 0.8 + (2.0 / 3.0)) / 3.0)) < 1e-12
    assert abs(metrics["maj_pred"] - 0.5) < 1e-12

    health = compute_teacher_health(metrics)
    expected_global = ((2.0 / 3.0) + ((0.5 + 0.8 + (2.0 / 3.0)) / 3.0) + 0.5) / 3.0
    assert abs(health["global_health"] - expected_global) < 1e-12
    assert np.allclose(health["class_health"], [0.5, 1.0, 0.5], atol=1e-12)

    bal_only = compute_teacher_health(metrics, weights=(1.0, 0.0, 0.0))
    assert abs(bal_only["global_health"] - metrics["bal_acc"]) < 1e-12
    anti_majority = compute_teacher_health(metrics, weights=(0.0, 0.0, 1.0))
    assert abs(anti_majority["global_health"] - (1.0 - metrics["maj_pred"])) < 1e-12

    se_first = estimate_health_se(metrics, draws=400, seed=11)
    se_second = estimate_health_se(metrics, draws=400, seed=11)
    assert se_first > 0.0
    assert abs(se_first - se_second) < 1e-15


def test_rc_tcrd():
    masses = rc_tcrd_masses(0.52, 0.08, 0.45, 0.60)
    assert abs(sum(masses.values()) - 1.0) < 1e-8
    assert sample_rc_tcrd_mode(masses, 7) in {"hard", "hakd", "kd"}
    certified = certified_rc_tcrd_mode(0.20, 0.01, 0.45, 0.60, rho=0.10)
    assert certified["mode"] == "hard"
    assert certified["certified"] is True
    uncertain = certified_rc_tcrd_mode(0.52, 0.08, 0.45, 0.60, rho=0.10)
    assert uncertain["mode"] == "hard"
    assert uncertain["certified"] is False
    assert uncertain["fallback"] is True


def test_balanced_softmax_logit_adjust_equivalence():
    """Training-time LA and BS must agree up to floating-point tolerance."""
    logits = torch.tensor(
        [[1.2, -0.4, 0.7], [-0.3, 0.8, 1.1], [0.2, 0.1, -0.5]],
        dtype=torch.float64,
        requires_grad=True,
    )
    labels = torch.tensor([0, 2, 1])
    counts = torch.tensor([7.0, 3.0, 2.0], dtype=torch.float64)
    priors = counts / counts.sum()

    bs = loss_balanced_softmax(logits, labels, counts)
    la = loss_logit_adjust(logits, labels, priors, tau=1.0)
    assert torch.allclose(bs, la, atol=1e-8, rtol=1e-6)

    grad_bs = torch.autograd.grad(bs, logits, retain_graph=True)[0]
    grad_la = torch.autograd.grad(la, logits)[0]
    assert torch.allclose(grad_bs, grad_la, atol=1e-8, rtol=1e-6)


if __name__ == "__main__":
    test_dynamic_labels()
    test_hakd_modes()
    test_hakd_zero_one_and_instance_weight_regressions()
    test_student_state_hash_is_deterministic()
    test_student_shuffle_generator_is_reproducible()
    test_metric_health_reference_confusion_matrix()
    test_rc_tcrd()
    test_balanced_softmax_logit_adjust_equivalence()
    print("release smoke tests: PASS")
