from dataclasses import asdict
from pathlib import Path

import pytest

from src.main import RunCfg
from tools.prepare_paper_data import locate_audio
from tools.replay_recorded import relocated_config


def test_audio_locations(tmp_path):
    audio = tmp_path / "wav48_silence_trimmed" / "p300" / "p300_001_mic1.flac"
    audio.parent.mkdir(parents=True)
    audio.touch()
    assert locate_audio(tmp_path, "audio/p300/p300_001_mic1.flac", "vctk") == str(audio)
    with pytest.raises(FileNotFoundError):
        locate_audio(tmp_path, "audio/p300/p300_002_mic1.flac", "vctk")


def test_relocation_preserves_experimental_parameters(tmp_path):
    for split in ("train", "dev", "test"):
        (tmp_path / f"{split}.csv").touch()
    source = RunCfg(data_root="/old/audio", train_csv="/old/train.csv",
                    dev_csv="/old/dev.csv", test_csv="/old/test.csv", seed=2,
                    epochs_teacher=4, epochs_student=5, batch_size=64,
                    n_mels=32, max_time=80, sigma=1, max_grad_norm=8,
                    health_weights=(.6, .2, .2), student_mode="hakd")
    result = relocated_config({"cfg": asdict(source)}, tmp_path, tmp_path / "new")
    relocation = {"data_root", "train_csv", "dev_csv", "test_csv", "out_dir",
                  "cache_dir", "paired_hard_ckpt", "save_eval_predictions", "train_teacher"}
    for key, value in asdict(source).items():
        if key not in relocation:
            assert asdict(result)[key] == value
    assert result.train_teacher and result.save_eval_predictions
    assert Path(result.data_root) == tmp_path
