# -*- coding: utf-8 -*-
"""Measure model size and inference cost on real cached VCTK test features."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from src.models import SmallCNNMel


CASES = (
    (
        "PATE_or_direct_DP_melCNN",
        40,
        100,
        "/home/fu/tmmrr_cache/PATE_VCTK_seed0_245760cf39a5_mel_sr16000_m40_L100_dsaf0_eta1e-05_X_test.npy",
    ),
    (
        "TMM_release_student_melCNN",
        32,
        80,
        "/home/fu/tmmrr_cache/vctk_timing_bench/CV_MEL_BS_WEAK_RELEASE_seed0_245760cf39a5_mel_sr16000_m32_L80_dsaf1_eta1e-05_X_test.npy",
    ),
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for name, n_mels, max_time, feature_path in CASES:
        X = np.load(feature_path, mmap_mode="r")
        model = SmallCNNMel(n_mels=n_mels, max_time=max_time, num_classes=4).to(device).eval()
        batch = torch.from_numpy(np.asarray(X[:64])).float().to(device)
        with torch.no_grad():
            for _ in range(20):
                model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            for _ in range(100):
                model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "model": name,
                "input_shape": list(batch.shape),
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "device": str(device),
                "warmup_batches": 20,
                "timed_batches": 100,
                "batch_size": int(batch.shape[0]),
                "total_seconds": elapsed,
                "milliseconds_per_example": elapsed * 1000.0 / (100 * int(batch.shape[0])),
                "feature_source": feature_path,
            }
        )
    output = Path("/home/fu/tmmrr_results/model_cost_benchmark.json")
    output.write_text(json.dumps({"cases": rows}, indent=2), encoding="utf-8")
    print(json.dumps({"device": str(device), "output": str(output), "cases": len(rows)}))


if __name__ == "__main__":
    main()
