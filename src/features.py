# -*- coding: utf-8 -*-

import os
from typing import List
import numpy as np
import torch
import torch.nn.functional as F


def extract_mel_features(
    audio_paths: List[str],
    n_mels: int,
    max_time: int,
    use_dsaf: bool,
    eta0: float,
    target_sr: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    win_length: int = 400,
    eps_log: float = 1e-6,
) -> np.ndarray:
    import torchaudio

    mel_tf = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
        center=True,
        pad_mode="reflect",
        norm=None,
        mel_scale="htk",
    )

    X = []
    bad = 0
    for i, p in enumerate(audio_paths):
        try:
            wav, sr = torchaudio.load(p)
            if wav.numel() == 0:
                raise RuntimeError("empty audio")
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != target_sr:
                wav = torchaudio.functional.resample(wav, sr, target_sr)

            mel = mel_tf(wav)
            mel = torch.log(mel + eps_log)
            mel = mel.squeeze(0)

            T = mel.shape[1]
            if T < max_time:
                mel = F.pad(mel, (0, max_time - T), mode="constant", value=0.0)
            elif T > max_time:
                mel = mel[:, :max_time]

            if use_dsaf:
                mu = mel.mean(dim=1, keepdim=True)
                std = mel.std(dim=1, keepdim=True, unbiased=False)
                mel = (mel - mu) / (std + float(eta0))

            X.append(mel.cpu().numpy().astype(np.float32))
        except Exception:
            bad += 1
            X.append(np.zeros((n_mels, max_time), dtype=np.float32))

        if (i + 1) % 2000 == 0:
            print(f"[mel] processed {i+1}/{len(audio_paths)} bad={bad}")

    X = np.stack(X, axis=0)
    print(f"[mel] done X={X.shape} bad={bad}")
    return X


@torch.no_grad()
def extract_ssl_embeddings(
    audio_paths: List[str],
    model_name: str,
    pooling: str,
    target_sr: int,
    device: torch.device,
    max_seconds: float = 12.0,
) -> np.ndarray:
    from transformers import AutoProcessor, AutoModel
    import torchaudio

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    embs = []
    bad = 0
    max_len = int(target_sr * max_seconds)

    for i, p in enumerate(audio_paths):
        try:
            wav, sr = torchaudio.load(p)
            if wav.numel() == 0:
                raise RuntimeError("empty audio")
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0)
            else:
                wav = wav.squeeze(0)

            if sr != target_sr:
                wav = torchaudio.functional.resample(wav, sr, target_sr)

            if wav.numel() > max_len:
                wav = wav[:max_len]

            inputs = processor(wav.numpy(), sampling_rate=target_sr, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            out = model(**inputs)
            h = out.last_hidden_state

            if pooling == "mean":
                v = h.mean(dim=1)
            elif pooling == "cls":
                v = h[:, 0, :]
            else:
                raise ValueError(f"Unknown pooling: {pooling}")

            embs.append(v.squeeze(0).detach().cpu().numpy().astype(np.float32))
        except Exception:
            bad += 1
            embs.append(None)

        if (i + 1) % 500 == 0:
            print(f"[ssl] processed {i+1}/{len(audio_paths)} bad={bad}")

    dim = None
    for v in embs:
        if v is not None:
            dim = int(v.shape[0])
            break
    if dim is None:
        raise RuntimeError("All SSL embeddings failed; check audio paths / model name.")

    out_arr = np.zeros((len(audio_paths), dim), dtype=np.float32)
    for i, v in enumerate(embs):
        if v is not None:
            out_arr[i] = v

    print(f"[ssl] done E={out_arr.shape} bad={bad}")
    return out_arr


def cache_key_mel(n_mels, max_time, use_dsaf, eta0, target_sr):
    return f"mel_sr{target_sr}_m{n_mels}_L{max_time}_dsaf{int(use_dsaf)}_eta{eta0:g}"


def cache_key_ssl(model_name: str, pooling: str, target_sr: int):
    safe = model_name.replace("/", "_")
    return f"ssl_{safe}_pool{pooling}_sr{target_sr}"


def build_features_for_backend(cfg, device, df_priv, df_aux, df_dev2, df_test2, backend: str):
    os.makedirs(cfg.cache_dir, exist_ok=True)

    if backend == "mel":
        key = cache_key_mel(cfg.n_mels, cfg.max_time, cfg.use_dsaf, cfg.eta0, cfg.target_sr)

        def path(tag):
            return os.path.join(cfg.cache_dir, f"{cfg.exp_id}_seed{cfg.seed}_{key}_X_{tag}.npy")

        def load_or_make(df, p):
            if os.path.exists(p):
                return np.load(p)
            X = extract_mel_features(
                df["audio_path"].tolist(),
                n_mels=cfg.n_mels,
                max_time=cfg.max_time,
                use_dsaf=cfg.use_dsaf,
                eta0=cfg.eta0,
                target_sr=cfg.target_sr,
            )
            np.save(p, X)
            return X

        return (
            load_or_make(df_priv, path("priv")),
            load_or_make(df_aux, path("aux")),
            load_or_make(df_dev2, path("dev")),
            load_or_make(df_test2, path("test")),
        )

    if backend == "ssl":
        key = cache_key_ssl(cfg.ssl_model_name, cfg.ssl_pooling, cfg.target_sr)

        def path(tag):
            return os.path.join(cfg.cache_dir, f"{cfg.exp_id}_seed{cfg.seed}_{key}_E_{tag}.npy")

        def load_or_make(df, p):
            if os.path.exists(p):
                return np.load(p)
            E = extract_ssl_embeddings(
                df["audio_path"].tolist(),
                model_name=cfg.ssl_model_name,
                pooling=cfg.ssl_pooling,
                target_sr=cfg.target_sr,
                device=device,
                max_seconds=float(cfg.ssl_max_seconds),
            )
            np.save(p, E)
            return E

        return (
            load_or_make(df_priv, path("priv")),
            load_or_make(df_aux, path("aux")),
            load_or_make(df_dev2, path("dev")),
            load_or_make(df_test2, path("test")),
        )

    raise ValueError(f"Unknown backend: {backend}")