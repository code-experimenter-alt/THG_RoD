# -*- coding: utf-8 -*-

import os
import hashlib
from typing import List
import numpy as np
import torch
import torch.nn.functional as F


def _load_audio(path: str):
    """Load audio with a codec-independent fallback.

    Recent torchaudio releases delegate decoding to TorchCodec, which may be
    unavailable on a compute node even when FLAC support through libsndfile is
    present.  Falling back to soundfile prevents silent zero-feature samples.
    """
    try:
        import torchaudio
        return torchaudio.load(path)
    except Exception as torchaudio_error:
        try:
            import soundfile as sf
            samples, sample_rate = sf.read(path, always_2d=True, dtype="float32")
            waveform = torch.from_numpy(np.asarray(samples.T, dtype=np.float32).copy())
            return waveform, int(sample_rate)
        except Exception:
            raise torchaudio_error


def _resample_audio(waveform: torch.Tensor, source_sr: int, target_sr: int):
    if int(source_sr) == int(target_sr):
        return waveform
    try:
        import torchaudio
        return torchaudio.functional.resample(waveform, int(source_sr), int(target_sr))
    except Exception:
        new_length = max(1, int(round(waveform.shape[-1] * float(target_sr) / float(source_sr))))
        return F.interpolate(waveform.unsqueeze(0), size=new_length, mode="linear", align_corners=False).squeeze(0)


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
    first_error = None
    for i, p in enumerate(audio_paths):
        try:
            wav, sr = _load_audio(p)
            if wav.numel() == 0:
                raise RuntimeError("empty audio")
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != target_sr:
                wav = _resample_audio(wav, sr, target_sr)

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
        except Exception as exc:
            bad += 1
            if first_error is None:
                first_error = f"{type(exc).__name__}: {exc}"
            X.append(np.zeros((n_mels, max_time), dtype=np.float32))

        if (i + 1) % 2000 == 0:
            print(f"[mel] processed {i+1}/{len(audio_paths)} bad={bad}")

    X = np.stack(X, axis=0)
    if bad == len(audio_paths) and len(audio_paths) > 0:
        raise RuntimeError(f"All mel feature extractions failed; first error: {first_error}")
    suffix = f" first_error={first_error}" if first_error is not None else ""
    print(f"[mel] done X={X.shape} bad={bad}{suffix}")
    return X


@torch.no_grad()
def extract_ssl_embeddings(
    audio_paths: List[str],
    model_name: str,
    pooling: str,
    target_sr: int,
    device: torch.device,
    max_seconds: float = 12.0,
    strict: bool = False,
) -> np.ndarray:
    from transformers import AutoFeatureExtractor, AutoModel, AutoProcessor
    import torchaudio

    try:
        processor = AutoProcessor.from_pretrained(model_name)
    except Exception as processor_error:
        # Audio-only encoders such as HuBERT may not ship a text tokenizer;
        # their feature extractor is sufficient for waveform embeddings.
        print(f"[ssl] AutoProcessor fallback: {type(processor_error).__name__}")
        processor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    embs = []
    bad = 0
    max_len = int(target_sr * max_seconds)

    for i, p in enumerate(audio_paths):
        try:
            wav, sr = _load_audio(p)
            if wav.numel() == 0:
                raise RuntimeError("empty audio")
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0)
            else:
                wav = wav.squeeze(0)

            if sr != target_sr:
                wav = _resample_audio(wav, sr, target_sr)

            if wav.numel() > max_len:
                wav = wav[:max_len]

            inputs = processor(wav.numpy(), sampling_rate=target_sr, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            out = model(**inputs)
            h = out.last_hidden_state
            mask = inputs.get("attention_mask")

            if pooling == "mean":
                if mask is not None and mask.shape[1] == h.shape[1]:
                    v = (h * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
                else:
                    v = h.mean(dim=1)
            elif pooling == "cls":
                v = h[:, 0, :]
            else:
                raise ValueError(f"Unknown pooling: {pooling}")

            embs.append(v.squeeze(0).detach().cpu().numpy().astype(np.float32))
        except Exception as error:
            if strict:
                raise RuntimeError(f"SSL extraction failed for {p}") from error
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
    split_sig = hashlib.sha1("|".join(
        str(x) for df in (df_priv, df_aux, df_dev2, df_test2)
        for x in df["audio_path"].tolist()
    ).encode("utf-8")).hexdigest()[:12]

    if backend == "mel":
        key = cache_key_mel(cfg.n_mels, cfg.max_time, cfg.use_dsaf, cfg.eta0, cfg.target_sr)

        def path(tag):
            return os.path.join(cfg.cache_dir, f"{cfg.exp_id}_seed{cfg.seed}_{split_sig}_{key}_X_{tag}.npy")

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
            return os.path.join(cfg.cache_dir, f"{cfg.exp_id}_seed{cfg.seed}_{split_sig}_{key}_E_{tag}.npy")

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
