"""Fail-closed CV17 frozen features, fixed-length input, exact manifest order."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import soxr
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2Model


class Audio(Dataset):
    def __init__(self, frame, root, start=0):
        self.paths = frame.path.tolist()[start:]
        self.root = root
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, index):
        name = self.paths[index]
        wav, sr = sf.read(self.root/name, dtype='float32', always_2d=True)
        wav = wav.mean(axis=1)
        assert len(wav) > 0 and np.isfinite(wav).all(), name
        if sr != 16000:
            wav = soxr.resample(wav, sr, 16000, quality='HQ')
        wav = np.asarray(wav[:192000], dtype=np.float32)
        length = len(wav)
        assert length >= 400 and float(wav.std()) > 1e-8, name
        # Normalize only real samples, then identical 12-second padding for every batch.
        wav = (wav-wav.mean())/np.sqrt(wav.var()+1e-7)
        fixed = np.zeros(192000, dtype=np.float32)
        fixed[:length] = wav
        return fixed, length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--splits', nargs='+', default=['selection', 'calibration', 'train', 'test'])
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = json.loads((args.data/'source_manifest.json').read_text())
    assert source['sampling_revision'] == 'quote_none_20260908'
    (args.out/'source_manifest.json').write_text(json.dumps(source, indent=2))
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    model = Wav2Vec2Model.from_pretrained(args.model, local_files_only=True,
                                         attn_implementation='sdpa').eval().cuda()
    config = dict(model='facebook/wav2vec2-base', revision='0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8',
        sample_rate=16000, max_seconds=12, normalization='per real waveform, variance + 1e-7',
        padding='fixed 192000 samples, zero pad after normalization',
        pooling='final hidden layer mean over valid convolution frames',
        model_attention_mask=False, dtype='float32', tf32=False, resample='soxr HQ',
        batch_size=args.batch_size, torch=torch.__version__, device=torch.cuda.get_device_name())
    target_config = args.out/'feature_protocol.json'
    if target_config.exists():
        assert json.loads(target_config.read_text()) == config, 'Do not mix feature protocols'
    else:
        target_config.write_text(json.dumps(config, indent=2))

    @torch.inference_mode()
    def embedding(wav, lengths):
        hidden = model(wav.cuda(non_blocking=True)).last_hidden_state
        n = model._get_feat_extract_output_lengths(lengths.cuda())
        mask = torch.arange(hidden.shape[1], device='cuda')[None, :] < n[:, None]
        return (hidden*mask[:, :, None]).sum(1)/n[:, None]

    # All batch members have exactly the same padded length, independently of batching.
    probe_name = args.splits[0]
    probe_frame = pd.read_csv(args.data/f'{probe_name}_manifest.csv').iloc[:4]
    probe = next(iter(DataLoader(Audio(probe_frame, args.data/'audio'), batch_size=4)))
    batched = embedding(*probe)
    singleton = torch.cat([embedding(probe[0][i:i+1], probe[1][i:i+1]) for i in range(4)])
    error = float((batched-singleton).abs().max())
    assert error < 1e-4, error
    (args.out/f'batch_consistency_{probe_name}.json').write_text(json.dumps(dict(max_abs_error=error,
        records=4, split=probe_name, threshold=1e-4)))

    for split in args.splits:
        frame = pd.read_csv(args.data/f'{split}_manifest.csv')
        output = args.out/f'{split}_ssl.npy'
        complete = args.out/f'{split}_complete.json'
        if complete.exists():
            check = np.load(output, mmap_mode='r')
            assert check.shape == (len(frame), 768) and np.isfinite(check).all()
            continue
        progress = args.out/f'{split}_progress.json'
        start = json.loads(progress.read_text())['completed'] if progress.exists() else 0
        if start:
            values = np.lib.format.open_memmap(output, mode='r+')
            assert values.shape == (len(frame), 768) and np.isfinite(values[:start]).all()
        else:
            values = np.lib.format.open_memmap(output, mode='w+', dtype=np.float32, shape=(len(frame), 768))
        loader = DataLoader(Audio(frame, args.data/'audio', start), batch_size=args.batch_size,
            num_workers=args.workers, pin_memory=True, persistent_workers=args.workers>0)
        t0 = time.perf_counter()
        index = start
        for wav, lengths in loader:
            feats = embedding(wav, lengths).cpu().numpy()
            assert np.isfinite(feats).all() and (np.linalg.norm(feats, axis=1)>0).all()
            values[index:index+len(feats)] = feats
            index += len(feats)
            if index % (args.batch_size*32) == 0 or index == len(frame):
                values.flush()
                temporary = progress.with_suffix('.json.part')
                temporary.write_text(json.dumps(dict(completed=index)))
                temporary.replace(progress)
                print(split, index, '/', len(frame), 'elapsed', round(time.perf_counter()-t0), flush=True)
        assert index == len(frame)
        values.flush()
        np.save(args.out/f'{split}_labels.npy', frame.label.to_numpy(np.int64))
        frame.to_csv(args.out/f'{split}_manifest.csv', index=False)
        complete.write_text(json.dumps(dict(records=index, errors=0, dimension=768)))
    print('FEATURES_COMPLETE', args.splits, flush=True)


if __name__ == '__main__':
    main()
