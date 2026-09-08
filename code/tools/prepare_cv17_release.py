"""Recover a declared CV17 accent subset; never infer absent audio or labels."""
import argparse
import csv
import json
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from urllib3.exceptions import HTTPError as Urllib3HTTPError
from sklearn.model_selection import StratifiedShuffleSplit

REVISION = "8262c16bf297c87a9cd88c51997c4758ed7a8ba2"
REPO = "fsicoli/common_voice_17_0"
ALIASES = {
    "england": {"england", "england english", "british", "uk", "united kingdom", "english (england)"},
    "indian": {"indian", "india", "india and south asia (india, pakistan, sri lanka)", "india and south asia", "south asia"},
    "us": {"us", "united states english", "united states", "american", "usa"},
}
LABELS = {name: i for i, name in enumerate(ALIASES)}


def prepare(metadata, out):
    out.mkdir(parents=True, exist_ok=True)
    frames = {}
    mapping = {alias: name for name, aliases in ALIASES.items() for alias in aliases}
    for split in ("train", "dev", "test"):
        rows = []
        with (metadata / f"{split}.tsv").open() as stream:
            for r in csv.DictReader(stream, delimiter="\t", quoting=csv.QUOTE_NONE):
                accent = mapping.get(r["accents"].strip().lower())
                if accent is None:
                    continue
                assert Path(r["path"]).name == r["path"] and r["client_id"]
                rows.append(dict(path=r["path"], client_id=r["client_id"], accent=accent,
                                 label=LABELS[accent], source_split=split))
        frames[split] = pd.DataFrame(rows)
        assert frames[split].path.is_unique
    for a, x in frames.items():
        for b, y in frames.items():
            if a != b:
                assert set(x.client_id).isdisjoint(y.client_id), (a, b, "speaker overlap")
                assert set(x.path).isdisjoint(y.path), (a, b, "clip overlap")
    source_counts = {s: len(f) for s, f in frames.items()}
    # Independent, fixed sampling limits domination by prolific speakers.
    train = frames["train"].sample(frac=1, random_state=20260908)
    train = train.groupby("client_id", sort=False).head(20)
    assert len(train) >= 50000
    frames["train"] = train.sample(n=50000, random_state=20260909).sort_values("path").reset_index(drop=True)
    # Separate checkpoint selection and health calibration by speaker.
    dev = frames["dev"]
    speaker_class = dev.groupby("client_id").label.agg(lambda x: int(x.mode().iloc[0]))
    sp = StratifiedShuffleSplit(n_splits=1, test_size=.5, random_state=20260910)
    fit, cal = next(sp.split(speaker_class.index, speaker_class.values))
    fit_ids, cal_ids = set(speaker_class.index[fit]), set(speaker_class.index[cal])
    frames["selection"] = dev[dev.client_id.isin(fit_ids)].copy()
    frames["calibration"] = dev[dev.client_id.isin(cal_ids)].copy()
    assert fit_ids.isdisjoint(cal_ids)
    for split, frame in frames.items():
        frame.to_csv(out / f"{split}_manifest.csv", index=False)
    result = dict(source=f"https://huggingface.co/datasets/{REPO}", revision=REVISION,
        license="CC0-1.0 according to the mirror dataset card", source_target_counts=source_counts,
        sampling="random cap of 20 clips/client, then fixed random 50000 training clips; seeds 20260908/9",
        selection_calibration="speaker-stratified split of official development; seed 20260910",
        class_names=list(LABELS), counts={s: dict(records=len(f), speakers=f.client_id.nunique(),
            classes=f.accent.value_counts().to_dict()) for s, f in frames.items()},
        official_client_and_clip_disjoint=True, old_experiment_replication=False,
        tsv_parsing="csv.QUOTE_NONE, matching source loader",
        sampling_revision="quote_none_20260908")
    (out / "source_manifest.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


def fetch_one(split, shard, wanted, out):
    relative = f"audio/en/{split}/en_{split}_{shard}.tar"
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{relative}"
    marker = out / "download_records" / f"{split}_{shard}.json"
    if marker.exists():
        record = json.loads(marker.read_text())
        assert all((out / "audio" / p).is_file() for p in record["extracted"])
        return record
    for attempt in range(3):
        extracted = []
        try:
            start = time.monotonic()
            with requests.get(url, stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                with tarfile.open(fileobj=response.raw, mode="r|") as archive:
                    for member in archive:
                        name = Path(member.name).name
                        if not member.isfile() or name not in wanted:
                            continue
                        assert Path(name).name == name and name.endswith(".mp3")
                        target = out / "audio" / name
                        if not target.exists() or target.stat().st_size != member.size:
                            payload = archive.extractfile(member).read()
                            assert len(payload) == member.size
                            temporary = target.with_suffix(".mp3.part")
                            temporary.write_bytes(payload)
                            temporary.replace(target)
                        extracted.append(name)
            record = dict(source=url, extracted=extracted, elapsed_seconds=time.monotonic()-start)
            marker.write_text(json.dumps(record))
            print("SHARD_COMPLETE", split, shard, "retained", len(extracted), "seconds", round(record["elapsed_seconds"]), flush=True)
            return record
        except (requests.RequestException, Urllib3HTTPError, tarfile.TarError, OSError) as exc:
            print("SHARD_RETRY", split, shard, attempt, repr(exc), flush=True)
            if attempt == 2:
                raise
            time.sleep(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--local-dev-archive", type=Path)
    args = ap.parse_args()
    if not (args.out / "source_manifest.json").exists():
        prepare(args.metadata, args.out)
    if args.local_dev_archive:
        wanted = set(pd.read_csv(args.out / 'dev_manifest.csv').path)
        extracted = []
        with tarfile.open(args.local_dev_archive, 'r|', bufsize=1024*1024) as archive:
            for member in archive:
                name = Path(member.name).name
                if not member.isfile() or name not in wanted:
                    continue
                target = args.out / 'audio' / name
                if not target.exists() or target.stat().st_size != member.size:
                    payload = archive.extractfile(member).read()
                    assert len(payload) == member.size
                    temporary = target.with_suffix('.mp3.local.part')
                    temporary.write_bytes(payload)
                    temporary.replace(target)
                extracted.append(name)
        assert set(extracted) == wanted
        record = dict(source=f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/audio/en/dev/en_dev_0.tar',
                      local_archive=str(args.local_dev_archive), extracted=extracted)
        (args.out / 'download_records' / 'dev_0.json').write_text(json.dumps(record))
        print('LOCAL_DEV_COMPLETE', len(extracted), flush=True)
    if not args.fetch:
        return
    (args.out / "audio").mkdir(exist_ok=True)
    (args.out / "download_records").mkdir(exist_ok=True)
    jobs = []
    for split, shards in (("dev", 1), ("test", 1), ("train", 28)):
        wanted = set(pd.read_csv(args.out / f"{split}_manifest.csv").path)
        jobs.extend((split, i, wanted) for i in range(shards))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch_one, s, i, w, args.out) for s, i, w in jobs]
        for future in as_completed(futures):
            future.result()
    missing = []
    for split in ("train", "dev", "test"):
        for name in pd.read_csv(args.out / f"{split}_manifest.csv").path:
            if not (args.out / "audio" / name).is_file():
                missing.append(name)
    assert not missing, (len(missing), missing[:5])
    (args.out / "audio_complete.json").write_text(json.dumps(dict(records=53133, missing=0)))
    print("AUDIO_COMPLETE", 53133, flush=True)


if __name__ == "__main__":
    main()
