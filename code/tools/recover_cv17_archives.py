"""Finish pending CV17 archive shards using validated range downloads."""
import argparse
import json
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
import requests
from prepare_cv17_release import REPO, REVISION

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();cache=args.out/'archive_cache';cache.mkdir(exist_ok=True)
    api=f'https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/audio/en/train'
    response=requests.get(api,timeout=30);response.raise_for_status()
    sizes={int(Path(x['path']).stem.split('_')[-1]):x['size'] for x in response.json() if x['type']=='file'}
    assert set(sizes)==set(range(28))
    wanted=set(pd.read_csv(args.out/'train_manifest.csv').path)
    def job(shard):
        # The fixed subset, not the complete English corpus, is the target.
        # Once every target clip is present, unneeded archive shards are skipped.
        if wanted.issubset({p.name for p in (args.out/'audio').iterdir() if p.suffix=='.mp3'}):return
        marker=args.out/'download_records'/f'train_{shard}.json'
        if marker.exists() and json.loads(marker.read_text()).get('sampling_revision')=='quote_none_20260908':return
        archive_path=cache/f'en_train_{shard}.tar'
        url=f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/audio/en/train/en_train_{shard}.tar'
        with (cache/f'train_{shard}.log').open('a') as log:
            subprocess.run([sys.executable,str(Path(__file__).with_name('fetch_range_archive.py')),
                '--url',url,'--out',str(archive_path),'--size',str(sizes[shard]),'--workers','16'],
                stdout=log,stderr=subprocess.STDOUT,check=True)
        extracted=[]
        # Sequential large reads avoid tens of thousands of random HDD seeks.
        with tarfile.open(archive_path,'r|',bufsize=1024*1024) as archive:
            for member in archive:
                name=Path(member.name).name
                if not member.isfile() or name not in wanted:continue
                dest=args.out/'audio'/name
                if not dest.exists() or dest.stat().st_size!=member.size:
                    body=archive.extractfile(member).read();assert len(body)==member.size
                    part=dest.with_suffix('.mp3.range.part');part.write_bytes(body);part.replace(dest)
                extracted.append(name)
        marker.write_text(json.dumps(dict(source=url,extracted=extracted,archive_size=sizes[shard],
                                           sampling_revision='quote_none_20260908',
                                           acquisition='validated parallel HTTP ranges')))
        print('RECOVERED_SHARD',shard,'clips',len(extracted),flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(job,reversed(range(28))))
    missing=[]
    for split in ['train','dev','test']:
        for name in pd.read_csv(args.out/f'{split}_manifest.csv').path:
            if not (args.out/'audio'/name).is_file():missing.append(name)
    assert not missing,(len(missing),missing[:5])
    (args.out/'audio_complete.json').write_text(json.dumps(dict(records=53133,missing=0)))
    print('AUDIO_COMPLETE',53133,flush=True)

if __name__=='__main__':main()
