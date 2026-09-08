"""Run disjoint seed groups on one available GPU; no shared seed writes."""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--bundle',type=Path,required=True)
    ap.add_argument('--protocol',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--jobs',type=int,default=4)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads(args.protocol.read_text())
    destination=args.out/'protocol.json'
    if destination.exists():assert json.loads(destination.read_text())==protocol
    else:destination.write_text(json.dumps(protocol,indent=2))
    groups=[protocol['split_seeds'][i::args.jobs] for i in range(args.jobs)]
    def run(i):
        with (args.out/f'group{i}.log').open('a') as log:
            command=[sys.executable,str(Path(__file__).with_name('run_cv17_release.py')),
                     '--bundle',str(args.bundle),'--protocol',str(args.protocol),'--out',str(args.out),
                     '--seeds',*map(str,groups[i])]
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        print('GROUP_FINISHED',i,groups[i],'exit',result.returncode,flush=True)
        assert result.returncode==0,(i,result.returncode)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:list(pool.map(run,range(args.jobs)))
    assert all((args.out/f'seed{s}'/'complete.json').exists() for s in protocol['split_seeds'])
    print('ALL_TEN_SEEDS_COMPLETE',flush=True)

if __name__=='__main__':main()
