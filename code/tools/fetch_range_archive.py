"""Resume a pinned, uncompressed archive via validated parallel HTTP ranges."""
import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--url',required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--size',type=int,required=True)
    ap.add_argument('--workers',type=int,default=16)
    args=ap.parse_args(); marker=args.out.with_suffix('.ranges.json')
    if marker.exists():
        state=json.loads(marker.read_text())
        assert state['url']==args.url and state['size']==args.size
    else:
        prefix=args.out.stat().st_size if args.out.exists() else 0
        assert prefix<args.size
        state=dict(url=args.url,size=args.size,prefix=prefix,completed=[])
        marker.write_text(json.dumps(state))
    fd=os.open(args.out,os.O_RDWR|os.O_CREAT,0o600);os.ftruncate(fd,args.size)
    lock=threading.Lock();chunk=16*1024**2
    def get(start):
        if start in state['completed']:return
        stop=min(start+chunk,args.size)-1
        for attempt in range(3):
            try:
                with requests.get(args.url,headers={'Range':f'bytes={start}-{stop}'},timeout=(30,120)) as r:
                    r.raise_for_status()
                    assert r.status_code==206 and r.headers['Content-Range']==f'bytes {start}-{stop}/{args.size}'
                    body=r.content;assert len(body)==stop-start+1
                    written=os.pwrite(fd,body,start);assert written==len(body)
                with lock:
                    state['completed'].append(start)
                    temp=marker.with_suffix('.json.part');temp.write_text(json.dumps(state));temp.replace(marker)
                    print('RANGE_COMPLETE',start,stop,flush=True)
                return
            except (requests.RequestException,OSError):
                if attempt==2:raise
    offsets=list(range(state['prefix'],args.size,chunk))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:list(pool.map(get,offsets))
    os.fsync(fd);os.close(fd)
    assert set(state['completed'])==set(offsets)
    state['complete']=True;marker.write_text(json.dumps(state))
    print('ARCHIVE_COMPLETE',args.out,args.size,flush=True)

if __name__=='__main__':main()
