"""Reacquire original public source files; never silently accept changed bytes."""
import argparse,hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,default=ROOT/'artifacts/snapshot/provenance/source_manifest.json');p.add_argument('--destination',type=Path,default=ROOT/'sources');a=p.parse_args()
 failures=[]
 for item in json.loads(a.manifest.read_text()):
  relative=Path(item['path'])
  if relative.is_absolute() or '..' in relative.parts: raise ValueError('Unsafe manifest path')
  dst=a.destination/relative
  try:
   if not dst.exists() or hashlib.sha256(dst.read_bytes()).hexdigest()!=item['sha256']:
    req=urllib.request.Request(item['url'],headers={'User-Agent':'LEG-SEIR reproducibility/0.1'})
    data=urllib.request.urlopen(req,timeout=120).read()
    if hashlib.sha256(data).hexdigest()!=item['sha256']: raise ValueError('Original URL has changed bytes')
    dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(data)
  except Exception as e: failures.append({'path':item['path'],'url':item['url'],'error':str(e)})
 a.destination.mkdir(parents=True,exist_ok=True)
 (a.destination/'retrieval_failures.json').write_text(json.dumps(failures,indent=2))
 print('Unavailable or changed original files:',len(failures))
 if failures: raise SystemExit(1)
