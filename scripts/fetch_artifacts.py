"""Retrieve the pinned public snapshot and enforce every SHA256."""
import argparse,hashlib,json,shutil,os,urllib.request
from pathlib import Path
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
from huggingface_hub import hf_hub_download
ROOT=Path(__file__).resolve().parents[1]
def fetch(data_only=False):
 lock=json.loads((ROOT/'artifacts.lock.json').read_text())
 for item in sorted(lock['files'],key=lambda x:x['path'].endswith('evaluation_outputs.zip')):
  name=item['path']
  if data_only and not (name.startswith('data/') or name.startswith('provenance/')): continue
  if name=='evidence/evaluation_outputs.zip':
   src=ROOT/'artifacts/.evaluation_download';src.parent.mkdir(parents=True,exist_ok=True)
   urls=[f"https://huggingface.co/datasets/{lock['repo_id']}/resolve/{lock['revision']}/{name}",
         'https://github.com/rathishTharusha/leg-seir/releases/download/v0.1.0/evaluation_outputs.zip']
   for url in urls:
    try:
     with urllib.request.urlopen(url,timeout=30) as response,src.open('wb') as output: shutil.copyfileobj(response,output)
     if hashlib.sha256(src.read_bytes()).hexdigest()!=item['sha256']: raise RuntimeError('Archive hash mismatch')
     break
    except Exception as error: print('Archive endpoint unavailable:',type(error).__name__)
   else: raise RuntimeError('Cannot retrieve archive from HF or GitHub; download the release ZIP manually')
  else:
   src=Path(hf_hub_download(lock['repo_id'],name,repo_type='dataset',revision=lock['revision']))
  if hashlib.sha256(src.read_bytes()).hexdigest()!=item['sha256']: raise RuntimeError('Hash mismatch: '+name)
  dst=ROOT/'artifacts/snapshot'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
  if name=='evidence/evaluation_outputs.zip': shutil.copy2(src,ROOT/'artifacts/evaluation_outputs.zip')
 print('Verified pinned snapshot:',lock['repo_id'],lock['revision'])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data-only',action='store_true');a=p.parse_args();fetch(a.data_only)
