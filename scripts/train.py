"""Execute the standalone notebook in a fresh kernel (full training by default)."""
import argparse,os
from pathlib import Path
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--output',type=Path,default=ROOT/'runs/executed_training.ipynb');a=p.parse_args()
 notebook=ROOT/'notebooks/LEG_SEIR_train_from_audited_data.ipynb'
 os.environ['STUDY_NOTEBOOK']=str(notebook);os.environ['LEG_SEIR_SMOKE']='1' if a.smoke else '0'
 nb=nbformat.read(notebook,as_version=4)
 try: NotebookClient(nb,timeout=None,resources={'metadata':{'path':str(ROOT)}}).execute()
 finally:
  a.output.parent.mkdir(parents=True,exist_ok=True);nbformat.write(nb,a.output)
 print(a.output)
