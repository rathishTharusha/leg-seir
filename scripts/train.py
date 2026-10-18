"""Run the training notebook in a new Python session; full training is the default."""
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
  # Save the executed cells even if a later cell fails.
  a.output.parent.mkdir(parents=True,exist_ok=True);nbformat.write(nb,a.output)
 print(a.output)
