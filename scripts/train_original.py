"""Run the original notebook that reads report PDFs, without changing its cells."""
import os
from pathlib import Path
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
 assert (ROOT/'sources/manifest.json').exists(),'Retrieve the original sources first'
 assert not (ROOT/'evaluation_outputs').exists(),'Never overwrite an original-source run; use a fresh checkout'
 notebook=ROOT/'experiments/frozen/Dengue_final_study.ipynb'
 os.environ['STUDY_NOTEBOOK']=str(notebook)
 nb=nbformat.read(notebook,as_version=4)
 try:NotebookClient(nb,timeout=None,resources={'metadata':{'path':str(ROOT)}}).execute()
 finally:
  # Keep the executed notebook so a failed run can also be inspected.
  (ROOT/'runs').mkdir(exist_ok=True)
  nbformat.write(nb,ROOT/'runs/executed_original.ipynb')
