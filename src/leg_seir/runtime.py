"""Build models from unchanged parts of the original notebook.

The files share variables, just like cells in a notebook. Keeping their order
and parameter names lets us load the saved experiment weights directly.
Building a model does not download files, read PDFs or train its weights.
"""
from pathlib import Path
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def scientific_namespace(population):
    """Prepare shared variables and load the model definitions in order."""
    pop=np.asarray(population,dtype=np.float32)
    if pop.shape!=(25,) or not np.isfinite(pop).all() or not (pop>0).all():
        raise ValueError("population must contain 25 finite positive values in district-column order")
    # Population order must match the 25 district columns in the case table.
    env=dict(np=np,torch=torch,nn=nn,F=F,math=math,N=25,WINDOW=3,HORIZON=3,WIDTH=64,POP=pop)
    for name in ['mechanistic.py','base_model.py','lagged_model.py','observer_model.py']:
        p=Path(__file__).parent/'_frozen'/name
        # Use one shared dictionary so later files can extend earlier classes.
        exec(compile(p.read_text(encoding='utf8'),str(p),'exec'),env)
    return env

def build_candidate(config,fold,population):
    """Create the selected model without changing saved-weight names."""
    return scientific_namespace(population)['Candidate'](config,fold)
