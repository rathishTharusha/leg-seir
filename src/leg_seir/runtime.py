"""Execute unchanged scientific fragments with explicit model dimensions.

The original notebook uses a shared namespace. This factory retains those
semantics and parameter names, enabling direct reload of archived checkpoints.
No PDF parsing, network access or training happens when constructing a model.
"""
from pathlib import Path
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def scientific_namespace(population):
    pop=np.asarray(population,dtype=np.float32)
    if pop.shape!=(25,) or not np.isfinite(pop).all() or not (pop>0).all():
        raise ValueError("population must contain 25 finite positive values in district-column order")
    env=dict(np=np,torch=torch,nn=nn,F=F,math=math,N=25,WINDOW=3,HORIZON=3,WIDTH=64,POP=pop)
    for name in ['mechanistic.py','base_model.py','lagged_model.py','observer_model.py']:
        p=Path(__file__).parent/'_frozen'/name
        exec(compile(p.read_text(encoding='utf8'),str(p),'exec'),env)
    return env

def build_candidate(config,fold,population):
    """Build a registered model with the exact original parameter names."""
    return scientific_namespace(population)['Candidate'](config,fold)
