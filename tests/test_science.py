import unittest,json,hashlib,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from leg_seir import scientific_namespace,build_candidate
class ScienceTests(unittest.TestCase):
 def test_frozen_sources(self):
  nb=json.loads((ROOT/'experiments/frozen/Dengue_final_study.ipynb').read_text())
  names=['data_helpers','mechanistic','base_model','lagged_model','observer_model','registry','full_study']
  for i,name in enumerate(names,5): self.assertEqual(''.join(nb['cells'][i]['source']),(ROOT/'src/leg_seir/_frozen'/f'{name}.py').read_text())
  codes=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
  self.assertEqual(hashlib.sha256(json.dumps(codes,sort_keys=True).encode()).hexdigest(),'3ea0794104f30993d7b0994c17e76e95435def0ad11305cfa535c91828009ed1')
  train=json.loads((ROOT/'notebooks/LEG_SEIR_train_from_audited_data.ipynb').read_text())
  actual=[''.join(c['source']) for c in train['cells'] if c['cell_type']=='code'][1:]
  self.assertEqual(actual,[''.join(nb['cells'][i]['source']) for i in range(5,12)])
 def test_conservative_solver(self):
  step=scientific_namespace(np.full(25,100000))['conservative_step']
  state=torch.softmax(torch.randn(7,25,4,dtype=torch.float64),-1)
  for force in [0.,.1,100.]:
   new,onset=step(state,torch.full((7,25),force,dtype=torch.float64))
   torch.testing.assert_close(new.sum(-1),state.sum(-1),rtol=1e-12,atol=1e-12)
   self.assertTrue(bool((new>=0).all()));self.assertTrue(bool((onset>=0).all()))
 def test_graph_rollout_and_gradient(self):
  torch.set_num_threads(1)
  cfg={'mode':'lagged','graph':True,'decoder':'seir','identity':False,'differences':True,'router_style':'horizon'}
  model=build_candidate(cfg,{'mean':3.,'std':1.2},np.full(25,100000))
  a=model.adjacency();torch.testing.assert_close(a.sum(-1),torch.ones(25))
  self.assertTrue(bool((a.diag()==0).all()));self.assertTrue(bool(((a>0).sum(-1)==5).all()))
  torch.testing.assert_close(model.lag_weights().sum(-1),torch.ones(25))
  raw=torch.rand(2,25,3)*100;x=(torch.log1p(raw)-3)/1.2;p=x[:,:,-1:].expand(-1,-1,3)
  y=model(x,p,raw);self.assertEqual(tuple(y.shape),(2,25,3));self.assertTrue(bool(torch.isfinite(y).all()))
  y.square().mean().backward()
  for name in ['lag_logits','left','right','import_gate','router_logits']:
   g=dict(model.named_parameters())[name].grad
   self.assertIsNotNone(g,name);self.assertTrue(bool(torch.isfinite(g).all()),name)
  forecast,states,mass=model.exposure_forecast(raw,torch.zeros_like(raw),return_state=True)
  torch.testing.assert_close(states.sum(-1),mass[...,None].expand(-1,-1,21),rtol=1e-5,atol=1e-6)
if __name__=='__main__': unittest.main()
