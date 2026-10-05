"""Meaningful checks: sparse gates, mixture normalization, export and RNG resume."""
import copy,sys,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from gated_mdn.model import GatedMDN,ComponentGates
from utils.mdn_distribution import decode_mdn_output,build_mdn_distribution
from utils.experiment import capture_rng_state,restore_rng_state
CFG={'lstm_input_shape':4,'lstm_hidden_size':8,'lstm_num_layers':1,'num_gaussians':5,
     'output_factor':6,'forecast_horizon':48,'mdn_parameterization':{'mode':'legacy'}}
class GatesTest(unittest.TestCase):
 def setUp(self):torch.manual_seed(2024)
 def test_hard_zero_and_survivor(self):
  g=ComponentGates(5)
  with torch.no_grad():g.log_alpha.fill_(-30)
  self.assertTrue(torch.equal(g(stochastic=False),torch.tensor([1.,0.,0.,0.,0.])))
  self.assertGreaterEqual(float(g.expected_active().detach()),1)
  with torch.no_grad():g.log_alpha.zero_()
  draws=torch.stack([g(stochastic=True) for _ in range(1000)])
  measured=(draws>0).float().mean(0)
  torch.testing.assert_close(measured,g.open_probabilities(),atol=.05,rtol=0)
 def test_closed_components_zero_pi_and_unchanged_covariance(self):
  m=GatedMDN(CFG).eval()
  with torch.no_grad():m.gates.log_alpha.fill_(-30)
  x=torch.randn(2,32,4);raw=m(x);p=decode_mdn_output(raw,5);ungated=decode_mdn_output(m.ungated_output(x),5)
  for key in ['mu','sigma','rho','covariance']:torch.testing.assert_close(p[key],ungated[key],rtol=0,atol=0)
  self.assertTrue(torch.equal(p['pi'][...,1:],torch.zeros_like(p['pi'][...,1:])))
  torch.testing.assert_close(p['pi'].sum(-1),torch.ones(2,48))
  self.assertTrue(torch.isfinite(raw).all());self.assertTrue(torch.isfinite(build_mdn_distribution(raw,5).log_prob(torch.randn(2,48,2))).all())
 def test_compact_distribution_matches(self):
  m=GatedMDN(CFG).eval()
  with torch.no_grad():m.gates.log_alpha.copy_(torch.tensor([-30.,0.,-30.,2.]))
  compact,cfg,active=m.compact();self.assertEqual(active,[0,2,4]);self.assertEqual(cfg['num_gaussians'],3)
  x=torch.randn(4,32,4);y=torch.randn(4,48,2)
  torch.testing.assert_close(build_mdn_distribution(m(x),5).log_prob(y),build_mdn_distribution(compact(x),3).log_prob(y),rtol=2e-5,atol=2e-5)
  self.assertLess(sum(p.numel() for p in compact.parameters()),sum(p.numel() for p in m.parameters()))
 def test_k8_can_export_five_survivors(self):
  cfg={**CFG,'num_gaussians':8};m=GatedMDN(cfg).eval()
  with torch.no_grad():m.gates.log_alpha.copy_(torch.tensor([0.,-30.,1.,-30.,2.,0.,-30.]))
  compact,small,active=m.compact();self.assertEqual(active,[0,1,3,5,6]);self.assertEqual(small['num_gaussians'],5)
  x=torch.randn(2,32,4);y=torch.randn(2,48,2)
  torch.testing.assert_close(build_mdn_distribution(m(x),8).log_prob(y),build_mdn_distribution(compact(x),5).log_prob(y),rtol=2e-5,atol=2e-5)
 def test_training_gradients_and_stochastic_resume(self):
  m=GatedMDN(CFG);opt=torch.optim.Adam(m.parameters(),lr=.001);x=torch.randn(3,32,4);y=torch.randn(3,48,2)
  def step(model,optimizer):
   model.train();optimizer.zero_grad();loss=-build_mdn_distribution(model(x),5).log_prob(y).mean()+.01*model.expected_active()
   loss.backward();self.assertTrue(torch.isfinite(model.gates.log_alpha.grad).all());self.assertGreater(float(model.gates.log_alpha.grad.abs().sum()),0)
   optimizer.step();return loss.detach()
  step(m,opt);weights=copy.deepcopy(m.state_dict());optim=copy.deepcopy(opt.state_dict());rng=capture_rng_state();expected=step(m,opt)
  resumed=GatedMDN(CFG);resumed.load_state_dict(weights);other=torch.optim.Adam(resumed.parameters());other.load_state_dict(optim);restore_rng_state(rng)
  actual=step(resumed,other);torch.testing.assert_close(expected,actual,rtol=0,atol=0)
  for key,value in m.state_dict().items():torch.testing.assert_close(value,resumed.state_dict()[key],rtol=0,atol=0)
if __name__=='__main__':unittest.main()
