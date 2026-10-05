"""Meaningful checks for the probabilistic construction and training gradients."""
import json,unittest
from pathlib import Path
import torch
from torch.distributions import Beta,kl_divergence
from bayesian_mdn.model import BayesianMDN
from utils.mdn_distribution import build_mdn_distribution

class BayesianChecks(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(2024)
  cfg=json.loads(Path('bayesian_mdn/configs/imptc/smoke_bayesian_peds_imptc.json').read_text())
  self.model=BayesianMDN(cfg['model_params'])
 def test_prior_and_mean_sticks(self):
  m=self.model;self.assertAlmostEqual(float(m.kl().detach()),0.,places=5)
  w=m.stick_weights(False)
  expected=torch.tensor([.5,.25,.125,.0625,.03125,.015625,.0078125,.0078125])
  torch.testing.assert_close(w,expected)
  draws=torch.stack([m.stick_weights(True).detach() for _ in range(3000)])
  torch.testing.assert_close(draws.sum(-1),torch.ones(3000))
  torch.testing.assert_close(draws.mean(0),w.detach(),atol=.02,rtol=0.)
 def test_variational_gradients_and_covariance(self):
  m=self.model;x=torch.randn(5,32,4);y=torch.randn(5,48,2)
  torch.manual_seed(8);raw=m(x)
  base=super(BayesianMDN,m).forward(x)
  torch.testing.assert_close(raw[...,:40],base[...,:40])
  loss=-build_mdn_distribution(raw,8).log_prob(y).mean()+m.kl()/(189*48)
  loss.backward()
  for p in [m.q_a_raw,m.q_b_raw,m.fc.weight]:
   self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(float(p.grad.abs().sum()),0.)
  torch.testing.assert_close(m.kl(),kl_divergence(m.posterior(),Beta(m.prior_a,m.prior_b)).sum())
 def test_rng_checkpoint_and_deterministic_eval(self):
  m=self.model;x=torch.randn(3,32,4)
  state=torch.get_rng_state();out=m(x)
  other=BayesianMDN(json.loads(Path('bayesian_mdn/configs/imptc/smoke_bayesian_peds_imptc.json').read_text())['model_params'])
  other.load_state_dict(m.state_dict());torch.set_rng_state(state)
  torch.testing.assert_close(out,other(x))
  m.eval();torch.testing.assert_close(m(x),m(x))
  self.assertTrue(1<=m.weight_report()['effective_k_global_mass']<=8)
 def test_predictive_is_mean_of_densities_not_logits(self):
  m=self.model.eval();x=torch.randn(3,32,4);y=torch.randn(3,48,2)
  raw=m.ungated_output(x)
  bank=m.sample_weight_bank(11,99)
  predictive=m.output_with_weights(raw,bank)
  logps=torch.stack([build_mdn_distribution(m.output_with_weights(raw,w),8).log_prob(y) for w in bank])
  expected=torch.logsumexp(logps,0)-torch.tensor(11.).log()
  torch.testing.assert_close(build_mdn_distribution(predictive,8).log_prob(y),expected,rtol=1e-5,atol=1e-5)
  # Input-conditioned normalization makes mean pi differ from pi(mean w).
  raw[...,40:]=torch.arange(8).to(raw).view(1,1,8)/2
  pi_pp=torch.softmax(m.output_with_weights(raw,bank)[...,40:],-1)
  pi_mean=torch.softmax(m.output_with_weights(raw,bank.mean(0))[...,40:],-1)
  self.assertGreater(float((pi_pp-pi_mean).abs().max().detach()),1e-4)

 def test_predictive_context_rng_and_batch_invariance(self):
  m=self.model;x=torch.randn(7,32,4)
  rng=torch.get_rng_state().clone()
  torch.testing.assert_close(m.sample_weight_bank(16,18),m.sample_weight_bank(64,18)[:16])
  before=m.sample_weight_bank(64,18)
  with torch.no_grad():m.q_a_raw.add_(1e-4)
  self.assertLess(float((m.sample_weight_bank(64,18)-before).abs().max()),.001)
  with m.inference(draws=17,seed=18):
   self.assertFalse(m.training)
   torch.testing.assert_close(m(x),torch.cat([m(x[:3]),m(x[3:])]),rtol=1e-5,atol=1e-6)
   torch.testing.assert_close(rng,torch.get_rng_state())
  self.assertTrue(m.training);self.assertIsNone(m._posterior_bank)
  with self.assertRaises(RuntimeError):
   with m.inference():raise RuntimeError('test context cleanup')
  self.assertTrue(m.training);self.assertIsNone(m._posterior_bank)

 def test_conditional_usage_and_prior_k(self):
  from bayesian_mdn.diagnostics import assess_split,mass_counts
  m=self.model;x=torch.randn(5,32,4);y=torch.randn(5,48,2)
  pi=torch.tensor([[.5,.3,.2],[.999,.0005,.0005]])
  torch.testing.assert_close(mass_counts(pi,.99),torch.tensor([3,1]))
  report=assess_split(m,x.numpy(),y.numpy(),'cpu',2,16,2024)
  self.assertEqual(report['sample_count'],5)
  for coverage,data in report['conditional_k'].items():
   self.assertEqual(sum(data['histogram_all_positions']),5*48)
   self.assertTrue(1<=data['mean_k']<=8)
  torch.testing.assert_close(torch.tensor(report['mean_pi']).sum(),torch.tensor(1.))
  torch.testing.assert_close(torch.tensor(report['mean_responsibility']).sum(),torch.tensor(1.))
  self.assertEqual(m.weight_report()['effective_k_global_mass'],7)
  self.assertTrue(m.training)

 def test_k10_extreme_tail_and_gradients(self):
  cfg=json.loads(Path('bayesian_mdn/configs/imptc/smoke_bayesian_k10_peds_imptc.json').read_text())
  m=BayesianMDN(cfg['model_params'])
  x=torch.randn(4,32,4);target=torch.randn(4,48,2)
  out=m(x)
  self.assertEqual(tuple(out.shape),(4,48,60))
  self.assertEqual(m.q_a_raw.numel(),9)
  loss=-build_mdn_distribution(out,10).log_prob(target).mean()+m.kl()/(189*48)
  loss.backward()
  self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))
  # K10 can underflow the last product in float32; logits must stay finite.
  sticks=torch.full((9,),1-1e-6,requires_grad=True)
  raw=m.ungated_output(x)
  extreme=m.output_with_weights(raw,m.weights_from_sticks(sticks))
  self.assertTrue(torch.isfinite(extreme).all())
  extreme[...,50:].sum().backward()
  self.assertTrue(torch.isfinite(sticks.grad).all())

if __name__=='__main__':unittest.main()
