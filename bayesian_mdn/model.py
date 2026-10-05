"""Partial Bayesian conditional MDN: global Beta sticks, deterministic LSTM."""
import sys,copy,math
from contextlib import contextmanager
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Beta,kl_divergence
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'base_mdn'))
from base_lstm import LSTM_Trajectory_Forecast

class BayesianMDN(LSTM_Trajectory_Forecast):
    def __init__(self,cfg):
        if cfg.get('mdn_parameterization')!={'mode':'legacy'}:raise ValueError('Legacy covariance required')
        super().__init__(cfg)
        self.k=int(cfg['num_gaussians']);self.settings=copy.deepcopy(cfg['bayesian_weights'])
        self._posterior_bank=None
        if self.k<2:raise ValueError('K_max >= 2 required')
        alpha=float(self.settings['prior_concentration'])
        if alpha<=0:raise ValueError('Positive concentration required')
        self.register_buffer('prior_a',torch.ones(self.k-1))
        self.register_buffer('prior_b',torch.full((self.k-1,),alpha))
        inv=lambda x:math.log(math.expm1(x-1e-4))
        self.q_a_raw=nn.Parameter(torch.full((self.k-1,),inv(1.)))
        self.q_b_raw=nn.Parameter(torch.full((self.k-1,),inv(alpha)))
        if self.settings['local_logit_bound']<0 or not 0<self.settings['mass_coverage']<1:raise ValueError('Invalid settings')

    def posterior(self):return Beta(F.softplus(self.q_a_raw)+1e-4,F.softplus(self.q_b_raw)+1e-4)
    @staticmethod
    def weights_from_sticks(v):
        v=v.clamp(1e-6,1-1e-6)
        remaining=torch.cumprod(1-v,dim=-1)
        prefix=torch.cat([torch.ones_like(v[...,:1]),remaining[...,:-1]],dim=-1)
        return torch.cat([v*prefix,remaining[...,-1:]],dim=-1)

    def stick_weights(self,stochastic=None):
        q=self.posterior()
        v=q.rsample() if (self.training if stochastic is None else stochastic) else q.mean
        return self.weights_from_sticks(v)

    @torch.no_grad()
    def sample_weight_bank(self,draws=64,seed=2024):
        from scipy.special import betaincinv
        if not isinstance(draws,int) or draws<1:raise ValueError('Positive integer posterior draws required')
        # Common scrambled Sobol points give continuous checkpoint comparisons.
        # Inverse Beta CDF is inference-only (CPU float64); train still uses rsample.
        # Scrambling randomizes the integration rule; points are not iid draws.
        with torch.random.fork_rng(devices=[]):
            engine=torch.quasirandom.SobolEngine(self.k-1,scramble=True,seed=seed)
            u=engine.draw(draws,dtype=torch.float64).numpy()
        q=self.posterior()
        a=q.concentration1.detach().cpu().double().numpy()
        b=q.concentration0.detach().cpu().double().numpy()
        v=torch.from_numpy(betaincinv(a,b,u)).to(device=self.q_a_raw.device,dtype=self.q_a_raw.dtype)
        if not torch.isfinite(v).all():raise FloatingPointError('Invalid Beta inverse CDF integration points')
        return self.weights_from_sticks(v)

    @contextmanager
    def inference(self,mode='posterior_predictive',draws=64,seed=2024):
        if mode not in ('posterior_predictive','plugin'):raise ValueError('Unknown inference mode')
        previous=self._posterior_bank;was_training=self.training
        try:
            self.eval()
            self._posterior_bank=self.sample_weight_bank(draws,seed) if mode=='posterior_predictive' else None
            yield self
        finally:
            self._posterior_bank=previous
            self.train(was_training)

    def kl(self):return kl_divergence(self.posterior(),Beta(self.prior_a,self.prior_b)).sum()

    def ungated_output(self,x):return super().forward(x)

    def output_with_weights(self,raw,weights):
        local=float(self.settings['local_logit_bound'])*torch.tanh(raw[...,5*self.k:])
        if weights.ndim==1:
            logits=local+weights.clamp_min(torch.finfo(raw.dtype).tiny).log()
        elif weights.ndim==2:
            # Gaussian parameters do not depend on v. Integrating the marginal
            # mixture therefore only requires averaging normalized conditional pi.
            sampled_logits=local.unsqueeze(0)+weights.clamp_min(torch.finfo(raw.dtype).tiny).log()[:,None,None,:]
            logpi=torch.log_softmax(sampled_logits,dim=-1)
            logits=torch.logsumexp(logpi,dim=0)-math.log(len(weights))
        else:raise ValueError('Expected K weights or S x K weight bank')
        return torch.cat([raw[...,:5*self.k],logits],dim=-1)

    def forward(self,x):
        raw=self.ungated_output(x)
        weights=self.stick_weights() if self.training or self._posterior_bank is None else self._posterior_bank
        return self.output_with_weights(raw,weights)
    @torch.no_grad()
    def weight_report(self):
        w=self.stick_weights(False);coverage=float(self.settings['mass_coverage'])
        ordered,indices=w.sort(descending=True)
        count=int(torch.searchsorted(ordered.cumsum(0),w.new_tensor(coverage)).item())+1
        count=min(count,self.k)
        return {'k_max':self.k,'mean_global_weights':w.cpu().tolist(),
            'effective_k_global_mass':count,'mass_coverage':coverage,
            'selected_indices':indices[:count].cpu().tolist(),'raw_kl':float(self.kl()),
            'posterior_a':self.posterior().concentration1.cpu().tolist(),
            'posterior_b':self.posterior().concentration0.cpu().tolist(),
            'inference':'posterior_predictive_qmc' if self._posterior_bank is not None else 'plug_in_mean_sticks',
            'posterior_draw_method':'scrambled_sobol_beta_inverse_cdf',
            'posterior_draws':len(self._posterior_bank) if self._posterior_bank is not None else 0,
            'warning':'Global mass K is a diagnostic, not proof that components can be removed safely.'}
