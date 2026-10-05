"""Global hard-concrete component gates on the unchanged legacy LSTM-MDN head."""
import copy
import math
import sys
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
BASE=str(ROOT/'base_mdn')
if BASE not in sys.path:sys.path.insert(0,BASE)
from base_lstm import LSTM_Trajectory_Forecast

class ComponentGates(nn.Module):
    """Protect component 0; learn K-1 gates shared by all samples and horizons."""
    def __init__(self,k,temperature=2/3,gamma=-.1,zeta=1.1,initial_open_probability=.95):
        super().__init__()
        if k<1 or not (temperature>0 and gamma<0 and zeta>1 and 0<initial_open_probability<1):
            raise ValueError('Invalid hard-concrete settings')
        self.k=int(k);self.temperature=float(temperature);self.gamma=float(gamma);self.zeta=float(zeta)
        # P(z>0)=sigmoid(log_alpha - beta*log(-gamma/zeta)).
        log_alpha=math.log(initial_open_probability/(1-initial_open_probability))+temperature*math.log(-gamma/zeta)
        self.log_alpha=nn.Parameter(torch.full((k-1,),log_alpha))

    def forward(self,stochastic=None):
        stochastic=self.training if stochastic is None else stochastic
        if stochastic:
            u=torch.rand_like(self.log_alpha).clamp(1e-6,1-1e-6)
            s=torch.sigmoid((torch.log(u)-torch.log1p(-u)+self.log_alpha)/self.temperature)
        else:
            s=torch.sigmoid(self.log_alpha)
        learned=(s*(self.zeta-self.gamma)+self.gamma).clamp(0,1)
        return torch.cat([self.log_alpha.new_ones(1),learned])

    def open_probabilities(self):
        prob=torch.sigmoid(self.log_alpha-self.temperature*math.log(-self.gamma/self.zeta))
        return torch.cat([self.log_alpha.new_ones(1),prob])

    def expected_active(self):return self.open_probabilities().sum()

class GatedMDN(LSTM_Trajectory_Forecast):
    def __init__(self,cfg):
        if cfg.get('mdn_parameterization',{'mode':'legacy'}).get('mode')!='legacy':
            raise ValueError('This variant preserves the base legacy covariance')
        super().__init__(cfg)
        self.cfg=copy.deepcopy(cfg);self.k=int(cfg['num_gaussians'])
        self.gates=ComponentGates(self.k,**cfg.get('component_gates',{}))

    def ungated_output(self,x):return super().forward(x)

    def forward(self,x):
        raw=self.ungated_output(x);g=self.gates()
        # Finite sentinel produces exactly zero softmax probability in float32.
        # clamp avoids log(0) and NaN gradients; component 0 is always available.
        logg=torch.where(g>0,g.clamp_min(torch.finfo(g.dtype).tiny).log(),g.new_full(g.shape,-1e9))
        return torch.cat([raw[...,:5*self.k],raw[...,5*self.k:]+logg],dim=-1)

    def expected_active(self):return self.gates.expected_active()

    @torch.no_grad()
    def gate_report(self):
        deterministic=self.gates(stochastic=False)
        return {'k_max':self.k,'protected_component':0,'deterministic_gates':deterministic.cpu().tolist(),
                'open_probabilities':self.gates.open_probabilities().cpu().tolist(),
                'expected_active_components':float(self.expected_active()),
                'deterministic_active_components':int((deterministic>0).sum()),
                'active_indices':torch.nonzero(deterministic>0).flatten().cpu().tolist()}

    @torch.no_grad()
    def compact(self):
        """Exact inference export: remove inactive rows and absorb surviving log-gates."""
        g=self.gates(stochastic=False);active=torch.nonzero(g>0).flatten();k=len(active)
        cfg=copy.deepcopy(self.cfg);cfg.pop('component_gates',None);cfg['num_gaussians']=k
        compact=LSTM_Trajectory_Forecast(cfg).to(device=self.fc.weight.device,dtype=self.fc.weight.dtype)
        compact.lstm.load_state_dict(self.lstm.state_dict())
        w=self.fc.weight.reshape(self.forecast_horizon,6,self.k,self.lstm_hidden_size)[:,:,active]
        b=self.fc.bias.reshape(self.forecast_horizon,6,self.k)[:,:,active].clone()
        b[:,5,:]+=g[active].log()
        compact.fc.weight.copy_(w.reshape_as(compact.fc.weight));compact.fc.bias.copy_(b.reshape_as(compact.fc.bias))
        return compact.eval(),cfg,active.cpu().tolist()
