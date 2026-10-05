"""Full-split output diagnostics; no optimizer or changes to official metrics."""
import sys,json,importlib.util
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'stable_mdn'))
from base_lstm import LSTM_Trajectory_Forecast
from utils.config_loader import ConfigLoader
from utils.data_loader import DataLoader
from utils.mdn_distribution import decode_mdn_output
from compare_models import JOBS,OUT,mixture_logpdf

def load_class(path,name):
    spec=importlib.util.spec_from_file_location(name+'_module',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return getattr(module,name)
def stats(a):
    return {'mean':float(a.mean()),'quantiles':dict(zip(['0','25','50','75','95','99','100'],np.percentile(a,[0,25,50,75,95,99,100]).tolist()))}
@torch.no_grad()
def main():
    torch.set_num_threads(2)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    out={}
    for label,arch,config,prefix in JOBS:
        cfg=ConfigLoader(str(ROOT/arch/'configs/imptc'/f'{config}.json'),'imptc',False,False,config,arch,'testing')
        run=ROOT/'results/trained_models'/arch/'imptc'/config/'runs'/(prefix+'_cont500')
        checkpoint=torch.load(run/'checkpoints/best.pt',map_location=device,weights_only=False)
        params=checkpoint['resolved_config']['model_params']
        cls=LSTM_Trajectory_Forecast if arch=='stable_mdn' else load_class(ROOT/arch/'model.py','AttentionMDN' if arch=='stable_attention_mdn' else 'ResidualMDN')
        model=cls(params).to(device);model.load_state_dict(checkpoint['model_state_dict']);model.eval()
        loader=DataLoader(cfg);loader.load_eval_data();loader.load_test_data();out[label]={}
        for split,data in [('validation',loader.eval_data),('test',loader.get_test_data())]:
            X=data[0];chunks=[]
            for start in range(0,len(X),1024):
                raw=model(torch.as_tensor(X[start:start+1024,-32:],dtype=torch.float32,device=device))
                p=decode_mdn_output(raw,3,params['mdn_parameterization']);pi,mu,cov=p['pi'],p['mu'],p['covariance'];mean=(pi[...,None]*mu).sum(2)
                within=(pi*cov.diagonal(dim1=-2,dim2=-1).sum(-1)).sum(2)
                between=(pi*((mu-mean[:,:,None])**2).sum(-1)).sum(2)
                # At six official horizons; per-sample mean across horizons.
                hs=[7,15,23,31,39,47]
                chunks.append(torch.stack([within[:,hs].mean(1),between[:,hs].mean(1),
                    (within+between)[:,hs].mean(1),p['sigma'][:,hs].amax((1,2,3)),
                    (pi[:,hs]*(p['sigma'][:,hs].amax(-1)>100)).sum(2).mean(1)],1).cpu().numpy())
            a=np.concatenate(chunks);keys=['within_trace_m2','between_trace_m2','total_trace_m2','max_sigma_m','weight_of_components_sigma_over100m']
            out[label][split]={'sample_count':len(X),'diagnostics':{k:stats(a[:,i]) for i,k in enumerate(keys)},
                'top_total_trace_indices':np.argsort(a[:,2])[-10:][::-1].tolist()}
            # Targeted tail audit, not an official metric rerun.
            indices=out[label][split]['top_total_trace_indices'][:3]
            raw=model(torch.as_tensor(X[indices,-32:],dtype=torch.float32,device=device))
            decoded={k:v.cpu().numpy() for k,v in decode_mdn_output(raw,3,params['mdn_parameterization']).items()}
            axis=np.linspace(-18,18,361); gx,gy=np.meshgrid(axis,axis);grid=np.stack([gx.ravel(),gy.ravel()],-1)
            rng=np.random.default_rng(2024); audits=[]
            for slot,index in enumerate(indices):
                areas=[]
                for h in hs:
                    pi=decoded['pi'][slot,h].astype(float);pi=pi/pi.sum();mu=decoded['mu'][slot,h];cov=decoded['covariance'][slot,h]
                    ks=rng.choice(3,1000,p=pi);points=np.empty((1000,2))
                    for k in range(3): points[ks==k]=rng.multivariate_normal(mu[k],cov[k],size=(ks==k).sum())
                    lp=np.sort(mixture_logpdf(points,pi,mu,cov));glp=mixture_logpdf(grid,pi,mu,cov)
                    conf=(1000-np.searchsorted(lp,glp,side='right'))/1000
                    areas.append([(conf<=c).mean()*1296 for c in [.68,.95]])
                audits.append({'index':index,'source':str(data[4][index]),'areas_m2_horizon_by_confidence':areas})
            out[label][split]['top3_grid_audit']=audits
            print(label,split,out[label][split],flush=True)
    (OUT/'full_split_distribution.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()
