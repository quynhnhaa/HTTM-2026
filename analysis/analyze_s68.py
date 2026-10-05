"""Per-sample legacy GMM HDR areas; same grid and MC definition as eval.py."""
import argparse
import csv
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'base_mdn'))
sys.path.insert(0,str(ROOT/'attention_mdn'))
from base_lstm import LSTM_Trajectory_Forecast
from model import AttentionMDN
from utils.config_loader import ConfigLoader
from utils.data_loader import DataLoader
from utils.experiment import set_global_seed
from utils.mdn_distribution import decode_mdn_output, build_mdn_distribution, checkpoint_parameterization

BASE=ROOT/'results/trained_models/base_mdn/imptc/default_peds_imptc/runs/imptc_baseline_seed2024'
ATTN=ROOT/'results/trained_models/attention_mdn/imptc/attention_peds_imptc/runs/imptc_attention_seed2024'

def load_model(run,cls,device):
 ck=torch.load(run/'checkpoints/best.pt',map_location=device,weights_only=False)
 if checkpoint_parameterization(ck)!= {'mode':'legacy'}:raise ValueError('Only legacy checkpoints supported')
 model=cls(ck['resolved_config']['model_params']).to(device)
 model.load_state_dict(ck['model_state_dict']);return model.eval(),int(ck['epoch'])

@torch.no_grad()
def areas(raw,cfg,device,seed):
 """Bounding rectangles omit only points provably below both HDR thresholds."""
 set_global_seed(seed)
 b,h,_=raw.shape;k=cfg.model_params['num_gaussians'];p=decode_mdn_output(raw,k)
 g=build_mdn_distribution(raw,k)
 logs=g.log_prob(g.sample((cfg.test_params['num_samples'],))).sort(dim=0).values
 n=len(logs);levels=[.68,.95]
 thresholds=torch.stack([logs[n-int(n*l)-1] for l in levels],-1)
 rx=cfg.test_params['mesh_range_x'];ry=cfg.test_params['mesh_range_y']
 steps=int((rx+ry)/cfg.test_params['mesh_resolution']+1)
 xs=torch.linspace(-rx,rx,steps,device=device);ys=torch.linspace(-ry,ry,steps,device=device)
 # GMM density <= max(component density). Each component's superlevel set is
 # contained in this rectangle, even with correlation. Thus the union contains HDR.
 det=p['sigma'].prod(-1)*torch.sqrt(1-p['rho'].square())
 logpeak=-np.log(2*np.pi)-torch.log(det)
 radius=torch.sqrt(torch.clamp(2*(logpeak-thresholds[...,1,None]),min=0))
 lower=(p['mu']-radius[...,None]*p['sigma']).amin(dim=(0,1,2))
 upper=(p['mu']+radius[...,None]*p['sigma']).amax(dim=(0,1,2))
 gx=xs[(xs>=lower[0])&(xs<=upper[0])];gy=ys[(ys>=lower[1])&(ys<=upper[1])]
 counts=torch.zeros(b,h,2,device=device)
 if len(gx) and len(gy):
  xx,yy=torch.meshgrid(gx,gy,indexing='xy');points=torch.stack([xx.flatten(),yy.flatten()],-1)
  for start in range(0,len(points),2048):
   density=g.log_prob(points[start:start+2048,None,None,:])
   counts+=(density[...,None]>=thresholds[None,...]).sum(0)
 return (counts*(2*rx)*(2*ry)/(steps*steps)).cpu().numpy(),thresholds.cpu().numpy(),p

def score(area,times):
 return np.mean(np.percentile(area,np.arange(101),axis=0),axis=0).dot(1/times)/times[-1]

def plot_case(index,h,label,X,y,raws,area,thresholds,cfg,dest):
 fig,axes=plt.subplots(1,2,figsize=(13,5.5));records=[]
 colors=['#2563eb','#e8790c','#8b5cf6'];time_s=(h+1)*.1
 for m,(ax,name) in enumerate(zip(axes,['Baseline','Attention'])):
  raw=torch.from_numpy(raws[m][index:index+1]);p=decode_mdn_output(raw,3)
  mu=p['mu'][0,h].numpy();cov=p['covariance'][0,h].numpy();pi=p['pi'][0,h].numpy()
  ax.plot(X[index,:,0],X[index,:,1],color='gray',label='Observed')
  ax.plot(y[index,:,0],y[index,:,1],'k-',label='Future GT')
  ax.scatter(*y[index,h],marker='*',s=140,color='red',zorder=6,label=f'GT at {time_s:.1f}s')
  mix=(p['mu'][0]*p['pi'][0,...,None]).sum(-2).numpy();ax.plot(mix[:,0],mix[:,1],color='#15803d',label='GMM expectation')
  for k in range(3):
   vals,vecs=np.linalg.eigh(cov[k]);angle=np.degrees(np.arctan2(vecs[1,1],vecs[0,1]))
   ax.add_patch(Ellipse(mu[k],2*np.sqrt(vals[1]),2*np.sqrt(vals[0]),angle=angle,fill=False,color=colors[k],linestyle='--'))
   ax.scatter(*mu[k],color=colors[k],label=f'Component {k+1}, pi={pi[k]:.2f}')
  extent=np.vstack([X[index,:,:2],y[index],mu-3*p['sigma'][0,h].numpy(),mu+3*p['sigma'][0,h].numpy()])
  records.append((extent,p))
 lo=np.vstack([r[0] for r in records]).min(0)-.5;hi=np.vstack([r[0] for r in records]).max(0)+.5
 lo=np.maximum(lo,-18);hi=np.minimum(hi,18)
 xs=np.linspace(lo[0],hi[0],250);ys=np.linspace(lo[1],hi[1],250);xx,yy=np.meshgrid(xs,ys);grid=torch.tensor(np.stack([xx,yy],-1),dtype=torch.float32)
 for m,ax in enumerate(axes):
  density=build_mdn_distribution(torch.from_numpy(raws[m][index,h][None,:]),3).log_prob(grid[:,:,None,:]).squeeze(-1).numpy()
  ax.contourf(xx,yy,density,levels=[thresholds[m,index,h,1],thresholds[m,index,h,0],max(density.max(),thresholds[m,index,h,0])+1e-5],colors=['#c7e9f1','#5bb5cd'],alpha=.35)
  ax.contour(xx,yy,density,levels=sorted(thresholds[m,index,h]),colors=['#0891b2','#075985'])
  ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),xlabel='x (m)',ylabel='y (m)',title=f'{["Baseline","Attention"][m]} | A68={area[m,index,h,0]:.2f} m², A95={area[m,index,h,1]:.2f} m²')
  ax.set_aspect('equal');ax.grid(alpha=.2);ax.legend(fontsize=7)
 fig.suptitle(f'{label}: test index {index}, horizon {time_s:.1f}s\nShading: GMM 68% / 95% HDR; dashed ellipses: component 1 std (not 68% GMM)')
 fig.tight_layout();fig.savefig(dest,dpi=160);plt.close(fig)
 return [{'model':['baseline','attention'][m],'pi':p['pi'][0,h].tolist(),'mu':p['mu'][0,h].tolist(),'sigma':p['sigma'][0,h].tolist(),'rho':p['rho'][0,h].tolist()} for m,(_,p) in enumerate(records)]

def main(args):
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');set_global_seed(args.seed)
 cfg=ConfigLoader(str(ROOT/'base_mdn/configs/imptc/default_peds_imptc.json'),'imptc',False,False,'default_peds_imptc','base_mdn','testing')
 loader=DataLoader(cfg);loader.load_test_data();X,y,*_=loader.get_test_data();n=len(X) if args.limit is None else min(args.limit,len(X));X=X[:n];y=y[:n]
 dest=ROOT/args.output;dest.mkdir(parents=True,exist_ok=True)
 models=[];epochs=[]
 for run,cls in [(BASE,LSTM_Trajectory_Forecast),(ATTN,AttentionMDN)]:
  model,epoch=load_model(run,cls,device);models.append(model);epochs.append(epoch)
 cache=dest/'per_sample.npz';horizons=cfg.test_params['test_horizons'];times=(np.array(horizons)+1)*.1
 if args.resume and cache.exists():
  saved=np.load(cache);assert saved['areas'].shape==(2,n,6,2);area=saved['areas'];thresholds=saved['thresholds'];raws=saved['raw_outputs']
 else:
  area=np.zeros((2,n,6,2));thresholds=np.zeros((2,n,48,2));raws=np.zeros((2,n,48,18),dtype=np.float32);st=time.time()
  for start in range(0,n,args.batch_size):
   stop=min(n,start+args.batch_size);inputs=torch.tensor(X[start:stop],device=device,dtype=torch.float32)
   for m,model in enumerate(models):
    with torch.no_grad():raw=model(inputs)
    raws[m,start:stop]=raw.cpu().numpy()
    a,t,_=areas(raw[:,horizons],cfg,device,args.seed+start+m*1000000)
    area[m,start:stop]=a;thresholds[m,start:stop][:,horizons]=t
   if start==0 or stop% (args.batch_size*100)==0 or stop==n:print(f'{stop}/{n} samples; {time.time()-st:.1f}s',flush=True)
  np.savez_compressed(cache,areas=area,thresholds=thresholds,raw_outputs=raws,sample_ids=np.asarray(loader.sample_ids['test'][:n]))
 gt_inside=np.zeros((2,n,6),dtype=bool)
 for start in range(0,n,256):
  stop=min(n,start+256)
  for m in range(2):
   raw=torch.from_numpy(raws[m,start:stop][:,horizons]).to(device)
   target=torch.as_tensor(y[start:stop][:,horizons],dtype=torch.float32,device=device)
   with torch.no_grad():lp=build_mdn_distribution(raw,3).log_prob(target).cpu().numpy()
   gt_inside[m,start:stop]=lp>=thresholds[m,start:stop][:,horizons,0]
 with (dest/'per_sample.csv').open('w') as f:
  writer=csv.writer(f);writer.writerow(['test_index','sample_id','horizon_s','baseline_A68_m2','attention_A68_m2','delta_A68_m2','baseline_A95_m2','attention_A95_m2','baseline_GT_in68','attention_GT_in68'])
  for i in range(n):
   for j,h in enumerate(horizons):
    writer.writerow([i,loader.sample_ids['test'][i],times[j],area[0,i,j,0],area[1,i,j,0],area[1,i,j,0]-area[0,i,j,0],area[0,i,j,1],area[1,i,j,1],*gt_inside[:,i,j]])
 fullarea=np.zeros((2,n,48,2));fullarea[:,:,horizons]=area
 delta=area[1,:,:,0]-area[0,:,:,0];groups={'largest_increase':delta,'largest_attention':area[1,:,:,0],'largest_decrease':-delta};cases=[]
 for label,values in groups.items():
  chosen=set()
  for flat in np.argsort(values.ravel())[::-1]:
   i,j=np.unravel_index(flat,values.shape)
   if i in chosen:continue
   chosen.add(i);h=horizons[j];filename=f'{label}_{len(chosen)}_sample{i}_h{h+1}.png'
   # Plot accepts area at the full horizon indices.
   parameters=plot_case(i,h,label,X,y,raws,fullarea,thresholds,cfg,dest/filename)
   cases.append({'category':label,'test_index':int(i),'sample_id':loader.sample_ids['test'][i],'horizon_s':float(times[j]),'delta_A68_m2':float(delta[i,j]),'figure':filename,'parameters':parameters})
   if len(chosen)>=args.top:break
 summary={'sample_count':n,'split':'test' if n==len(loader.test_data[0]) else 'test_subset','epochs':epochs,'seed':args.seed,'protocol':{'grid':[-18,18,.1],'MC_samples':1000,'area_definition':'fraction of official grid with MC density rank <= level, times 1296 m2','bounding_box':'exact pruning via GMM density <= max unweighted component density','note':'New MC draws; diagnostic aggregates are not replacements for previously reported official metrics.'},'scores':{name:{'S68':float(score(area[m,:,:,0],times)),'S95':float(score(area[m,:,:,1],times)),'mean_A68_by_horizon':area[m,:,:,0].mean(0).tolist()} for m,name in enumerate(['baseline','attention'])},'GT_coverage68_by_horizon':{name:gt_inside[m].mean(0).tolist() for m,name in enumerate(['baseline','attention'])},'fraction_pairs_attention_wider68':float((delta>0).mean()),'cases':cases}
 # Expose the official percentile aggregation, including its maximum term.
 fig,axes=plt.subplots(1,3,figsize=(15,4.2))
 summary['percentile_aggregation']={}
 for m,name in enumerate(['baseline','attention']):
  q=np.percentile(area[m,:,:,0],np.arange(101),axis=0)
  contributions=(q/times).sum(1)/times[-1]/101
  summary['percentile_aggregation'][name]={'maximum_percentile_contribution':float(contributions[-1]),'other_100_percentiles_contribution':float(contributions[:-1].sum()),'final_A68_percentiles':dict(zip(['p50','p90','p95','p99','max'],np.percentile(area[m,:,-1,0],[50,90,95,99,100]).tolist()))}
  axes[0].plot(times,area[m,:,:,0].mean(0),'-o',label=name)
  axes[1].plot(np.arange(100),q[:-1,-1],label=name)
  axes[2].bar(m,contributions[:-1].sum(),label='Percentiles 0–99' if m==0 else None,color='#3b82f6')
  axes[2].bar(m,contributions[-1],bottom=contributions[:-1].sum(),label='Percentile 100 (maximum)' if m==0 else None,color='#f97316')
 axes[0].set(xlabel='Horizon (s)',ylabel='Mean A68 (m²)',title='Mean area by horizon');axes[0].legend()
 axes[1].set(xlabel='Percentile (maximum shown separately)',ylabel='A68 at 4.8s (m²)',title='Tail of per-sample area');axes[1].legend()
 axes[2].set(xticks=[0,1],xticklabels=['Baseline','Attention'],ylabel='Contribution to S68 (m²/s)',title='Official score includes the maximum');axes[2].legend(fontsize=8)
 for ax in axes:ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(dest/'overview.png',dpi=160);plt.close(fig)
 with (dest/'case_parameters.csv').open('w') as f:
  writer=csv.writer(f);writer.writerow(['category','test_index','horizon_s','model','component','pi','mu_x','mu_y','sigma_x','sigma_y','rho'])
  for case in cases:
   for pars in case['parameters']:
    for k in range(3):writer.writerow([case['category'],case['test_index'],case['horizon_s'],pars['model'],k+1,pars['pi'][k],*pars['mu'][k],*pars['sigma'][k],pars['rho'][k]])
 (dest/'summary.json').write_text(json.dumps(summary,indent=2))
 print(json.dumps(summary['scores'],indent=2),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--limit',type=int);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--seed',type=int,default=2024);p.add_argument('--top',type=int,default=2);p.add_argument('--resume',action='store_true');p.add_argument('--output',default='results/analysis/legacy_s68');main(p.parse_args())
