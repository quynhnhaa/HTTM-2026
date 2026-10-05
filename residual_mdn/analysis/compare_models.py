"""Read-only post-training comparison of persisted, identical validation samples."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'residual_mdn/reports/analysis'
JOBS = [('Baseline', 'stable_mdn', 'stable_peds_imptc', 'imptc_stable_seed2024'),
        ('Attention', 'stable_attention_mdn', 'stable_attention_peds_imptc', 'imptc_stable_attention_seed2024'),
        ('Residual', 'residual_mdn', 'residual_peds_imptc', 'imptc_residual_seed2024')]
COLORS = ['#2378b5', '#dc7b19', '#339357']

def load_np(path):
    with np.load(path) as d:
        return {k: d[k] for k in d.files}

def ellipse(ax, mean, cov, color, weight):
    eig, vec = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vec[1, 1], vec[0, 1]))
    # 68% probability ellipse for a bivariate Gaussian (not GMM HPD).
    axes = 2 * np.sqrt(eig * (-2*np.log(1-.68)))
    ax.add_patch(Ellipse(mean, axes[1], axes[0], angle=angle,
                         fill=False, color=color, alpha=max(.15, float(weight)), lw=1.3))

def mixture_logpdf(points, p, mu, cov):
    return logsumexp(np.stack([np.log(max(float(p[k]), 1e-300)) +
        multivariate_normal.logpdf(points, mean=mu[k], cov=cov[k]) for k in range(3)]), axis=0)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs=[]; inputs=[]; preds=[]; manifests=[]
    for name, arch, config, prefix in JOBS:
        root=ROOT/'results/trained_models'/arch/'imptc'/config/'runs'
        run=root/(prefix+'_cont500'); parent=root/(prefix+'_tmux')
        runs.append((run,parent))
        inputs.append(load_np(run/'fixed_samples/inputs.npz'))
        preds.append(load_np(run/'fixed_samples/predictions/best.npz'))
        manifests.append(json.loads((run/'fixed_samples/manifest.json').read_text()))
    for i in range(1,3):
        for key in ('sample_ids','X','y'):
            np.testing.assert_array_equal(inputs[0][key],inputs[i][key])
        np.testing.assert_array_equal(preds[0]['sample_ids'],preds[i]['sample_ids'])
    X,y=inputs[0]['X'],inputs[0]['y']; t=np.arange(1,49)*.1
    cv=X[:,-1:,:2]+X[:,-5:,2:4].mean(1)[:,None,:]*t[None,:,None]
    np.testing.assert_allclose(cv,preds[2]['cv_trajectory'],atol=1e-5)
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    for j,((run,parent),color) in enumerate(zip(runs,COLORS)):
        hist=pd.concat([pd.read_csv(parent/'history.csv'),pd.read_csv(run/'history.csv')]).drop_duplicates('epoch').sort_values('epoch')
        for col,style in [('train_nll','--'),('validation_nll','-')]:
            ax[0].plot(hist.epoch,hist[col],style,color=color,label=f'{JOBS[j][0]} {col.split("_")[0]}',lw=1)
            ax[1].plot(hist.epoch,hist[col],style,color=color,lw=1)
    ax[0].set(xlabel='Epoch',ylabel='Marginal NLL / position',title='Full training history'); ax[0].legend(fontsize=7)
    ax[1].set(xlabel='Epoch',ylabel='Marginal NLL / position',xlim=(2000,3000),ylim=(-1.4,-.85),title='Late training')
    fig.tight_layout(); fig.savefig(OUT/'learning.png',dpi=170);plt.close(fig)
    metrics=['minade20_m','minfde20_m','ravg_percent','rmin_percent','s68_m2_per_s','s95_m2_per_s']
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for j,(run,parent) in enumerate(runs):
        records={}
        for directory in (parent,run):
            for path in (directory/'metrics').glob('epoch_*.json'):
                d=json.loads(path.read_text());records[d['epoch']]=d['metrics']
        for ax,key in zip(axes.flat,metrics):
            valid=[(e,d[key]) for e,d in sorted(records.items()) if key in d]
            if valid: ax.plot(*zip(*valid),color=COLORS[j],label=JOBS[j][0],marker='.',lw=1)
            ax.set(title=key,xlabel='Epoch'); ax.grid(alpha=.2)
    axes.flat[0].legend(); fig.suptitle('Full validation metrics at periodic checkpoints (not best-checkpoint test)');fig.tight_layout();fig.savefig(OUT/'validation_metrics.png',dpi=170);plt.close(fig)
    diagnostic={}; means=[]
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for j,p in enumerate(preds):
        pi,mu,cov=p['pi'].astype(float),p['mu'].astype(float),p['covariance'].astype(float)
        mean=(pi[...,None]*mu).sum(2); means.append(mean)
        within=(pi*np.trace(cov,axis1=-2,axis2=-1)).sum(2)
        between=(pi*((mu-mean[:,:,None,:])**2).sum(-1)).sum(2)
        error=np.linalg.norm(mean-y,axis=-1)
        diagnostic[JOBS[j][0]]={'best_epoch':int(p['epoch']), 'mean_position_error_diagnostic_m':float(error.mean()),
            'within_trace_mean_m2':float(within.mean()),'between_trace_mean_m2':float(between.mean()),
            'per_sample_mean_position_error_m':error.mean(1).tolist(),
            'within_trace_by_horizon_m2':within.mean(0).tolist(),'between_trace_by_horizon_m2':between.mean(0).tolist()}
        for ax,values,title in zip(axes[0],[within,between,within+between],['Within-component trace','Between-component trace','Total covariance trace']):
            ax.plot(t,values.mean(0),label=JOBS[j][0],color=COLORS[j]);ax.set(title=title,xlabel='Horizon (s)',ylabel='m²')
        for ax,values,title in zip(axes[1],[error,pi.max(-1),np.sqrt(np.maximum(np.linalg.det(cov),0)).sum(2)/3],['Mixture-mean position error (diagnostic)','Maximum component weight','Unweighted mean sqrt(det covariance)']):
            ax.plot(t,values.mean(0),color=COLORS[j]);ax.set(title=title,xlabel='Horizon (s)')
    axes[0,0].legend();fig.suptitle('Eight fixed validation samples: actual outputs; no monotonicity assumed');fig.tight_layout();fig.savefig(OUT/'uncertainty_decomposition.png',dpi=170);plt.close(fig)
    for slot in range(len(X)):
        points=np.concatenate([X[slot,:,:2],y[slot],cv[slot]]+[m[slot] for m in means])
        lower,upper=points.min(0),points.max(0); margin=np.maximum((upper-lower)*.15,.15)
        fig,axes=plt.subplots(1,4,figsize=(17,4))
        for j,ax in enumerate(axes[:3]):
            ax.plot(X[slot,:,0],X[slot,:,1],color='gray',label='Observed')
            ax.plot(y[slot,:,0],y[slot,:,1],'k-',label='Ground truth')
            ax.plot(cv[slot,:,0],cv[slot,:,1],'--',color='purple',label='CV prior')
            ax.plot(means[j][slot,:,0],means[j][slot,:,1],color=COLORS[j],label='Mixture mean')
            for h in [7,23,47]:
                for k in range(3):
                    ellipse(ax,preds[j]['mu'][slot,h,k],preds[j]['covariance'][slot,h,k],COLORS[j],preds[j]['pi'][slot,h,k])
                    ax.scatter(*preds[j]['mu'][slot,h,k],s=10+70*preds[j]['pi'][slot,h,k],color=COLORS[j])
            ax.set(title=f'{JOBS[j][0]} best epoch {int(preds[j]["epoch"])}',xlabel='x (m)',ylabel='y (m)');ax.set_aspect('equal',adjustable='datalim');ax.legend(fontsize=6)
        axes[3].plot(t,np.linalg.norm(cv[slot]-y[slot],axis=-1),'--',color='purple',label='CV')
        for j in range(3): axes[3].plot(t,np.linalg.norm(means[j][slot]-y[slot],axis=-1),color=COLORS[j],label=JOBS[j][0])
        axes[3].set(title='Mean position error (diagnostic)',xlabel='Horizon (s)',ylabel='m');axes[3].legend(fontsize=7)
        for ax in axes[:3]: ax.set_xlim(lower[0]-margin[0],upper[0]+margin[0]);ax.set_ylim(lower[1]-margin[1],upper[1]+margin[1])
        sample=manifests[0]['samples'][slot]
        fig.suptitle(f'Slot {slot}: {sample["sample_id"]} | class: {sample.get("movement_class")}\nIndividual Gaussian 68% ellipses at 0.8, 2.4, 4.8 s; opacity/marker size encodes weight; not GMM confidence regions; ellipses clipped to shared trajectory view',fontsize=9)
        fig.tight_layout();fig.savefig(OUT/f'sample_{slot:02d}.png',dpi=160);plt.close(fig)
    # Same sample and ground truth over training. Component identity is not a trajectory mode.
    fig,axes=plt.subplots(3,4,figsize=(14,10))
    for j,(run,parent) in enumerate(runs):
        for ax,label in zip(axes[j],['epoch_0001','epoch_0010','parent_final','best']):
            path=(parent if label.startswith('epoch_00') else run)/'fixed_samples/predictions'/f'{label}.npz'
            p=load_np(path); mean=(p['pi'][...,None]*p['mu']).sum(2)
            ax.plot(X[0,:,0],X[0,:,1],color='gray');ax.plot(y[0,:,0],y[0,:,1],'k-',label='GT');ax.plot(mean[0,:,0],mean[0,:,1],color=COLORS[j],label='Mixture mean')
            ax.set(title=f'{JOBS[j][0]} epoch {int(p["epoch"])}',xlabel='x (m)',ylabel='y (m)');ax.axis('equal')
    fig.suptitle('Fixed validation slot 0 across checkpoints');fig.tight_layout();fig.savefig(OUT/'checkpoint_evolution.png',dpi=160);plt.close(fig)
    # Reproduce grid/MC area recipe on fixed validation only; no replacement of official results.
    grid1=np.linspace(-18,18,361);xx,yy=np.meshgrid(grid1,grid1);grid=np.stack([xx.ravel(),yy.ravel()],-1)
    rng=np.random.default_rng(2024);hs=[7,15,23,31,39,47]
    areas=np.zeros((3,len(X),2,len(hs)))
    for j,p in enumerate(preds):
        for slot in range(len(X)):
            for hi,h in enumerate(hs):
                pi,mu,cov=p['pi'][slot,h].astype(float),p['mu'][slot,h].astype(float),p['covariance'][slot,h].astype(float)
                pi=pi/pi.sum();ks=rng.choice(3,1000,p=pi);samples=np.empty((1000,2))
                for k in range(3): samples[ks==k]=rng.multivariate_normal(mu[k],cov[k],size=(ks==k).sum())
                sample_lp=np.sort(mixture_logpdf(samples,pi,mu,cov)); grid_lp=mixture_logpdf(grid,pi,mu,cov)
                conf=(1000-np.searchsorted(sample_lp,grid_lp,side='right'))/1000
                for ci,c in enumerate([.68,.95]):areas[j,slot,ci,hi]=(conf<=c).mean()*1296
    np.savez_compressed(OUT/'fixed_diagnostics.npz',areas_m2=areas,horizons_seconds=t[hs],sample_ids=inputs[0]['sample_ids'])
    for j in range(3):
        diagnostic[JOBS[j][0]]['fixed_grid_area_mean_by_horizon_m2']=areas[j].mean(0).tolist()
        diagnostic[JOBS[j][0]]['fixed_grid_area_median_by_horizon_m2']=np.median(areas[j],axis=0).tolist()
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ci,ax in enumerate(axes):
        for j in range(3):
            ax.plot(t[hs],areas[j,:,ci].mean(0),color=COLORS[j],label=JOBS[j][0]+' mean')
            ax.plot(t[hs],np.median(areas[j,:,ci],axis=0),'--',color=COLORS[j])
        ax.set(title=f'{[68,95][ci]}% GMM area: mean solid / median dashed',xlabel='Horizon (s)',ylabel='m²')
    axes[0].legend(fontsize=7);fig.suptitle('Fixed validation only; independent NumPy MC 1000, same grid recipe');fig.tight_layout();fig.savefig(OUT/'fixed_confidence_areas.png',dpi=170);plt.close(fig)
    (OUT/'diagnostics.json').write_text(json.dumps({'scope':'8 fixed validation samples; descriptive diagnostics, not official test metrics','samples':manifests[0]['samples'],'models':diagnostic},indent=2))
    print(json.dumps(diagnostic,indent=2))

if __name__=='__main__': main()
