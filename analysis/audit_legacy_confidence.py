"""Persist each confidence area, replaying the stable evaluator sampling order.
No changes to models/official evaluator; sorted ranks avoid its large MC×grid mask.
"""
import argparse,hashlib,json,logging,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
JOBS=[('Baseline','base_mdn','default_peds_imptc','imptc_baseline_seed2024'),
      ('Attention','attention_mdn','attention_peds_imptc','imptc_attention_seed2024')]
sys.path.insert(0,str(ROOT/'base_mdn'))
def load_class(path,name):
    import importlib.util
    spec=importlib.util.spec_from_file_location(name+'_legacy_audit',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return getattr(module,name)
from base_lstm import LSTM_Trajectory_Forecast
from eval import MDN_Forecaster
from utils.config_loader import ConfigLoader
from utils.data_loader import DataLoader
from utils.experiment import set_global_seed

DEST=ROOT/'results/analysis/legacy_s68_exact'

def write(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n')

def ranked_confidence(sample_lp,grid_lp):
    # sample_lp [MC,1,H], grid_lp [G,H]; exact strict > comparison including ties.
    ordered=sample_lp[:,0,:].T.contiguous().sort(dim=-1).values
    rank=torch.searchsorted(ordered,grid_lp.T.contiguous(),right=True)
    return ((sample_lp.shape[0]-rank).float()/sample_lp.shape[0]).T

def verify():
    a=torch.tensor([[[1.,2.]],[[1.,3.]],[[4.,2.]]])
    b=torch.tensor([[1.,2.],[0.,4.],[4.,3.]])
    torch.testing.assert_close(ranked_confidence(a,b),(a>b).float().mean(0),rtol=0,atol=0)

@torch.no_grad()
def run(job,split,limit):
    label,arch,config,prefix=job
    out=DEST/('smoke' if limit else 'full')/arch/split
    out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists() and not limit:
        print('Skip completed',out,flush=True);return
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg=ConfigLoader(str(ROOT/arch/'configs/imptc'/f'{config}.json'),'imptc',False,False,config,arch,'testing')
    run_dir=ROOT/'results/trained_models'/arch/'imptc'/config/'runs'/prefix
    cp=run_dir/'checkpoints/best.pt';checkpoint=torch.load(cp,map_location=device,weights_only=False)
    cfg.model_params=checkpoint['resolved_config']['model_params'];cfg.model_params['mdn_parameterization']={'mode':'legacy'};cfg.experiment_params['evaluation_seed']=2024
    cls=LSTM_Trajectory_Forecast if arch=='base_mdn' else load_class(ROOT/arch/'model.py','AttentionMDN' if arch=='attention_mdn' else 'ResidualMDN')
    model=cls(cfg.model_params).to(device);model.load_state_dict(checkpoint['model_state_dict']);model.eval()
    loader=DataLoader(cfg)
    if split=='test':loader.load_test_data();data=loader.get_test_data();ids=loader.sample_ids['test']
    else:loader.load_eval_data();data=loader.get_eval_data();ids=loader.current_eval_ids
    n=len(data[0]) if not limit else min(limit,len(data[0]));hs=cfg.test_params['test_horizons'];levels=cfg.test_params['confidence_levels']
    # Constructor provides original grid/distribution methods and consumes no randomness.
    evaluator=MDN_Forecaster(cfg,model,loader,'testing' if split=='test' else 'eval',logging.getLogger('audit'),device)
    grid=evaluator.build_mesh_grid(18,18,.1);side=361
    assert grid.shape[0]==side*side
    mask=np.zeros((side,side),bool);mask[[0,-1],:]=True;mask[:,[0,-1]]=True
    border=torch.as_tensor(mask.ravel(),device=device)
    area=np.lib.format.open_memmap(out/'areas_m2.npy',mode='w+',dtype='float32',shape=(n,len(levels),len(hs)))
    touch=np.lib.format.open_memmap(out/'touches_boundary.npy',mode='w+',dtype='bool',shape=area.shape)
    np.savez_compressed(out/'identifiers.npz',index=np.arange(n),sample_id=np.asarray(ids[:n],dtype=str),source=np.asarray(data[4][:n],dtype=str))
    meta={'status':'running','label':label,'split':split,'sample_count':n,'checkpoint':str(cp),'checkpoint_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),
          'epoch':int(checkpoint['epoch']),'seed':2024,'test_params':cfg.test_params,'mdn_parameterization':cfg.model_params['mdn_parameterization'],
          'area_axes':['sample','confidence','horizon'],'levels':levels,'times_seconds':[(h+1)*.1 for h in hs],
          'rng_protocol':'Replay ADE sampling, reliability sampling, then each sample grid sampling in evaluator batch order',
          'boundary_definition':'At least one selected confidence-region grid point on the square border',
          'device':str(device),'torch_version':torch.__version__,'completed_samples':0}
    write(out/'progress.json',meta);set_global_seed(2024);start_time=time.time()
    for start in range(0,n,128):
        stop=min(start+128,n)
        X=torch.as_tensor(data[0][start:stop,-32:],dtype=torch.float32,device=device);y=torch.as_tensor(data[1][start:stop],dtype=torch.float32,device=device)
        outputs=model(X)[:,hs];targets=y[:,hs]
        # Preserve official RNG progression, even though ADE/reliability are not aggregated here.
        if cfg.eval_metrics['ade_fde_k']:evaluator.sample_with_probs(outputs,3,cfg.test_params['num_k_samples'])
        if cfg.eval_metrics['reliability']:evaluator.build_confidence_set_mdn(outputs,targets,3,1000)
        for offset in range(stop-start):
            gmm=evaluator.build_distribution(outputs[offset:offset+1],3)
            grid_lp=gmm.log_prob(grid)
            samples=gmm.sample((1000,));sample_lp=gmm.log_prob(samples)
            conf=ranked_confidence(sample_lp,grid_lp)
            if limit and start==0 and offset==0:
                # Real checkpoint verification at all horizons and 31 grid points.
                torch.testing.assert_close(conf[:31],(sample_lp>grid_lp[:31]).float().mean(0),rtol=0,atol=0)
            for ci,level in enumerate(levels):
                selected=conf<=level
                area[start+offset,ci]=(selected.float().mean(0)*1296).cpu().numpy()
                touch[start+offset,ci]=selected[border].any(0).cpu().numpy()
        area.flush();touch.flush();meta.update(completed_samples=stop,elapsed_seconds=time.time()-start_time)
        write(out/'progress.json',meta)
        if start%1280==0:print(label,split,stop,'/',n,'seconds',round(meta['elapsed_seconds'],1),flush=True)
    summary={'metadata':dict(meta,status='completed'),'confidence':{}}
    for ci,level in enumerate(levels):
        a=np.asarray(area[:,ci],dtype=np.float64);q=np.percentile(a,np.arange(101),axis=0)
        times=np.asarray(meta['times_seconds']);contrib=(q/times).sum(1)/4.8/101
        score=float(contrib.sum()); key='s95_m2_per_s' if level==.95 else 's68_m2_per_s'
        result={'local_scalar':score,'mean_area_by_horizon':a.mean(0).tolist(),'quantiles_area_by_horizon':{str(p):np.percentile(a,p,axis=0).tolist() for p in [0,25,50,75,95,99,100]},
                'scalar_contribution_by_percentile':contrib.tolist(),'q100_contribution':float(contrib[100]),'q100_fraction_of_scalar':float(contrib[100]/score),
                'boundary_touch_fraction_by_horizon':np.asarray(touch[:,ci]).mean(0).tolist(),'full_grid_fraction_by_horizon':(a>=1295.99).mean(0).tolist(),
                'top_indices_per_horizon':[np.argsort(a[:,h])[-10:][::-1].tolist() for h in range(len(hs))]}
        if split=='test' and not limit:
            official=(json.loads((ROOT/'results/comparisons/imptc_m1_vs_m3/comparison.json').read_text())['results'][1][key] if arch=='base_mdn' else json.loads((run_dir/'testing/evaluation.json').read_text())['official_metrics'][key])
            result.update(previous_official_scalar=official,delta_vs_previous=score-official)
        summary['confidence'][str(level)]=result
    write(out/'summary.json',summary);meta['status']='completed';write(out/'progress.json',meta)
    print('Completed',out,flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int);args=parser.parse_args()
    verify();torch.set_num_threads(2)
    for job in JOBS:
        for split in ['test']:run(job,split,args.limit)
if __name__=='__main__':main()
