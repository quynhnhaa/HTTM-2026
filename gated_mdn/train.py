"""Separate gated-MDN trainer. Full training requires --full; smoke is explicit."""
import argparse,copy,json,logging,os,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'base_mdn'))
from utils.config_loader import ConfigLoader
from utils.data_loader import DataLoader
from utils.experiment import set_global_seed,capture_rng_state,restore_rng_state,restore_checkpoint,utc_now
from utils.mdn_distribution import build_mdn_distribution
from eval import MDN_Forecaster
# Explicit local imports avoid a previously imported unrelated model module.
from gated_mdn.model import GatedMDN
from gated_mdn.artifacts import GatedTracker

@torch.no_grad()
def validation_nll(model,X,y,device,batch_size,k):
    model.eval();total=0.
    for start in range(0,len(X),batch_size):
        stop=min(start+batch_size,len(X));raw=model(torch.as_tensor(X[start:stop],dtype=torch.float32,device=device))
        target=torch.as_tensor(y[start:stop],dtype=torch.float32,device=device)
        total+=float(-build_mdn_distribution(raw,k).log_prob(target).mean())*(stop-start)
    return total/len(X)

def run(config,run_id,resume=None,smoke=False):
    cfg=ConfigLoader(str(config),'imptc',False,False,Path(config).stem,'gated_mdn','training')
    if cfg.train_params['eval_data_reduction']!=1 or cfg.train_params['dynamic_input_horizon']:
        raise ValueError('Use full deterministic validation and fixed observation horizon')
    if cfg.model_params.get('mdn_parameterization',{'mode':'legacy'})!={'mode':'legacy'}:
        raise ValueError('Do not change the baseline covariance policy')
    if cfg.experiment_params['loss_name']!='marginal_nll_plus_expected_l0':raise ValueError('Wrong loss metadata')
    strength=float(cfg.experiment_params['lambda_l0'])
    if strength<0 or not np.isfinite(strength):raise ValueError('Invalid lambda_l0')
    run_dir=Path(cfg.result_path)/'runs'/run_id
    if run_dir.exists() and not resume:raise FileExistsError(run_dir)
    set_global_seed(cfg.experiment_params['seed']);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader=DataLoader(cfg);loader.load_train_data();loader.load_eval_data()
    model=GatedMDN(cfg.model_params).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg.train_params['lr_default'])
    scheduler=torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=cfg.train_params['lr_start_factor'],
        end_factor=cfg.train_params['lr_end_factor'],total_iters=cfg.train_params['train_epochs'])
    saved=None
    if resume:
        saved=torch.load(resume,map_location='cpu',weights_only=False)
        if saved.get('architecture')!='global_hard_concrete_mdn_v1' or saved['run_id']!=run_id:raise ValueError('Incompatible resume checkpoint')
        for key,current in [('model_params',cfg.model_params),('train_params',cfg.train_params),('experiment_params',cfg.experiment_params)]:
            if saved['resolved_config'][key]!=current:raise ValueError(f'Resume config differs: {key}')
    if resume:
        import csv
        with (run_dir/'history.csv').open() as f:rows=list(csv.DictReader(f))
        if not rows or int(rows[-1]['epoch'])!=saved['epoch']:raise ValueError('Resume only latest completed epoch of this run')
    tracker=GatedTracker(cfg,loader,device,run_id);history=[];start_epoch=1
    if resume:
        saved=restore_checkpoint(resume,model,optimizer,scheduler,map_location=device)
        if not loader.load_state_dict(saved['data_loader_state']):raise ValueError('Missing loader state')
        history=saved['train_history'];start_epoch=saved['epoch']+1
        tracker.best_epoch=saved['best_epoch'];tracker.best_validation_nll=saved['best_validation_nll']
        tracker.event('run_resumed',saved['epoch'],{'checkpoint':str(resume)})
    k=cfg.model_params['num_gaussians'];batch=cfg.train_params['batch_size']
    for epoch in range(start_epoch,cfg.train_params['train_epochs']+1):
        started=time.time();model.train();X,y=loader.get_train_data();total_nll=0.;total_objective=0.
        lr=optimizer.param_groups[0]['lr']
        for start in range(0,len(X),batch):
            stop=min(start+batch,len(X));optimizer.zero_grad(set_to_none=True)
            inp=torch.as_tensor(X[start:stop],dtype=torch.float32,device=device);target=torch.as_tensor(y[start:stop],dtype=torch.float32,device=device)
            try:
                raw=model(inp);nll=-build_mdn_distribution(raw,k).log_prob(target).mean()
                objective=nll+strength*model.expected_active()
                if not torch.isfinite(objective):raise FloatingPointError('Non-finite loss')
                objective.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('Non-finite gradient')
            except (ValueError,RuntimeError,FloatingPointError) as exc:
                torch.save({'epoch':epoch,'start':start,'X':inp.cpu(),'y':target.cpu(),'error':str(exc),
                            'model_state_dict':model.state_dict(),'rng_state':capture_rng_state()},tracker.run_dir/'failure.pt')
                tracker.diverged(epoch,epoch-1);raise
            optimizer.step();total_nll+=float(nll.detach())*(stop-start);total_objective+=float(objective.detach())*(stop-start)
        scheduler.step()
        val=validation_nll(model,loader.eval_data[0],loader.eval_data[1],device,cfg.test_params['batch_size'],k)
        if not np.isfinite(val):tracker.diverged(epoch,epoch-1);raise FloatingPointError('Non-finite validation')
        gate=model.gate_report();penalty=strength*gate['expected_active_components']
        row={'run_id':run_id,'epoch':epoch,'train_nll':total_nll/len(X),'validation_nll':val,
             'train_objective':total_objective/len(X),'validation_objective':val+penalty,
             'l0_penalty':penalty,'lambda_l0':strength,'learning_rate':lr,'duration_seconds':time.time()-started,
             'train_sample_count':len(X),'validation_sample_count':len(loader.eval_data[0]),
             'expected_active_components':gate['expected_active_components'],'deterministic_active_components':gate['deterministic_active_components'],
             'gpu_peak_memory_bytes':int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else 0,
             'finite':True,'timestamp_utc':utc_now()}
        # Metric RNG cannot change gate samples or next-epoch data selection.
        evaluated=tracker.should_evaluate_metrics(epoch)
        if evaluated:
            rng=capture_rng_state()
            try:
                set_global_seed(cfg.experiment_params['seed']);metric_loader=loader
                if smoke:
                    metric_loader=copy.copy(loader);fixed=[a[tracker.fixed_indices] for a in loader.eval_data]
                    metric_loader.get_eval_data=lambda:fixed
                evaluator=MDN_Forecaster(cfg,model,metric_loader,'eval',logging.getLogger('gated'),device)
                metrics=evaluator.evaluate(epoch=epoch)
                tracker.save_metrics(epoch,metrics,split='fixed_validation_smoke' if smoke else 'eval')
            finally:restore_rng_state(rng)
        row['full_metrics_evaluated']=bool(evaluated and not smoke)
        history.append(row);best=tracker.update_best(epoch,val,model,optimizer,scheduler,history)
        periodic=tracker.should_checkpoint(epoch)
        if periodic:tracker.save_checkpoint(f'epoch_{epoch:04d}',epoch,model,optimizer,scheduler,history)
        row.update(is_best=best,checkpoint_saved=bool(periodic or best))
        tracker.log_epoch(row);tracker.log_gates(epoch,model)
        tracker.save_checkpoint('last',epoch,model,optimizer,scheduler,history,capture_predictions=False)
        print(f'Epoch {epoch}: train NLL={row["train_nll"]:.5f}, val NLL={val:.5f}, expected K={gate["expected_active_components"]:.3f}, eval K={gate["deterministic_active_components"]}',flush=True)
    last=history[-1]['epoch']
    tracker.save_checkpoint('final',last,model,optimizer,scheduler,history);tracker.complete(last)
    return tracker.run_dir

def main():
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--run-id',required=True);p.add_argument('--resume');p.add_argument('--gpu',default='0')
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--smoke',action='store_true');mode.add_argument('--full',action='store_true');args=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    cfg=ROOT/'gated_mdn/configs/imptc'/(args.config or ('smoke_gated_peds_imptc.json' if args.smoke else 'gated_peds_imptc.json'))
    if args.smoke and json.loads(cfg.read_text())['train_params']['train_epochs']>5:raise ValueError('Smoke is limited to <=5 epochs')
    print(run(cfg,args.run_id,args.resume,args.smoke))
if __name__=='__main__':main()
