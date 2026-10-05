"""Full-split conditional mixture usage and posterior-predictive NLL diagnostics."""
import math
import numpy as np
import torch
from utils.mdn_distribution import build_mdn_distribution


def mass_counts(pi,coverage):
    ordered=pi.sort(dim=-1,descending=True).values
    return (ordered.cumsum(-1)<coverage).sum(-1).add(1).clamp_max(pi.shape[-1])


def mass_summary(pi,coverage):
    order=np.argsort(-np.asarray(pi))
    count=min(len(order),int(np.searchsorted(np.cumsum(np.asarray(pi)[order]),coverage))+1)
    return {'k':count,'indices':order[:count].tolist()}


@torch.no_grad()
def assess_split(model,X,y,device,batch_size,draws=64,seed=2024):
    if len(X)<1 or len(X)!=len(y):raise ValueError('Nonempty matching inputs/targets required')
    was_training=model.training
    k=model.k;h=model.forecast_horizon
    pi_sum=np.zeros((h,k));resp_sum=np.zeros((h,k));hist={c:np.zeros((h,k),dtype=np.int64) for c in (.95,.99)}
    nll_plugin=0.;nll_predictive=0.
    try:
        with model.inference('posterior_predictive',draws,seed):
            bank=model._posterior_bank
            for start in range(0,len(X),batch_size):
                stop=min(start+batch_size,len(X));n=stop-start
                inp=torch.as_tensor(X[start:stop],dtype=torch.float32,device=device)
                target=torch.as_tensor(y[start:stop],dtype=torch.float32,device=device)
                raw=model.ungated_output(inp)
                plugin=model.output_with_weights(raw,model.stick_weights(False))
                predictive=model.output_with_weights(raw,bank)
                mixture=build_mdn_distribution(predictive,k)
                lp=mixture.log_prob(target)
                nll_predictive+=float(-lp.mean())*n
                nll_plugin+=float(-build_mdn_distribution(plugin,k).log_prob(target).mean())*n
                pi=torch.softmax(predictive[...,5*k:],-1)
                # Responsibilities: posterior membership given actual future point.
                log_resp=mixture.component_distribution.log_prob(target.unsqueeze(-2))+pi.log()
                resp=torch.softmax(log_resp,-1)
                if not (torch.isfinite(lp).all() and torch.isfinite(resp).all()):raise FloatingPointError('Non-finite diagnostics')
                pi_sum+=pi.sum(0).cpu().double().numpy()
                resp_sum+=resp.sum(0).cpu().double().numpy()
                for coverage in hist:
                    counts=mass_counts(pi,coverage)
                    hist[coverage]+=torch.stack([(counts==j).sum(0) for j in range(1,k+1)],dim=-1).cpu().numpy()
            metadata=model.weight_report()
    finally:
        model.train(was_training)
    mean_pi=pi_sum/len(X);mean_resp=resp_sum/len(X)
    coverage=float(model.settings['mass_coverage'])
    conditional={str(c):{'histogram_by_horizon':a.tolist(),
        'histogram_all_positions':a.sum(0).tolist(),
        'mean_k':float((a*np.arange(1,k+1)).sum()/(len(X)*h))} for c,a in hist.items()}
    return {'sample_count':len(X),'horizon_count':h,'positions':len(X)*h,
        'inference':'posterior_predictive_qmc','posterior_draws':draws,'posterior_seed':seed,'posterior_draw_method':'scrambled_sobol_beta_inverse_cdf',
        'plugin_nll':nll_plugin/len(X),'posterior_predictive_nll':nll_predictive/len(X),
        'delta_predictive_minus_plugin':(nll_predictive-nll_plugin)/len(X),
        'weights':metadata,'mean_pi_by_horizon':mean_pi.tolist(),
        'mean_responsibility_by_horizon':mean_resp.tolist(),
        'mean_pi':mean_pi.mean(0).tolist(),'mean_responsibility':mean_resp.mean(0).tolist(),
        'effective_k_mean_conditional_mass':mass_summary(mean_pi.mean(0),coverage),
        'effective_k_by_horizon':[mass_summary(row,coverage) for row in mean_pi],
        'conditional_k':conditional,
        'note':'Usage and mass counts are diagnostics, not proof of safe pruning or optimal K. Responsibilities depend on ground truth.'}


@torch.no_grad()
def compare_mc_budgets(model,X,y,device,seed=2024):
    """Fixed validation only; do not tune MC count on test."""
    reports={str(s):assess_split(model,X,y,device,len(X),s,seed) for s in (16,64,256)}
    reference=reports['256']
    seed_reports={str(s):assess_split(model,X,y,device,len(X),64,s) for s in (seed,seed+1,seed+2)}
    return {'split':'fixed_validation','sample_count':len(X),'seed':seed,
        'budgets':{s:{'nll':r['posterior_predictive_nll'],
            'nll_delta_vs_256':r['posterior_predictive_nll']-reference['posterior_predictive_nll'],
            'max_mean_pi_delta_vs_256':float(np.max(np.abs(np.asarray(r['mean_pi_by_horizon'])-np.asarray(reference['mean_pi_by_horizon']))))} for s,r in reports.items()},
        'seed_sensitivity_64':{s:{'nll':r['posterior_predictive_nll'],
            'max_mean_pi_delta_vs_256':float(np.max(np.abs(np.asarray(r['mean_pi_by_horizon'])-np.asarray(reference['mean_pi_by_horizon']))))} for s,r in seed_reports.items()},
        'method':'scrambled_sobol_beta_inverse_cdf',
        'note':'Finite randomized QMC error diagnostic only; 256 points is a reference approximation, not ground truth. Points are not iid.'}
