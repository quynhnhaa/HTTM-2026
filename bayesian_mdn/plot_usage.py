"""Scientific plots of recorded validation diagnostics, including epoch zero."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',default='bayesian_k8_predictive_qmc_smoke_seed2024');a=p.parse_args()
    run=ROOT/'results/trained_models/bayesian_mdn/imptc/smoke_bayesian_peds_imptc/runs'/a.run_id
    rows=[json.loads(s) for s in (run/'usage_history.jsonl').read_text().splitlines()]
    out=ROOT/'bayesian_mdn/reports/predictive_smoke_figures';out.mkdir(exist_ok=True)
    epochs=[r['epoch'] for r in rows];first,last=rows[0],rows[-1]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,key,label in [(axes[0],'weights','Global mean stick weights'),(axes[1],'mean_pi','Conditional mean pi: full validation')]:
        values=np.asarray([r[key]['mean_global_weights'] if key=='weights' else r[key] for r in rows])
        for k in range(values.shape[-1]):ax.plot(epochs,values[:,k],marker='o',label=f'Gaussian {k+1}')
        ax.set_yscale('log');ax.set_xlabel('Epoch (0 = before training)');ax.set_ylabel('Weight (log scale)');ax.set_title(label);ax.set_xticks(epochs)
    axes[1].legend(loc='center left',bbox_to_anchor=(1,0.5),fontsize=8)
    fig.suptitle('Smoke only: tiny weight changes do not establish learned component selection')
    fig.savefig(out/'01_weights_before_after.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    ax=axes[0]
    ax.plot(epochs,[r['weights']['effective_k_global_mass'] for r in rows],'o-',label='Global weights, 99%')
    ax.plot(epochs,[r['effective_k_mean_conditional_mass']['k'] for r in rows],'s--',label='Mean conditional pi, 99%')
    for c in ['0.95','0.99']:ax.plot(epochs,[r['conditional_k'][c]['mean_k'] for r in rows],'.:',label=f'Mean per-position K, {float(c):.0%}')
    ax.set_ylim(.5,8.5);ax.set_yticks(range(1,9));ax.set_xticks(epochs);ax.set_xlabel('Epoch');ax.set_ylabel('Effective K diagnostic');ax.legend(fontsize=8)
    ax=axes[1];positions=last['positions'];x=np.arange(1,9)
    for i,c in enumerate(['0.95','0.99']):ax.bar(x+(i-.5)*.3,np.asarray(last['conditional_k'][c]['histogram_all_positions'])/positions,width=.3,label=f'{float(c):.0%} mass')
    ax.set_xticks(x);ax.set_xlabel('Per-position effective K');ax.set_ylabel('Fraction of validation positions');ax.legend()
    fig.suptitle('Mass threshold affects K; no Gaussian is physically removed')
    fig.savefig(out/'02_k_diagnostics.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    axes[0].plot(epochs,[r['plugin_nll'] for r in rows],'o-',label='Plug-in mean sticks')
    axes[0].plot(epochs,[r['posterior_predictive_nll'] for r in rows],'s--',label='Posterior predictive, 64 QMC points')
    axes[0].set_xlabel('Epoch');axes[0].set_ylabel('Full validation NLL');axes[0].set_xticks(epochs);axes[0].legend()
    axes[1].bar(epochs,[r['delta_predictive_minus_plugin'] for r in rows]);axes[1].axhline(0,color='black',lw=.8)
    axes[1].set_xlabel('Epoch');axes[1].set_ylabel('Predictive NLL - plug-in NLL');axes[1].set_xticks(epochs)
    fig.suptitle('Comparison of inference methods; smoke is not a converged experiment')
    fig.savefig(out/'03_inference_nll.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    x=np.arange(1,9)
    for ax,row,label in [(axes[0],first,'Before training'),(axes[1],last,f'Epoch {last["epoch"]}')]:
        ax.bar(x-.17,row['mean_pi'],width=.34,label='Predictive pi')
        ax.bar(x+.17,row['mean_responsibility'],width=.34,label='GT responsibility')
        ax.set_xticks(x);ax.set_xlabel('Gaussian component');ax.set_ylabel('Mean over validation sample x horizon');ax.set_title(label);ax.legend()
    fig.suptitle('Predicted weight vs membership given ground truth (not interchangeable)')
    fig.savefig(out/'04_pi_vs_responsibility.png',dpi=160);plt.close(fig)
    print(out)
if __name__=='__main__':main()
