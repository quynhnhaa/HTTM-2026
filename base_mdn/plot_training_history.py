#!/usr/bin/env python3
"""Plot saved M=3 NLL history, full and zoomed. Never trains/evaluates a model."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
DEFAULT_RUN=ROOT/'results/trained_models/base_mdn/imptc/default_peds_imptc/runs/imptc_baseline_seed2024'
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();run=args.run_dir.resolve()
    rows=list(csv.DictReader((run/'history.csv').open()))
    epoch=np.array([int(r['epoch']) for r in rows]);train=np.array([float(r['train_nll']) for r in rows]);val=np.array([float(r['validation_nll']) for r in rows])
    manifest=json.loads((run/'run_manifest.json').read_text());best=int(manifest['best_epoch']);idx=np.flatnonzero(epoch==best)
    assert len(idx)==1 and len(epoch)==len(np.unique(epoch)), 'Missing or duplicated best epoch'
    assert np.isfinite(train).all() and np.isfinite(val).all()
    best_idx=int(idx[0]);assert best==int(epoch[np.argmin(val)]), 'Best epoch does not match history'
    fig,axes=plt.subplots(2,1,figsize=(16,6.2),layout='constrained')
    for ax,start,title in zip(axes,[1,100],['Entire training history (epochs 1–2500)','Later training (epochs 100–2500)']):
        keep=epoch>=start
        ax.plot(epoch[keep],train[keep],color='#bd5b20',lw=1.5,label='M=3 train')
        ax.plot(epoch[keep],val[keep],color='#f5a26c',lw=1.5,label='M=3 validation')
        ax.axvline(best,color='#555555',ls=':',lw=1.8,label=f'Best epoch = {best}')
        ax.scatter([best],[val[best_idx]],s=65,color='#f5a26c',edgecolors='white',zorder=5)
        ax.set_xlim(start,int(epoch[-1]));ax.set_title(title,loc='left',fontsize=15)
        ax.set_ylabel('NLL',fontsize=14);ax.tick_params(labelsize=12);ax.grid(alpha=.22)
        ax.legend(loc='upper right',fontsize=12,ncol=3,framealpha=.9)
    axes[1].set_xlabel('Epoch',fontsize=14)
    output=args.output or run/'figures/training/train_validation_nll_m3_full_zoom.png';output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=200);plt.close(fig)
    metadata={'image':str(output.relative_to(ROOT)),'script':str(Path(__file__).resolve().relative_to(ROOT)),'history_csv':str((run/'history.csv').relative_to(ROOT)),'manifest':str((run/'run_manifest.json').relative_to(ROOT)),'num_epochs':len(epoch),'best_epoch':best,'best_validation_nll':float(val[best_idx]),'panels':[[1,int(epoch[-1])],[100,int(epoch[-1])]],'sampling':'Every epoch, no smoothing or downsampling','model':'M=3 only'}
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))
if __name__=='__main__':main()
