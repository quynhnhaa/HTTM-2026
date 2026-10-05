"""Reproducible teaching illustrations plus an immutable snapshot of the live gate log."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse
ROOT=Path(__file__).resolve().parents[2];DEST=Path(__file__).resolve().parent
IM=DEST/'images';IM.mkdir(exist_ok=True)
RUN=ROOT/'results/trained_models/gated_mdn/imptc/gated_peds_imptc/runs/gated_k8_seed2024_tmux'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
BLUE='#2377b6';GREEN='#2c9764';ORANGE='#df8324';RED='#cf4c4c';GRAY='#737d88'

def save(fig,name):fig.savefig(IM/name,dpi=180,bbox_inches='tight');plt.close(fig)
def box(ax,x,y,w,h,text,color=BLUE):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015',facecolor=color,edgecolor='none',alpha=.13))
 ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11)
def arrow(ax,start,end,color=GRAY):ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=14,color=color,lw=1.5))

fig,ax=plt.subplots(figsize=(13,5));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
for x,w,text in [(.02,.18,'Observed trajectory\n32 steps × 4 channels'),(.25,.15,'LSTM\nhidden size 8'),(.46,.19,'MDN head\n48 steps × (6 × 8)'),(.72,.25,'Marginal 2D GMM\n8 candidate components')]:box(ax,x,.56,w,.28,text)
for a,b in [(.20,.25),(.40,.46),(.65,.72)]:arrow(ax,(a,.70),(b,.70))
box(ax,.28,.10,.30,.24,'7 learned GLOBAL gates\n+ 1 always-open gate',GREEN)
arrow(ax,(.58,.23),(.78,.56),GREEN);ax.text(.69,.31,'Modify mixture logits only',color=GREEN,ha='center')
ax.text(.5,.95,'01 | Where the gates enter the baseline',ha='center',fontsize=17)
ax.text(.5,.02,'Means / sigma / rho retain the baseline head; gates are shared across all samples and 48 horizons.',ha='center',fontsize=10)
save(fig,'01_architecture.png')

fig,axes=plt.subplots(2,1,figsize=(13,4.8));demo=np.array([1,.8,0,.65,0,.5,.9,0]);labels=[f'k={i}' for i in range(8)]
for row,ax in enumerate(axes):
 ax.set(xlim=(-.7,7.7),ylim=(0,1.2));ax.axis('off')
 for i in range(8):
  alive=row==0 or demo[i]>0;color=GREEN if alive else GRAY
  box(ax,i-.37,.28,.74,.55,labels[i]+('\nprotected' if i==0 else ''),color)
  if row:ax.text(i,.10,f'gate={demo[i]:.2f}',ha='center',color=color)
 ax.text(-.65,1.01,'START: 8 candidates' if not row else 'ILLUSTRATION: 5 survivors (not current run result)',fontsize=13)
fig.suptitle('02 | K_max is a ceiling; learned active K can be any integer from 1 to 8',fontsize=15)
save(fig,'02_survivors_example.png')

fig,ax=plt.subplots(figsize=(13,5));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
boxes=[(.02,.55,.19,.26,'Sample batch\n+ stochastic gates'),(.28,.55,.19,.26,'Predict GMM\nCompute GT NLL'),(.55,.55,.20,.26,'Objective\nNLL + λ E[K]'),(.79,.55,.18,.26,'Backprop\nAdam update')]
for x,y,w,h,txt in boxes:box(ax,x,y,w,h,txt)
for a,b in [(.21,.28),(.47,.55),(.75,.79)]:arrow(ax,(a,.68),(b,.68))
box(ax,.52,.12,.38,.20,'Update LSTM + MDN + gate log_alpha',GREEN)
arrow(ax,(.88,.55),(.88,.32),GREEN);arrow(ax,(.52,.22),(.11,.55),GREEN)
ax.text(.5,.94,'03 | One optimizer learns predictions and component gates together',ha='center',fontsize=16)
ax.text(.5,.02,'Fresh gate vector each batch; closed gates can reopen during training. No separate run for each K.',ha='center')
save(fig,'03_learning_loop.png')

u=np.linspace(.001,.999,1200);beta=2/3;gamma=-.1;zeta=1.1;alpha=0.
s=1/(1+np.exp(-(np.log(u)-np.log1p(-u)+alpha)/beta));stretch=s*(zeta-gamma)+gamma;z=np.clip(stretch,0,1)
fig,axes=plt.subplots(1,3,figsize=(13,4))
for ax,v,title in zip(axes,[s,stretch,z],['Binary concrete s','Stretch to [-0.1, 1.1]','Clamp to [0,1]: actual gate z']):
 ax.plot(u,v,color=BLUE,lw=2);ax.axhline(0,color=GRAY,lw=.8);ax.axhline(1,color=GRAY,lw=.8);ax.set(xlabel='Random u in (0,1)',ylabel='Value',title=title)
axes[2].fill_between(u,0,1,where=z==0,color=RED,alpha=.12);axes[2].text(.035,.55,'Exact\nzeros',color=RED)
axes[2].fill_between(u,0,1,where=z==1,color=GREEN,alpha=.12);axes[2].text(.78,.25,'Exact\nones',color=GREEN)
fig.suptitle('04 | Why hard-concrete can turn a component OFF exactly (illustration: log_alpha=0)',fontsize=14);fig.tight_layout();save(fig,'04_hard_concrete.png')

x=np.linspace(-8,6,600);det=np.clip((1/(1+np.exp(-x)))*(zeta-gamma)+gamma,0,1);prob=1/(1+np.exp(-(x-beta*np.log(-gamma/zeta))))
fig,ax=plt.subplots(figsize=(10,4.5));ax.plot(x,det,label='Deterministic evaluation gate',color=BLUE,lw=2);ax.plot(x,prob,label='P(stochastic gate > 0)',color=ORANGE,lw=2)
threshold=np.log(-gamma/zeta);ax.axvline(threshold,color=GRAY,ls='--',label=f'Deterministic OFF threshold: {threshold:.3f}')
ax.set(xlabel='Learned log_alpha',ylabel='Value / probability',ylim=(-.03,1.05),title='05 | Gate value and open probability are different quantities');ax.legend(loc='lower right');ax.grid(alpha=.2);save(fig,'05_probability_vs_gate.png')

base=np.array([.15,.25,.15,.10,.10,.10,.10,.05]);g=demo;weighted=base*g;pi=weighted/weighted.sum()
fig,axes=plt.subplots(1,3,figsize=(13,4))
for ax,v,title,col in zip(axes,[base,g,pi],['Ungated mixture weights','Global gates (illustrative)','Renormalized gated weights'],[BLUE,ORANGE,GREEN]):
 ax.bar(np.arange(8),v,color=col);ax.set(xticks=range(8),xlabel='Component index',ylim=(0,1.05) if title.startswith('Global') else (0,.5),title=title)
fig.suptitle('06 | Gate multiplies weight, then all remaining weights are normalized to sum 1',fontsize=14);fig.tight_layout();save(fig,'06_weight_normalization.png')

fig,ax=plt.subplots(figsize=(12,5));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
box(ax,.03,.62,.36,.23,'GLOBAL gate vector\nSame for every sample and horizon',GREEN)
box(ax,.57,.62,.39,.23,'LOCAL MDN logits / pi\nDepend on trajectory and timestep',BLUE)
for i,txt in enumerate(['Pedestrian A at 0.8 s','Pedestrian B at 4.8 s']):
 y=.38-i*.23;box(ax,.36,y,.29,.15,txt,GRAY);arrow(ax,(.20,.62),(.36,y+.075),GREEN);arrow(ax,(.77,.62),(.65,y+.075),BLUE)
ax.text(.5,.97,'07 | Learning a common component subset, not a different K for each pedestrian',ha='center',fontsize=15)
ax.text(.5,.02,'A surviving component may still have tiny pi on a particular sample. Component index is not a joint trajectory mode.',ha='center',fontsize=10)
save(fig,'07_global_vs_local.png')

fig,ax=plt.subplots(figsize=(12,4));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
box(ax,.03,.22,.28,.51,'TRAIN\nRandom gates\nNLL + expected L0 penalty\nAll K_max head rows exist',BLUE)
box(ax,.37,.22,.27,.51,'VALIDATION / TEST\nDeterministic gates\nCount gates > 0\nNo sampling of gate vector',ORANGE)
box(ax,.71,.22,.26,.51,'COMPACT EXPORT\nRemove inactive head rows\nAbsorb log-gate into logits\nKeep same eval distribution',GREEN)
arrow(ax,(.31,.47),(.37,.47));arrow(ax,(.64,.47),(.71,.47));ax.text(.5,.93,'08 | Three stages; computational pruning happens at export',ha='center',fontsize=15)
save(fig,'08_train_eval_export.png')

# Capture complete JSONL records once; final incomplete line is ignored.
records=[]
for line in (RUN/'gate_history.jsonl').read_text().splitlines():
 try:records.append(json.loads(line))
 except json.JSONDecodeError:break
snapshot={'run_id':RUN.name,'scope':'historical snapshot, not live or final results','last_epoch':records[-1]['epoch'],'records':records}
(DEST/'gate_snapshot.json').write_text(json.dumps(snapshot,indent=2)+'\n')
e=np.array([r['epoch'] for r in records]);values=np.array([r['deterministic_gates'] for r in records]);expected=np.array([r['expected_active_components'] for r in records]);active=np.array([r['deterministic_active_components'] for r in records])
fig,axes=plt.subplots(1,3,figsize=(15,4));im=axes[0].imshow(values.T,aspect='auto',origin='lower',vmin=0,vmax=1,extent=[e[0]-.5,e[-1]+.5,-.5,7.5],cmap='YlGnBu');fig.colorbar(im,ax=axes[0],label='Deterministic gate')
axes[0].set(title='Actual gate values',xlabel='Epoch',ylabel='Component index',yticks=range(8))
axes[1].plot(e,expected,color=ORANGE,label='Expected active K');axes[1].plot(e,active,color=BLUE,label='Deterministic active K');axes[1].set(title='Actual K logs (different definitions)',xlabel='Epoch',ylabel='K',ylim=(0,8.4));axes[1].legend(fontsize=8)
axes[2].bar(range(8),values[-1],color=GREEN);axes[2].set(title=f'Actual gates at epoch {e[-1]}',xlabel='Component index',ylabel='Gate',xticks=range(8),ylim=(0,1.05))
fig.suptitle(f'09 | Real snapshot: {RUN.name}, through epoch {e[-1]} — not a final result',fontsize=13);fig.tight_layout();save(fig,'09_actual_gate_snapshot.png')

# Toy marginal GMM: teaching only, not captured predictions.
xx,yy=np.meshgrid(np.linspace(-4,4,240),np.linspace(-3,4,210));points=np.stack([xx,yy],-1)
centers=np.array([[-1.5,0],[0,1.7],[1.5,.2]]);sig=[.55,.65,.75];weights_before=np.array([.3,.3,.4]);weights_after=np.array([.3,0,.4]);weights_after/=weights_after.sum()
fig,axes=plt.subplots(1,2,figsize=(11,4.6))
for ax,weights,title in zip(axes,[weights_before,weights_after],['Before: 3 components','After: middle component gated OFF']):
 density=sum(w*np.exp(-((points-c)**2).sum(-1)/(2*s*s))/(2*np.pi*s*s) for w,c,s in zip(weights,centers,sig))
 ax.contourf(xx,yy,density,levels=18,cmap='Blues');ax.scatter(centers[:,0],centers[:,1],marker='x',color=RED)
 for i,(c,w) in enumerate(zip(centers,weights)):ax.text(c[0],c[1]-.5,f'k{i}: pi={w:.2f}',ha='center')
 ax.set(title=title,xlabel='x (m)',ylabel='y (m)');ax.set_aspect('equal')
fig.suptitle('10 | A gate removes mixture mass, not the observed trajectory (toy distribution)',fontsize=13);fig.tight_layout();save(fig,'10_toy_gmm.png')
print('Created 10 figures; real snapshot epoch',e[-1])
