"""Sinh hình giải thích từ công thức source và artifact đã lưu; không train/inference model.
History đang chạy được chụp tối đa đến epoch 32; hình dự đoán dùng checkpoint 1,5,10 cố định.
"""
from pathlib import Path
import csv,json
from datetime import datetime,timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Ellipse
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/'figures'; OUT.mkdir(exist_ok=True)
RUN=ROOT/'results/trained_models/kappa_mdn/imptc/kappa_k16_peds_imptc/runs/kappa_k16_v2_seed2024'
SMOKE=ROOT/'results/trained_models/kappa_mdn/imptc/smoke_kappa_k16_peds_imptc/runs/kappa_smoke_seed2024'
BLUE,ORANGE,GREEN='#2677c9','#df663b','#219b78'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
 fig.tight_layout();fig.savefig(OUT/name,dpi=160,bbox_inches='tight',pad_inches=.18);plt.close(fig)
def box(ax,x,y,w,h,text,color='#e3eefb'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.03',facecolor=color,edgecolor='.45'))
 ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'.3','lw':1.5})
def blank(ax):ax.set_xlim(0,12);ax.set_ylim(0,5);ax.axis('off')
def gate(k,tau,j):return 1/(1+np.exp(np.clip(-(np.asarray(k)[...,None]-j+.5)/tau,-700,700)))
j=np.arange(1,17)
fig,ax=plt.subplots(figsize=(12,4.5));blank(ax)
for x,txt in [(0.1,'Quá khứ X\n[B, 32, 4]'),(3.0,'LSTM → h cuối\n[B, 8]'),(6.0,'fc → μ, σ, ρ, z\n48 bước × 16 Gaussian')]:box(ax,x,2.8,2.5,1.1,txt)
box(ax,9.2,2.8,2.5,1.1,'Train: w = softmax(z) × g\nKHÔNG chuẩn hóa lại','#ffe7da')
for a,b in [(2.6,3),(5.5,6),(8.5,9.2)]:arrow(ax,(a,3.35),(b,3.35))
box(ax,6,0.5,2.5,1.1,'48 tham số kappa_logit\nκ = 1 + 15 sigmoid(a); init 12','#dbf1e8');box(ax,9.2,.5,2.5,1.1,'Cổng g(κ, j, τ)\nchung cho mọi mẫu','#dbf1e8');arrow(ax,(8.5,1.05),(9.2,1.05));arrow(ax,(10.45,1.6),(10.45,2.8))
ax.set_title('Sơ đồ train: v2: kappa suy ra từ tham số riêng, không phải đầu ra LSTM');save(fig,'01_pipeline.png')
fig,ax=plt.subplots(figsize=(11,3.6));blank(ax)
box(ax,.2,3,3.3,1,'Một vector κ[48]\nđược lưu trong checkpoint','#dbf1e8')
for y,tag in [(1.6,'Mẫu test A'),(.2,'Mẫu test B')]:
 box(ax,4.3,y,2.2,1,tag);box(ax,7.4,y,4.3,1,'Cùng K(t), cùng cổng mở\nμ, Σ, pi có thể khác nhau','#ffe7da');arrow(ax,(3.5,3.5),(4.3,y+.5));arrow(ax,(6.5,y+.5),(7.4,y+.5))
ax.set_title('K(t) chung theo bước tương lai; phân phối vẫn phụ thuộc đầu vào');save(fig,'02_shared_k.png')
fig,axs=plt.subplots(1,2,figsize=(11,4))
for tau in [1,.5,.1]:
 axs[0].plot(j,gate(3.2,tau,j),'-o',label=f'τ={tau}')
 xx=np.linspace(1,6,300);axs[1].plot(xx,gate(xx,tau,np.array([4]))[:,0],label=f'τ={tau}')
axs[0].step(j,(j<=3).astype(float),where='mid',color='black',ls='--',label='Eval: K=3')
axs[0].set(xlabel='Chỉ số Gaussian j',ylabel='Độ mở g',title='κ=3.2: cổng giảm theo chỉ số');axs[1].set(xlabel='κ',ylabel='Độ mở g của Gaussian j=4',title='Điểm giữa ở κ=3.5, g=0.5')
for ax in axs:ax.legend();ax.grid(alpha=.2)
save(fig,'03_gate_curves.png')
# Example weights, deliberately not measured model data.
z=np.array([1,.2,.6,-.4,.3,-.2]);jj=np.arange(1,7);g=gate(3.2,.5,jj);base=np.exp(z-z.max());base/=base.sum();soft=base*g;hard=base*(jj<=3);hard/=hard.sum()
fig,axs=plt.subplots(1,3,figsize=(12,3.6))
for ax,values,title in zip(axs,[g,soft,hard],['Cổng g (không cần tổng = 1)','Train v2: w = softmax(z) × g','Eval: chỉ j=1,2,3 có pi dương']):
 ax.bar(jj,values,color=BLUE);ax.set_title(title,fontsize=10);ax.set_xlabel('j');ax.set_ylim(0,1)
 for a,b in zip(jj,values):ax.text(a,b+.02,f'{b:.3f}',ha='center',fontsize=8)
fig.suptitle('Ví dụ giả định κ=3.2; τ=0.5; K_max=6',y=1.03);save(fig,'04_weight_example.png')
fig,ax=plt.subplots(figsize=(11.5,4));blank(ax)
for y,a,b,c in [(3,'Train','g = sigmoid((κ − j + 0.5)/τ)','Surrogate mất khối lượng + penalty'),(1,'Validation / test','K = clamp(round(κ), 1, K_max)','NLL cứng; không penalty')]:
 box(ax,.1,y,2.1,1,a);box(ax,3,y,4.1,1,b,'#dbf1e8');box(ax,8,y,3.8,1,c,'#ffe7da');arrow(ax,(2.2,y+.5),(3,y+.5));arrow(ax,(7.1,y+.5),(8,y+.5))
ax.set_title('Train và eval dùng hai nhánh khác nhau trong forward');save(fig,'05_train_eval.png')
fig,axs=plt.subplots(1,2,figsize=(11,4));ks=np.linspace(-1,18,400)
for tau in [1,.1]:axs[0].plot(ks,gate(ks,tau,j).sum(-1),label=f'τ={tau}')
axs[0].plot(ks,np.clip(np.round(ks),1,16),'k--',label='K cứng');axs[0].set(xlabel='κ',ylabel='Số cổng',title='Số cổng mềm và K cứng');axs[0].legend()
for tau in [1,.5,.1]:
 gg=gate(ks,tau,j);axs[1].plot(ks,.01*(gg*(1-gg)/tau).sum(-1)/48,label=f'τ={tau}')
axs[1].set(xlabel='Một κ_t',ylabel='∂ penalty / ∂κ_t',title='Penalty luôn đẩy κ xuống khi gradient > 0');axs[1].legend()
for ax in axs:ax.grid(alpha=.2)
save(fig,'06_count_gradient.png')
fig,axs=plt.subplots(1,2,figsize=(11,3.7));e=np.arange(1,2501);tau=1-.9*np.clip((e-1)/1249,0,1)
axs[0].plot(e,tau,color=BLUE);axs[0].set(xlabel='Epoch',ylabel='τ',title='Lịch được đặt trước: 1 → 0.1 ở epoch 1250')
for kap in [2,8,14]:axs[1].plot(j,gate(kap,.1,j),'-o',label=f'κ={kap}')
axs[1].set(xlabel='Chỉ số j',ylabel='g',title='Cổng có thứ tự: ưu tiên các chỉ số đầu');axs[1].legend()
for ax in axs:ax.grid(alpha=.2)
save(fig,'07_schedule_prefix.png')
fig,ax=plt.subplots(figsize=(11,4));blank(ax)
for y,title,body in [(3,'Sparsemax','z(X,t) → ngưỡng → K(X,t)\nSupport tùy mẫu; có thể giữ {2,5,9}'),(1,'Kappa MDN','κ_t được học → round → K(t)\nSupport chung mọi mẫu: {1,2,...,K(t)}')]:
 box(ax,.2,y,3,1,title);box(ax,4,y,7.5,1,body,'#ffe7da' if y==3 else '#dbf1e8');arrow(ax,(3.2,y+.5),(4,y+.5))
ax.set_title('Hai cơ chế khác nhau để tạo số Gaussian hoạt động');save(fig,'08_compare_sparsemax.png')
fig,axs=plt.subplots(1,2,figsize=(11,4))
a=np.linspace(-7,7,300);axs[0].plot(a,1+15/(1+np.exp(-a)),color=BLUE);axs[0].axhline(1,c='.6',ls='--');axs[0].axhline(16,c='.6',ls='--');axs[0].set(xlabel='a = kappa_logit',ylabel='κ',title='v2: giới hạn κ bằng sigmoid')
zz=np.linspace(.001,1,300);axs[1].plot(zz,-np.log(zz),color=ORANGE);axs[1].set(xlabel='Z = tổng trọng số sau cổng',ylabel='−log Z',title='Mất khối lượng tạo chi phí trong loss')
for ax in axs:ax.grid(alpha=.2)
save(fig,'16_bounded_mass_loss.png')
# Freeze history snapshot for the running experiment.
with (RUN/'history.csv').open() as f:rows=[r for r in csv.DictReader(f) if int(r['epoch'])<=32]
with (SMOKE/'history.csv').open() as f:smoke=list(csv.DictReader(f))
inputs=np.load(RUN/'fixed_samples/inputs.npz');preds={e:np.load(RUN/f'fixed_samples/predictions/epoch_{e:04d}.npz') for e in [1,5,10]}
for p in preds.values():assert np.array_equal(inputs['sample_ids'],p['sample_ids'])
fig,axs=plt.subplots(1,2,figsize=(12,4))
for e,p in preds.items():
 kap=p['kappa'];axs[0].plot(np.arange(1,49),kap,label=f'Epoch {e}');axs[1].step(np.arange(1,49),np.clip(np.round(kap),1,16),label=f'Epoch {e}')
axs[0].axhline(12,c='.5',ls='--',label='Khởi tạo');axs[0].set(xlabel='Bước tương lai',ylabel='κ_t',title='Tham số thực đã lưu');axs[1].set(xlabel='Bước tương lai',ylabel='K(t)',title='K sau làm tròn')
for ax in axs:ax.legend();ax.grid(alpha=.2)
fig.suptitle('Artifact thật — giai đoạn đầu run đầy đủ; chưa phải kết quả cuối',y=1.03);save(fig,'09_real_kappa.png')
fig,axs=plt.subplots(1,3,figsize=(14,4));ep=[int(r['epoch']) for r in rows]
for key,title in [('train_nll','Train: surrogate mềm'),('validation_nll','Validation: cổng cứng')]:axs[0].plot(ep,[float(r[key]) for r in rows],label=title)
axs[0].set(title='NLL theo hai nhánh',xlabel='Epoch',ylabel='NLL');axs[0].legend(fontsize=8)
for key,title in [('train_objective','Objective'),('train_penalty','Penalty')]:axs[1].plot(ep,[float(r[key]) for r in rows],label=title)
axs[1].set(title='Objective và penalty',xlabel='Epoch');axs[1].legend()
for key,title in [('kappa_mean','κ trung bình'),('mean_K','K cứng trung bình'),('soft_open_count','Số cổng mềm')]:axs[2].plot(ep,[float(r[key]) for r in rows],label=title)
axs[2].set(title='K mềm / cứng',xlabel='Epoch');axs[2].legend(fontsize=8)
for ax in axs:ax.grid(alpha=.2)
fig.suptitle(f'History thật, snapshot đến epoch {ep[-1]}; không chứng minh hội tụ',y=1.03);save(fig,'10_real_history.png')
p=preds[10];pi,mu,cov=p['pi'],p['mu'],p['covariance'];K=np.clip(np.round(p['kappa']),1,16)
fig,axs=plt.subplots(1,3,figsize=(14,4))
im=axs[0].imshow(p['support_size'],aspect='auto',vmin=1,vmax=16);fig.colorbar(im,ax=axs[0],label='Support');axs[0].set(title='Số pi > 0: 8 mẫu × 48 bước',xlabel='Bước (index từ 0)',ylabel='Mẫu (index từ 0)')
for ax,s in zip(axs[1:],[0,2]):
 masked=np.ma.masked_where(pi[s].T<=0,pi[s].T);cm=plt.get_cmap('viridis').copy();cm.set_bad('#eeeeee')
 im=ax.imshow(masked,aspect='auto',vmin=0,vmax=1,cmap=cm);fig.colorbar(im,ax=ax,label='pi');ax.set(title=f'Trọng số mẫu {s+1}',xlabel='Bước (index từ 0)',ylabel='Gaussian (index từ 0)')
fig.suptitle('Epoch 10: support chung theo bước, trọng số khác theo mẫu; xám = 0',y=1.03);save(fig,'11_real_support.png')
weighted=np.einsum('ntk,ntkd->ntd',pi,mu);err=np.linalg.norm(weighted-inputs['y'],axis=-1).mean(-1);picks=[int(err.argmin()),int(np.argsort(err)[4]),int(err.argmax())]
fig,axs=plt.subplots(1,3,figsize=(14,4))
for ax,s,tag in zip(axs,picks,['Sai số thấp nhất','Sai số trung vị','Sai số cao nhất']):
 ax.plot(*inputs['X'][s,:,:2].T,'o-',ms=2,label='Quan sát');ax.plot(*inputs['y'][s].T,'k-',label='Thật');ax.plot(*weighted[s].T,'--',c=ORANGE,label='Trung bình hỗn hợp');ax.set(title=f'{tag} trong 8 mẫu\nMẫu {s+1}; ADE minh họa {err[s]:.2f} m',xlabel='x (m)',ylabel='y (m)');ax.axis('equal');ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.suptitle('Epoch 10 — model còn ở giai đoạn đầu; ADE minh họa không phải minADE20',y=1.03);save(fig,'12_real_trajectories.png')
s=picks[2];fig,axs=plt.subplots(1,3,figsize=(14,4))
for ax,(e,q) in zip(axs,preds.items()):
 m=np.einsum('tk,tkd->td',q['pi'][s],q['mu'][s]);ax.plot(*inputs['X'][s,:,:2].T,label='Quan sát');ax.plot(*inputs['y'][s].T,'k-',label='Thật');ax.plot(*m.T,'--',c=ORANGE,label='Trung bình');ax.set(title=f'Epoch {e}',xlabel='x (m)',ylabel='y (m)');ax.axis('equal');ax.grid(alpha=.2)
axs[0].legend(fontsize=8);fig.suptitle(f'Cùng mẫu {s+1}, cùng ID và ground truth, qua 3 checkpoint đầu',y=1.03);save(fig,'13_real_fixed_epochs.png')
fig,axs=plt.subplots(1,3,figsize=(14,4.5))
for ax,t in zip(axs,[7,23,47]):
 for k in np.flatnonzero(pi[s,t]>0):
  vals,vec=np.linalg.eigh(cov[s,t,k]);ang=np.degrees(np.arctan2(vec[1,1],vec[0,1]));ax.add_patch(Ellipse(mu[s,t,k],4*np.sqrt(vals[1]),4*np.sqrt(vals[0]),angle=ang,alpha=.15));ax.scatter(*mu[s,t,k],s=25)
 ax.scatter(*inputs['y'][s,t],marker='*',c='black',s=90,label='Ground truth');ax.scatter(*weighted[s,t],marker='x',c=ORANGE,s=60,label='Trung bình')
 ax.text(.02,.98,'\n'.join(f'j{k+1}: pi={pi[s,t,k]:.3f}' for k in np.flatnonzero(pi[s,t]>0)),transform=ax.transAxes,va='top',fontsize=8,bbox=dict(facecolor='white',alpha=.8,edgecolor='.8'))
 ax.set(title=f't={(t+1)*.1:.1f} s; K={int(K[t])}',xlabel='x (m)',ylabel='y (m)');ax.axis('equal');ax.grid(alpha=.2);ax.legend(loc='lower right',fontsize=8)
fig.suptitle(f'Mẫu {s+1}, epoch 10: ellipse từng Gaussian, bán kính Mahalanobis 2',y=1.03);save(fig,'14_real_gaussians.png')
metrics=[json.loads((SMOKE/f'metrics/epoch_{e:04d}.json').read_text())['metrics'] for e in [1,2,3]]
spec=[('ravg_percent','Ravg (%)'),('rmin_percent','Rmin (%)'),('s68_m2_per_s','S68'),('s95_m2_per_s','S95'),('minade20_m','minADE20 (m)'),('minfde20_m','minFDE20 (m)')]
fig,axs=plt.subplots(2,3,figsize=(12,6))
for ax,(key,title) in zip(axs.flat,spec):ax.plot([1,2,3],[m[key] for m in metrics],'-o');ax.set(title=title,xlabel='Epoch smoke');ax.set_xticks([1,2,3]);ax.grid(alpha=.2)
fig.suptitle('Metric validation thật của SMOKE 3 epoch — kiểm tra pipeline, không kết luận chất lượng',y=1.02);save(fig,'15_smoke_metrics.png')
info={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'architecture':'kappa_mdn_v2','full_run':str(RUN.relative_to(ROOT)),'full_status_at_read':json.loads((RUN/'run_manifest.json').read_text())['status'],'history_snapshot_last_epoch':ep[-1],'history_snapshot_last_row':rows[-1],'fixed_epochs':[1,5,10],'selected_sample_indices_zero_based':picks,'illustrative_mean_ade_m':err.tolist(),'smoke_run':str(SMOKE.relative_to(ROOT)),'epoch10_kappa':p['kappa'].tolist(),'epoch10_hard_K':K.tolist()}
(OUT/'figure_info.json').write_text(json.dumps(info,indent=2));(OUT/'history_snapshot.json').write_text(json.dumps(rows,indent=2));print(json.dumps({'figures':16,'history_through_epoch':ep[-1],'sample_indices':picks,'epoch10_mean_K':float(K.mean())}))
