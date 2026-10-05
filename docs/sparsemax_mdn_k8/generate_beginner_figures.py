"""Vẽ từ artifact đã lưu; không huấn luyện hoặc suy luận lại mô hình."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'results/trained_models/sparsemax_mdn/imptc/sparsemax_k8_peds_imptc/runs/sparsemax_k8_v2_seed2024'
OUT = Path(__file__).resolve().parent/'figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
d = np.load(RUN/'fixed_samples/inputs.npz')
p = np.load(RUN/'fixed_samples/predictions/best.npz')
pi, mu, cov = p['pi'], p['mu'], p['covariance']
weighted = np.einsum('ntk,ntkd->ntd',pi,mu)
errors = np.linalg.norm(weighted-d['y'],axis=-1).mean(-1)
picks = [int(errors.argmin()),int(np.argsort(errors)[len(errors)//2]),int(errors.argmax())]
def save(fig,name):
 fig.tight_layout(); fig.savefig(OUT/name,dpi=160,bbox_inches='tight'); plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(14,4.4))
for ax,s,tag in zip(axs,picks,['Sai số thấp nhất','Sai số trung vị','Sai số cao nhất']):
 ax.plot(*d['X'][s,:,:2].T,'o-',ms=2,label='Quan sát')
 ax.plot(*d['y'][s].T,'k-',label='Tương lai thật')
 ax.plot(*weighted[s].T,'--',color='#eb6834',label='Trung bình hỗn hợp')
 ax.set_title(f'{tag} trong 8 mẫu\nMẫu {s+1}; ADE minh họa = {errors[s]:.3f} m')
 ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.axis('equal'); ax.grid(alpha=.2); ax.legend(fontsize=8)
fig.suptitle(f'Artifact validation thật — best epoch {int(p["epoch"])}; không phải minADE20',y=1.03)
save(fig,'14_real_trajectories.png')
s=picks[2]
fig,axs=plt.subplots(1,3,figsize=(14,4.5))
for ax,t in zip(axs,[7,23,47]):
 ax.plot(*d['y'][s].T,color='.8',label='Quỹ đạo thật')
 for k in np.flatnonzero(pi[s,t]>0):
  vals,vecs=np.linalg.eigh(cov[s,t,k]); angle=np.degrees(np.arctan2(vecs[1,1],vecs[0,1]))
  # Mahalanobis radius 2: component coverage 1-exp(-2), not a mixture confidence set.
  ell=Ellipse(mu[s,t,k],4*np.sqrt(vals[1]),4*np.sqrt(vals[0]),angle=angle,alpha=.20)
  ax.add_patch(ell); ax.scatter(*mu[s,t,k],s=30)
 ax.text(.03,.97,'\n'.join(f'k{k+1}: π={pi[s,t,k]:.3f}' for k in np.flatnonzero(pi[s,t]>0)),transform=ax.transAxes,va='top',fontsize=9,bbox=dict(facecolor='white',alpha=.85,edgecolor='.8'))
 ax.scatter(*d['y'][s,t],marker='*',s=120,c='black',label='Vị trí thật')
 ax.scatter(*weighted[s,t],marker='x',s=70,c='#eb6834',label='Trung bình hỗn hợp')
 ax.set_title(f't={(t+1)*.1:.1f} s; K hoạt động={(pi[s,t]>0).sum()}'); ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.axis('equal'); ax.grid(alpha=.2); ax.legend(fontsize=8)
fig.suptitle(f'Mẫu {s+1}, best: ellipse từng Gaussian, bán kính Mahalanobis 2',y=1.03)
save(fig,'15_real_gaussians.png')
fig,axs=plt.subplots(1,4,figsize=(16,4))
for ax,label in zip(axs,['epoch_0001','epoch_0005','epoch_0010','best']):
 q=np.load(RUN/f'fixed_samples/predictions/{label}.npz')
 assert np.array_equal(q['sample_ids'],d['sample_ids'])
 mean=np.einsum('tk,tkd->td',q['pi'][s],q['mu'][s])
 ax.plot(*d['X'][s,:,:2].T,label='Quan sát'); ax.plot(*d['y'][s].T,'k-',label='Thật'); ax.plot(*mean.T,'--',color='#eb6834',label='Trung bình')
 ax.set_title(f'Epoch {int(q["epoch"])}'); ax.axis('equal'); ax.grid(alpha=.2); ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
axs[0].legend(fontsize=8); fig.suptitle(f'Cùng mẫu {s+1}, cùng quan sát và ground truth; chỉ dự đoán thay đổi',y=1.03)
save(fig,'16_same_sample_epochs.png')
with (RUN/'history.csv').open() as f: rows=list(csv.DictReader(f))
fig,ax=plt.subplots(figsize=(10,4))
for key,title in [('train_nll','Train'),('validation_nll','Validation')]:
 ax.plot([int(r['epoch']) for r in rows],[float(r[key]) for r in rows],label=title,lw=.7,alpha=.8)
ax.set_xlabel('Epoch'); ax.set_ylabel('NLL (nat)'); ax.set_title('Lịch sử thật: toàn bộ epoch, không cắt giai đoạn đầu'); ax.legend(); ax.grid(alpha=.2)
save(fig,'17_train_validation_nll.png')
print('Created 4 figures; selected sample indices:',picks)
