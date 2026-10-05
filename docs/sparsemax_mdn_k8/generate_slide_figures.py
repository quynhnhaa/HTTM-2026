"""Presentation plots from saved test artifacts. No model training/evaluation."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent/'figures/slides';OUT.mkdir(exist_ok=True)
BLUE='#2563eb';ORANGE='#ed713a';NAVY='#102753'
plt.rcParams.update({'font.size':16,'axes.titlesize':19,'axes.labelsize':16,'xtick.labelsize':14,'ytick.labelsize':14,'axes.spines.top':False,'axes.spines.right':False,'text.color':NAVY,'axes.labelcolor':NAVY,'savefig.dpi':180})
t=json.load(open(ROOT/'results/trained_models/sparsemax_mdn/imptc/sparsemax_k8_peds_imptc/runs/sparsemax_k8_v2_seed2024/evaluation/test_best_clampedpi.json'))
b=next(r for r in json.load(open(ROOT/'results/ablations/imptc_num_gaussians/ablation.json'))['results'] if r['num_gaussians']==3)
s=t['official_metrics']|{'test_nll':t['exact_nll']}
rows=[('NLL ↓','test_nll'),('Ravg (%) ↑','ravg_percent'),('Rmin (%) ↑','rmin_percent'),('minADE20 (m) ↓','minade20_m'),('S68 (m²/s) ↓','s68_m2_per_s'),('S95 (m²/s) ↓','s95_m2_per_s'),('minFDE20 (m) ↓','minfde20_m'),('ASAEE (m/s) ↓','asaee_m_per_s')]
fig,axs=plt.subplots(2,4,figsize=(18,6.6),layout='constrained')
for ax,(title,k) in zip(axs.flat,rows):
 vals=[b[k],s[k]];bars=ax.bar([0,1],vals,color=[BLUE,ORANGE],width=.52);ax.set_xticks([0,1],['K=3','Sparsemax K8']);ax.set_title(title,loc='left',weight='bold',pad=20);ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
 if min(vals)>=0:
  span=max(vals)-min(vals);pad=max(span*1.5,max(vals)*.012);ax.set_ylim(max(0,min(vals)-pad),max(vals)+pad*1.3)
 else:ax.set_ylim(min(vals)*1.16,0)
 for bar,v in zip(bars,vals):ax.annotate(f'{v:.2f}' if k.startswith('r') else f'{v:.3f}',(bar.get_x()+bar.get_width()/2,v),ha='center',va='bottom' if v>=0 else 'top',xytext=(0,7 if v>=0 else -5),textcoords='offset points',fontsize=19,weight='bold',color=NAVY)
fig.savefig(OUT/'14_test_metrics_slide.png');plt.close(fig)
a=json.load(open(Path(__file__).resolve().parent/'data/s95_explained.json'))['s95']
fig,axs=plt.subplots(1,2,figsize=(15,5.8),gridspec_kw={'width_ratios':[1,1.75]},layout='constrained')
ax=axs[0]
for i,key,col in [(0,'K3',BLUE),(1,'K8',ORANGE)]:
 body=sum(r['body'] for r in a[key]);tail=sum(r['p100'] for r in a[key]);ax.bar(i,body,color=col,width=.55);ax.bar(i,tail,bottom=body,color=col,alpha=.4,hatch='///',width=.55)
 ax.text(i,body/2,f'Thân\n{body:.2f}',ha='center',va='center',color='white',fontsize=21,weight='bold');ax.text(i,body+tail/2,f'p100\n{tail:.2f}',ha='center',va='center',fontsize=19);ax.text(i,body+tail+.16,f'{body+tail:.2f}',ha='center',fontsize=22,weight='bold')
ax.set_xticks([0,1],['Baseline K3','Sparsemax K8']);ax.set_ylim(0,8);ax.set_title('Phần thân gần bằng nhau',loc='left',weight='bold',pad=18);ax.set_ylabel('Đóng góp vào S95');ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
ax=axs[1];xx=np.arange(6)
for off,key,col in [(-.17,'K3',BLUE),(.17,'K8',ORANGE)]:
 body=[r['body'] for r in a[key]];tail=[r['p100'] for r in a[key]];ax.bar(xx+off,body,width=.32,color=col,label='Baseline K3' if key=='K3' else 'Sparsemax K8');ax.bar(xx+off,tail,bottom=body,width=.32,color=col,alpha=.4,hatch='///')
ax.set_xticks(xx,['0.8','1.6','2.4','3.2','4.0','4.8']);ax.set_xlabel('Horizon (giây)');ax.set_title('Chênh lệch nổi bật tại 2,4 giây',loc='left',weight='bold',pad=18);ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True);ax.legend(fontsize=13,loc='upper left');ax.annotate('p100: 0,968 vs 0,110',xy=(2.17,1.31),xytext=(1.5,2.1),arrowprops={'arrowstyle':'->','color':NAVY},fontsize=17)
fig.savefig(OUT/'18_s95_explained_slide.png');plt.close(fig)
(OUT/'manifest.json').write_text(json.dumps({'script':'docs/sparsemax_mdn_k8/generate_slide_figures.py','figures':['14_test_metrics_slide.png','18_s95_explained_slide.png'],'sources':['evaluation/test_best_clampedpi.json (K8)','results/ablations/imptc_num_gaussians/ablation.json (K3)','docs/sparsemax_mdn_k8/data/s95_explained.json'],'method':'Replot saved data for readability. No inference/evaluation.'},indent=2))
print(OUT)
