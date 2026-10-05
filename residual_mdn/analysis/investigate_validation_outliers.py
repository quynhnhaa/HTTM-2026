"""Inspect validation-only confidence-area outliers and raw Gaussian heads."""
import json,sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from compare_models import ROOT,OUT,JOBS,COLORS
from scan_distribution import load_class,stats
sys.path.insert(0,str(ROOT/'stable_mdn'))
from base_lstm import LSTM_Trajectory_Forecast
from utils.config_loader import ConfigLoader
from utils.data_loader import DataLoader
from utils.mdn_distribution import decode_mdn_output

DEST=OUT/'validation_outliers'
AUDIT=ROOT/'results/comparisons/confidence_area_audit/full'

def features(X):
 return {'max_speed_m_s':np.linalg.norm(X[...,2:4],axis=-1).max(1),
 'recent5_mean_velocity_norm_m_s':np.linalg.norm(X[:,-5:,2:4].mean(1),axis=-1),
 'max_acceleration_m_s2':np.linalg.norm(np.diff(X[...,2:4],axis=1)/.1,axis=-1).max(1),
 'observed_extent_m':np.linalg.norm(X[...,:2]-X[:,-1:,:2],axis=-1).max(1)}

@torch.no_grad()
def main():
 DEST.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
 device='cuda' if torch.cuda.is_available() else 'cpu'
 cfg=ConfigLoader(str(ROOT/'stable_mdn/configs/imptc/stable_peds_imptc.json'),'imptc',False,False,'stable_peds_imptc','stable_mdn','testing')
 loader=DataLoader(cfg);loader.load_train_data();loader.load_eval_data()
 trainX=loader.train_data[0];X,y=loader.eval_data[:2];f=features(X);tf=features(trainX)
 indices=set();selectors={}
 for name,arch,_,_ in JOBS:
  d=json.loads((AUDIT/arch/'validation/summary.json').read_text())
  for level,m in d['confidence'].items():
   for hi,t in enumerate(d['metadata']['times_seconds']):
    top=m['top_indices_per_horizon'][hi][:2];indices.update(top);selectors[f'{name}:{level}:{t:.1f}']=top
 indices=np.asarray(sorted(indices),dtype=int);payload={'indices':indices,'X':X[indices],'y':y[indices],
    'sample_ids':np.asarray(loader.sample_ids['eval'])[indices]}
 summary={'scope':'validation only; no test-based selection or training','selection':'top2 confidence-area samples per model/level/horizon; union',
  'selected_count':len(indices),'selectors':selectors,'input_statistics':{'train':{k:stats(v) for k,v in tf.items()},'validation':{k:stats(v) for k,v in f.items()}},
  'velocity_recompute_max_error':float(np.abs(X[:,1:,2:4]-np.diff(X[:,:,:2],axis=1)/.1).max()),'models':{},'samples':[]}
 decoded=[]
 for name,arch,config,prefix in JOBS:
  run=ROOT/'results/trained_models'/arch/'imptc'/config/'runs'/(prefix+'_cont500')
  cp=torch.load(run/'checkpoints/best.pt',map_location=device,weights_only=False);params=cp['resolved_config']['model_params']
  cls=LSTM_Trajectory_Forecast if name=='Baseline' else load_class(ROOT/arch/'model.py','AttentionMDN' if name=='Attention' else 'ResidualMDN')
  model=cls(params).to(device);model.load_state_dict(cp['model_state_dict']);model.eval()
  # All validation output distributions for percentile comparisons of selected cases.
  allraw=[]
  for start in range(0,len(X),1024):allraw.append(model(torch.as_tensor(X[start:start+1024],dtype=torch.float32,device=device)).cpu())
  raw=torch.cat(allraw);p={k:v.numpy() for k,v in decode_mdn_output(raw,3,params['mdn_parameterization']).items()}
  decoded.append(p);payload[name+'_raw']=raw.numpy()[indices]
  for k,v in p.items():payload[name+'_'+k]=v[indices]
  area=np.load(AUDIT/arch/'validation/areas_m2.npy',mmap_mode='r');payload[name+'_areas_m2']=area[indices]
  wtrace=p['pi']*np.trace(p['covariance'],axis1=-2,axis2=-1)
  summary['models'][name]={'epoch':int(cp['epoch']),'max_raw_log_sigma_statistics':stats(raw[...,6:12].numpy().max((1,2))),
    'max_sigma_statistics':stats(p['sigma'].max((1,2,3))),
    'largest_weighted_component_trace_statistics':stats(wtrace.max((1,2)))}
  if name=='Residual':
   inp=torch.as_tensor(X[indices],dtype=torch.float32,device=device);res=model.residual_output(inp);cv=model.cv_trajectory(inp)
   # Direct forward prior modifies mean entries only; covariance outputs identical.
   np.testing.assert_array_equal(res[...,6:].cpu().numpy(),model(inp)[...,6:].cpu().numpy())
   np.testing.assert_allclose(res[...,6:].cpu().numpy(),raw.numpy()[indices,...,6:],rtol=1e-4,atol=3e-6)
   payload['CV']=cv.cpu().numpy();payload['raw_residual']=res.cpu().numpy()
 for slot,index in enumerate(indices):
  sample={'index':int(index),'sample_id':loader.sample_ids['eval'][index],'source':str(loader.eval_data[4][index]),
    'inputs':{k:{'value':float(v[index]),'validation_percentile':float((v<=v[index]).mean()*100),'train_percentile':float((tf[k]<=v[index]).mean()*100)} for k,v in f.items()},'models':{}}
  for name,p in zip([j[0] for j in JOBS],decoded):
   sigma=p['sigma'][index];pi=p['pi'][index];mu=p['mu'][index];cov=p['covariance'][index]
   w=pi*np.trace(cov,axis1=-2,axis2=-1);h,k=np.unravel_index(w.argmax(),w.shape)
   sample['models'][name]={'max_sigma_m':float(sigma.max()),'max_sigma_weight':float(pi[np.unravel_index(sigma.argmax(),sigma.shape)[:2]]),
     'official_horizons':[{ 'horizon_seconds':(h+1)*.1, 'pi':pi[h].tolist(), 'mu':mu[h].tolist(), 'sigma':sigma[h].tolist(), 'rho':p['rho'][index,h].tolist(), 'sqrt_determinant_m2':np.sqrt(np.maximum(np.linalg.det(cov[h]),0)).tolist(), 'raw_log_sigma_x':payload[name+'_raw'][slot,h,6:9].tolist(), 'raw_log_sigma_y':payload[name+'_raw'][slot,h,9:12].tolist()} for h in [7,15,23,31,39,47]],
     'dominant_covariance_component':{'horizon_seconds':(int(h)+1)*.1,'component':int(k),'pi':float(pi[h,k]),'sigma_x':float(sigma[h,k,0]),'sigma_y':float(sigma[h,k,1]),'rho':float(p['rho'][index,h,k]),'mean':mu[h,k].tolist(),'weighted_trace_m2':float(w[h,k])}}
  summary['samples'].append(sample)
 np.savez_compressed(DEST/'parameters.npz',**payload)
 (DEST/'diagnostics.json').write_text(json.dumps(summary,indent=2))
 # Show model-dependent validation tails with same data and no cherry-picked test inputs.
 fig,axes=plt.subplots(1,3,figsize=(13,4))
 for ax,(name,arch,_,_) in zip(axes,JOBS):
  area=np.load(AUDIT/arch/'validation/areas_m2.npy')[:,1].max(1)
  ax.scatter(f['max_speed_m_s'],area,s=3,alpha=.15,color=COLORS[[j[0] for j in JOBS].index(name)])
  ax.set(xlabel='Max observed speed (m/s)',ylabel='Max 68% area over six horizons (m²)',title=name,yscale='log')
 fig.tight_layout();fig.savefig(DEST/'speed_vs_area.png',dpi=170);plt.close(fig)
 # Representative selected union: early residual max, late residual max, attention late max.
 keys=['Residual:0.68:0.8','Residual:0.68:4.8','Attention:0.68:4.8']
 for key in keys:
  index=selectors[key][0];slot=int(np.flatnonzero(indices==index)[0]);t=np.arange(1,49)*.1
  fig,axes=plt.subplots(2,3,figsize=(14,8))
  axes[0,0].plot(X[index,:,0],X[index,:,1],color='gray',label='Observed');axes[0,0].plot(y[index,:,0],y[index,:,1],'k-',label='GT')
  axes[0,0].plot(payload['CV'][slot,:,0],payload['CV'][slot,:,1],'--',color='purple',label='CV')
  axes[0,0].axis('equal');axes[0,0].set(title='Observed / GT / CV',xlabel='x (m)',ylabel='y (m)');axes[0,0].legend(fontsize=8)
  axes[0,1].plot(np.arange(-31,1)*.1,np.linalg.norm(X[index,:,2:4],axis=-1),color='black');axes[0,1].set(title='Input observed speed',xlabel='Observed time (s)',ylabel='m/s')
  for j,(name,arch,_,_) in enumerate(JOBS):
   p=decoded[j];pi=p['pi'][index];sigma=p['sigma'][index];cov=p['covariance'][index]
   axes[0,2].plot(t,sigma.max((1,2)),label=name,color=COLORS[j])
   for k in range(3):
    axes[1,j].plot(t,sigma[:,k].max(-1),label=f'k{k} max sigma',lw=1.3)
   ax2=axes[1,j].twinx()
   for k in range(3):ax2.plot(t,pi[:,k],'--',alpha=.6,lw=.8)
   ax2.set(ylim=(0,1),ylabel='pi (dashed)');axes[1,j].set(title=name,xlabel='Horizon (s)',ylabel='sigma (m)',yscale='log');axes[1,j].legend(fontsize=7)
  axes[0,2].set(title='Maximum component sigma',xlabel='Horizon (s)',ylabel='m',yscale='log');axes[0,2].legend()
  fig.suptitle(f'Validation index {index}: {loader.sample_ids["eval"][index]}',fontsize=10);fig.tight_layout();fig.savefig(DEST/f'case_{index}.png',dpi=170);plt.close(fig)
 text='''# Truy nguyên các outlier validation

Chọn hợp các top2 diện tích confidence ở từng model/level/horizon từ audit validation. Không dùng test để chọn mẫu hay ngưỡng; không sửa model/dataset. Dữ liệu tham số đầy đủ: parameters.npz; nguồn ID và thống kê: diagnostics.json.

## Kiểm tra input

'''
 text+=f"Có {len(indices)} mẫu validation được chọn. Sai khác lớn nhất giữa velocity đã lưu và diff(position)/0.1 ở toàn validation: {summary['velocity_recompute_max_error']:.3e}. Điều này đối chiếu preprocessing hiện có, không chứng minh sensor/track gốc không có lỗi.\n\n"
 text+='| Input feature | Train median | Train P99 | Train max | Validation median | Validation P99 | Validation max |\n|---|---:|---:|---:|---:|---:|---:|\n'
 for feature in f:
  a=summary['input_statistics']['train'][feature]['quantiles'];b=summary['input_statistics']['validation'][feature]['quantiles'];text+=f"| {feature} | {a['50']:.3f} | {a['99']:.3f} | {a['100']:.3f} | {b['50']:.3f} | {b['99']:.3f} | {b['100']:.3f} |\n"
 text+='\n![Speed và diện tích](speed_vs_area.png)\n\nĐồ thị cho thấy mối liên hệ mô tả, không chứng minh speed gây ra covariance lớn.\n\n## Các trường hợp đại diện\n\n'
 for key in keys:
  index=selectors[key][0];s=summary['samples'][int(np.flatnonzero(indices==index)[0])]
  text+=f"### {key} — index {index}\n\nSource: `{s['source']}`.\n\n"
  for feature,v in s['inputs'].items():text+=f"- {feature}: {v['value']:.3f}; phân vị train {v['train_percentile']:.2f}%, validation {v['validation_percentile']:.2f}%.\n"
  text+='\n| Model | Sigma lớn nhất | Pi tại sigma lớn nhất | Horizon thành phần đóng góp covariance lớn nhất | Pi | Sigma x | Sigma y | Rho | Weighted trace |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
  for name,m in s['models'].items():
   c=m['dominant_covariance_component'];text+=f"| {name} | {m['max_sigma_m']:.3f} | {m['max_sigma_weight']:.3e} | {c['horizon_seconds']:.1f} | {c['pi']:.3e} | {c['sigma_x']:.3f} | {c['sigma_y']:.3f} | {c['rho']:.5f} | {c['weighted_trace_m2']:.3f} |\n"
  selected_horizon=float(key.rsplit(':',1)[1])
  text+=f'\nTham số ở horizon được chọn {selected_horizon:.1f} s (component không phải mode quỹ đạo chung):\n\n| Model | k | Pi | Sigma x (m) | Sigma y (m) | Raw log sigma x | Raw log sigma y | Rho | sqrt(det Sigma) (m²) |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
  for name,m in s['models'].items():
   c=next(v for v in m['official_horizons'] if abs(v['horizon_seconds']-selected_horizon)<1e-6)
   for k in range(3):text+=f"| {name} | {k} | {c['pi'][k]:.3e} | {c['sigma'][k][0]:.3f} | {c['sigma'][k][1]:.3f} | {c['raw_log_sigma_x'][k]:.3f} | {c['raw_log_sigma_y'][k]:.3f} | {c['rho'][k]:.5f} | {c['sqrt_determinant_m2'][k]:.3f} |\n"
  text+=f'\n![Tham số từng horizon](case_{index}.png)\n\n'
 text+='''## Kết luận cụ thể từ các mẫu đã đọc

1. Preprocessing velocity khớp với position differences; không tìm thấy lỗi tính vận tốc trong validation. Các giá trị speed/acceleration vẫn nằm trong miền giá trị từng xuất hiện ở train, dù một số thuộc đuôi rất hiếm. Không đủ căn cứ để xóa mẫu hay gán dữ liệu hỏng.
2. Residual index 5469 có thay đổi tốc độ mạnh; recent velocity thuộc khoảng top 0.21% train. Một Gaussian sigma_y≈204 m nhưng pi≈0.0035. Tại 4.8 s, Gaussian mang pi≈0.9947 có sigma≈(1.97,3.85) m. Do đó không được gán toàn bộ vùng 68% lớn cho component 204 m chỉ vì nó có max sigma/weighted trace lớn nhất; vùng density cao phải xét toàn mixture.
3. Attention index 13232 có max speed≈1.38 m/s, ở khoảng phân vị 62% train. Gaussian mang pi≈0.997 lại có sigma≈(57.27,29.44) m. Đây là ví dụ rõ về output covariance rất rộng trên một input không cực trị theo tốc độ. Không thể giải thích mọi lỗi bằng vận tốc cao.
4. Stable parameterization hiện tại ngăn covariance suy biến/overflow nhưng vẫn cho sigma rất lớn. NLL tốt trung bình không đảm bảo head được kiểm soát ở mọi lịch sử. Cơ chế trực tiếp đã thấy là raw log-sigma của head tạo component rộng; nguyên nhân học được sâu hơn cần ablation, chưa có chứng minh nhân quả.
5. Phép cộng prior CV chỉ dịch mean, không trực tiếp nhân hoặc cộng vào sigma. Nó có thể gián tiếp thay đổi quá trình học của mạng chung, nhưng kết quả hiện tại chưa chứng minh đó là nguyên nhân outlier.

Vì các component đổi vai trò theo timestep, không xem k cố định là một hành vi hay quỹ đạo liên tục. Hình sigma/pi từng k có thể nhảy mà vẫn là marginal GMM hợp lệ.

## Phân biệt cơ chế trong code và nguyên nhân học được

- Mean: residual chỉ cộng cùng một vector CV vào ba component tại mỗi horizon. Phép cộng này dịch chuyển toàn bộ GMM; không thay sigma/rho/pi trong forward. Đã kiểm tra equality phần raw từ sigma đến logits giữa residual_output và forward trên toàn bộ mẫu được chọn. Vùng confidence trên mặt phẳng không giới hạn bất biến theo phép dịch; trên grid hữu hạn, clipping có thể thay đổi diện tích đo.
- Sigma: stable dùng exp(raw_log_sigma) + 0.01, chỉ guard exponent ở sigma khoảng 1000 m. Guard này chống số học overflow, không phải giới hạn uncertainty hợp lý cho pedestrian. Raw log-sigma lớn có thể tạo ellipse rộng mà output vẫn finite và NLL tổng thể vẫn tốt.
- Pi: sigma lớn với pi rất nhỏ không nhất thiết làm confidence region 68% rộng. Phải xét cả pi và covariance có trọng số. Bảng hiển thị riêng max sigma và component đóng góp covariance lớn nhất để tránh nhầm hai khái niệm này.
- Rho: rho gần ±1 làm Gaussian hẹp theo một trục và kéo dài theo trục kia; determinant khác trace. Không thể dùng max sigma để thay sharpness GMM.
- Marginal NLL tối ưu density của GT, không trực tiếp phạt covariance outlier hoặc scalar sharpness cực đại. Đây là đặc điểm objective, chưa chứng minh một cơ chế nhân quả cụ thể trên các mẫu này.

Chưa xác định được nguyên nhân gốc trong dữ liệu cảm biến gốc hoặc vì sao quá trình tối ưu tạo head như vậy. Không tự loại mẫu hay kết luận dữ liệu hỏng chỉ vì nó ở percentile cao. Cần xem lịch sử/track gốc nếu muốn xác nhận lỗi tracking.

## Hướng tiếp theo

Dùng các mẫu validation đã lưu để thiết kế một ablation kiểm soát covariance: giới hạn mềm log-sigma hoặc penalty cho covariance quá lớn, chọn tham số bằng validation và giữ K/LSTM/CV/NLL nền. Kiểm tra NLL, reliability và sharpness cùng nhau vì siết covariance có thể làm undercoverage. Đây là đề xuất thí nghiệm; chưa áp dụng hay train. Giữ các run hiện tại làm đối chứng.

Tái tạo: `.venv/bin/python residual_mdn/analysis/investigate_validation_outliers.py`.
'''
 (DEST/'REPORT.md').write_text(text)
 print('Selected',len(indices),'cases; report',DEST/'REPORT.md')
if __name__=='__main__':main()
