"""Explain legacy S68 using exact replay areas and independently drawn old figures."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_legacy_confidence import ROOT,DEST,JOBS

OUT=ROOT/'attention_mdn/reports'

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 reports=[];areas=[]
 for name,arch,_,_ in JOBS:
  folder=DEST/'full'/arch/'test';d=json.loads((folder/'summary.json').read_text());assert d['metadata']['status']=='completed'
  reports.append(d);a=np.load(folder/'areas_m2.npy');assert a.shape==(56694,2,6) and np.isfinite(a).all();areas.append(a)
 for level in ['0.68','0.95']:
  for d in reports:assert abs(d['confidence'][level]['delta_vs_previous'])<1e-10
 ids=[np.load(DEST/'full'/j[1]/'test/identifiers.npz')['sample_id'] for j in JOBS];np.testing.assert_array_equal(*ids)
 delta=areas[1][:,1]-areas[0][:,1];fraction=float((delta>0).mean());t=reports[0]['metadata']['times_seconds']
 q=np.array([np.percentile(a[:,1],np.arange(101),axis=0) for a in areas]);contrib=(q/np.asarray(t)).sum(-1)/4.8/101
 total_delta=contrib[1].sum()-contrib[0].sum();maxdelta=contrib[1,-1]-contrib[0,-1]
 fig,axes=plt.subplots(1,4,figsize=(17,4))
 for i,name in enumerate(['Base','Attention']):
  axes[0].plot(t,areas[i][:,1].mean(0),label=name);axes[1].plot(t,q[i,50],label=name);axes[2].plot(np.arange(100),q[i,:100,-1],label=name)
 axes[0].set(title='Mean A68',ylabel='m²',xlabel='Horizon (s)');axes[1].set(title='Median A68',ylabel='m²',xlabel='Horizon (s)');axes[2].set(title='A68 percentiles at 4.8s (Q100 excluded)',ylabel='m²',xlabel='Percentile')
 axes[3].bar(['Base','Attention'],contrib[:,:100].sum(1),label='Q0–Q99');axes[3].bar(['Base','Attention'],contrib[:,100],bottom=contrib[:,:100].sum(1),label='Q100')
 axes[3].set(title='Contributions to local S68',ylabel='Scalar contribution');axes[3].legend(fontsize=8);axes[0].legend()
 for ax in axes:ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(OUT/'legacy_s68_exact.png',dpi=170);plt.close(fig)
 legacy=json.loads((ROOT/'results/analysis/legacy_s68/summary.json').read_text())
 text='''# Vì sao attention gốc cải thiện nhiều metric nhưng S68 kém hơn?

## Kết luận

Hai model trong phân tích này là `base_mdn` best epoch 1961 và `attention_mdn` best epoch 1740, covariance legacy (`sigma=exp(raw)`, `rho=tanh(raw)`). Không sử dụng checkpoint stable. Audit toàn bộ 56.694 test × sáu horizon, replay thứ tự sampling của evaluator gốc. Scalar S68/S95 tính lại khớp kết quả đã công bố trong sai số làm tròn. Không sửa model/loss/evaluator hoặc train thêm.

Attention không làm vùng 68% rộng hơn đồng loạt. Model có vùng nhỏ hơn ở nhiều mẫu, nhưng phần đuôi của phân bố diện tích lớn hơn. Một số Gaussian rộng có trọng số chi phối tạo vùng confidence rất lớn; phép tổng hợp S68 gồm phân vị cực đại nên khuếch đại tác động của các trường hợp này. Đây là cơ chế trực tiếp có bằng chứng; chưa chứng minh attention layer là nguyên nhân nhân quả của quá trình học covariance rộng.

## Phân rã chính xác điểm S68 test

| Phần | Base | Attention | Attention − base |
|---|---:|---:|---:|
'''
 for label,values in [('Q0–Q99',contrib[:,:100].sum(1)),('Q100 / maximum',contrib[:,100]),('S68',contrib.sum(1))]:text+=f'| {label} | {values[0]:.6f} | {values[1]:.6f} | {values[1]-values[0]:+.6f} |\n'
 text+=f'\n**{maxdelta/total_delta*100:.2f}% mức tăng S68 đến từ phần chênh lệch Q100.** Các phân vị còn lại đóng góp {total_delta-maxdelta:.6f}; không quy toàn bộ cho cực đại. Attention rộng hơn ở **{fraction*100:.2f}%** cặp mẫu–horizon trong audit replay.\n\n'
 text+='''Scalar local = (1/4.8) × Σ_h [mean của Q0,…,Q100(Area68 tại horizon h) / t_h]. Mỗi Q100 có trọng số 1/101, dù chỉ là cực đại của tập 56.694 mẫu. Đường sharpness evaluator vẽ median, nhưng scalar dùng trung bình 101 phân vị. Đây không phải mean area thông thường hay median. Phép tổng hợp và nhãn đơn vị được giữ theo repository; không gán sự chuẩn hóa đó cho paper khi chưa đối chiếu.

![Phân rã replay](legacy_s68_exact.png)

## Diện tích theo horizon

| Horizon | Base mean A68 | Attention mean A68 | Base median A68 | Attention median A68 | Base P99 | Attention P99 |
|---|---:|---:|---:|---:|---:|---:|
'''
 for h,seconds in enumerate(t):text+=f'| {seconds:.1f} | {areas[0][:,1,h].mean():.3f} | {areas[1][:,1,h].mean():.3f} | {q[0,50,h]:.3f} | {q[1,50,h]:.3f} | {q[0,99,h]:.3f} | {q[1,99,h]:.3f} |\n'
 text+='\nDiện tích là m² trong grid, không phải scalar S68. Mean/median/P99 trả lời các câu hỏi khác nhau; median thấp hơn vẫn có thể đi cùng mean và scalar cao hơn.\n\n## Kiểm tra component ở trường hợp rộng lên mạnh\n\n'
 index,h=np.unravel_index(delta.argmax(),delta.shape);seconds=t[h]
 # Raw cache is from the same immutable best checkpoints but different inference batch/MC draws.
 cache=np.load(ROOT/'results/analysis/legacy_s68/per_sample.npz');raw=cache['raw_outputs'][:,index,int(round(seconds/.1))-1]
 text+=f'Trong replay, cặp tăng A68 lớn nhất là index **{index}**, ID `{ids[0][index]}`, horizon **{seconds:.1f}s**: baseline **{areas[0][index,1,h]:.3f} m²**, attention **{areas[1][index,1,h]:.3f} m²**.\n\n'
 text+='| Model | k | Pi | mu x | mu y | sigma x | sigma y | rho |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
 for m,name in enumerate(['Base','Attention']):
  logits=raw[m,15:18];pi=np.exp(logits-logits.max());pi/=pi.sum()
  for k in range(3):text+=f'| {name} | {k} | {pi[k]:.6f} | {raw[m,k]:.3f} | {raw[m,k+3]:.3f} | {np.exp(raw[m,k+6]):.3f} | {np.exp(raw[m,k+9]):.3f} | {np.tanh(raw[m,k+12]):.5f} |\n'
 text+='''
Tham số raw lấy từ cache inference cùng checkpoint; batch khác có thể tạo sai số làm tròn float32 nhỏ. Mean/pi/sigma/rho phải đọc cùng nhau: sigma lớn với pi gần 0 chưa đủ để giải thích vùng 68% lớn. Gaussian có pi lớn và covariance rộng là bằng chứng trực tiếp hơn. Mean phân tán cũng có thể làm confidence region rộng, nhưng không được kết luận đó là nguyên nhân duy nhất.

Hình dưới từ audit MC độc lập có trước, không dùng để thay thế scalar replay. Nó minh họa vùng GMM và tham số của mẫu 4881; số diện tích trên hình có thể lệch nhẹ do lượt MC khác.

![Mẫu 4881](../../results/analysis/legacy_s68/largest_increase_1_sample4881_h48.png)

![Attention gọn hơn ở mẫu khác](../../results/analysis/legacy_s68/largest_decrease_1_sample11983_h48.png)

## Vì sao không mâu thuẫn với các metric tốt hơn?

- **NLL** đánh giá density tại GT, lấy trung bình trên mẫu và timestep. Model có thể tăng density ở nhiều mẫu để cải thiện NLL tổng thể, đồng thời thất bại uncertainty ở một số mẫu hiếm. NLL không tối ưu trực tiếp cực đại confidence area.
- **minADE20/minFDE20** đánh giá sai số nhỏ nhất trong các sample theo cách sắp xếp marginal của repo. Chúng không đo diện tích confidence region. Có sample gần GT và vùng xác suất rộng là hai điều có thể cùng xảy ra.
- **Reliability** đánh giá độ khớp giữa mức confidence và coverage thực nghiệm. Sharpness đánh giá diện tích vùng đó. Reliability tốt hơn không tự suy ra vùng nhỏ hơn. Ravg/Rmin ở bảng cũ dùng cách bin legacy, không được đọc như số từ evaluator stable đã sửa ECDF.
- **S68 và S95** là hai mức mass khác nhau. Vùng 68% nằm trong vùng 95% của từng distribution, nhưng thứ hạng *giữa hai model* ở hai mức không bắt buộc giống nhau. Việc tái phân bố mass/shape giữa component có thể làm S68 tăng và S95 giảm. Với các run này S95 giảm rất nhẹ, cần tránh diễn giải quá mạnh.

Vì vậy không có quy tắc yêu cầu mọi metric cùng cải thiện khi thêm attention. Trên kết quả hiện tại, cải thiện displacement/density đi cùng hạn chế ở phần đuôi uncertainty.

## Giới hạn và hướng xử lý

- Đây là một seed và best checkpoint mỗi model. Attention run dừng ở epoch1764 do covariance không hợp lệ; chưa có training ổn định tới2500 epoch.
- Grid [-18,18]² giới hạn diện tích ở1296 m². Nếu confidence region chạm biên, diện tích ngoài miền chưa được đo. Audit lưu touches_boundary.npy; đây là cờ hình học, không phải xác suất ngoài grid.
- Phân rã Q100 giải thích scalar tăng, không phải lý do bỏ metric, xóa mẫu hay loại cực trị khỏi báo cáo.
- Nếu tiếp tục: audit các trường hợp covariance cực trị trên validation, rồi chọn cách kiểm soát covariance/ổn định training bằng train/validation. Không chọn ngưỡng theo các mẫu test ở trên. Mọi variant cần đánh giá NLL, displacement, reliability và sharpness cùng nhau, có nhiều seed nếu muốn kết luận ổn định.

## Nguồn và tái tạo

```bash
.venv/bin/python analysis/audit_legacy_confidence.py --limit 8
.venv/bin/python analysis/audit_legacy_confidence.py
.venv/bin/python analysis/summarize_legacy_s68_exact.py
```

Audit replay ở `results/analysis/legacy_s68_exact/full/{base_mdn,attention_mdn}/test/`: areas_m2.npy, touches_boundary.npy, identifiers.npz, summary.json và progress.json. Metadata có hash checkpoint, seed, phiên bản Torch/device và sampling protocol. Audit MC độc lập trước đó ở `results/analysis/legacy_s68/`. Các score test gốc lấy từ `results/comparisons/imptc_m1_vs_m3/comparison.json` và attention run testing/evaluation.json, được giữ nguyên.
'''
 (OUT/'LEGACY_S68_ANALYSIS.md').write_text(text)
 (DEST/'comparison.json').write_text(json.dumps({'status':'completed','split':'test','sample_count':56694,'fraction_pairs_attention_wider68':fraction,
    'delta_s68':float(total_delta),'delta_q100_contribution':float(maxdelta),'fraction_delta_from_q100':float(maxdelta/total_delta),
    'largest_increase_index':int(index),'horizon_seconds':seconds,'sources':[str(DEST/'full'/j[1]/'test/summary.json') for j in JOBS]},indent=2))
 print(OUT/'LEGACY_S68_ANALYSIS.md')
if __name__=='__main__':main()
