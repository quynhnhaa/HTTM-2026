"""Summarize completed per-sample audits without modifying official scores."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_confidence_areas import DEST,JOBS
from compare_models import OUT,COLORS

def main():
 text='''# Audit diện tích confidence theo từng mẫu

Audit best checkpoint cả validation/test, seed 2024, sáu horizon, grid [-18,18]² với resolution 0.1, MC 1000. Lưu areas_m2.npy, touches_boundary.npy, identifiers.npz và summary.json trong results/comparisons/confidence_area_audit/full. Replay thứ tự sampling ADE/reliability/grid của evaluator; phép rank bằng searchsorted tương đương strict > và đã kiểm tra ties/đầu ra checkpoint khi smoke. Không sửa metric chính thức.

Boundary-touch nghĩa là ít nhất một điểm vùng confidence nằm trên biên grid, không phải tỷ lệ probability mass ngoài grid. Confidence area chạm trần 1296 m² bị giới hạn bởi miền đo. Scalar local trung bình 101 phân vị và có thêm chuẩn hóa horizon như code hiện tại; Q100 contribution dưới đây là thành phần chính xác của scalar audit, không phải độ cải thiện model.

| Model | Split | Level | Scalar audit | Scalar test trước | Delta | Q100 contribution | Q100 / scalar |
|---|---|---:|---:|---:|---:|---:|---:|
'''
 data={}
 for job in JOBS:
  name,arch,_,_=job;data[name]={}
  for split in ['validation','test']:
   path=DEST/'full'/arch/split/'summary.json';d=json.loads(path.read_text());assert d['metadata']['status']=='completed';data[name][split]=d
   for level,m in d['confidence'].items():
    previous=m.get('previous_official_scalar'); delta=m.get('delta_vs_previous')
    text+=f"| {name} | {split} | {level} | {m['local_scalar']:.6f} | {previous if previous is not None else '—'} | {delta if delta is not None else '—'} | {m['q100_contribution']:.6f} | {m['q100_fraction_of_scalar']*100:.2f}% |\n"
 text+='\n## Phân rã chênh lệch test so với baseline\n\n| Model | Level | Delta scalar | Delta phần Q100 | Delta phần Q0–Q99 |\n|---|---|---:|---:|---:|\n'
 for name in ['Attention','Residual']:
  for level in ['0.68','0.95']:
   m=data[name]['test']['confidence'][level];b=data['Baseline']['test']['confidence'][level]
   delta=m['local_scalar']-b['local_scalar'];tail=m['q100_contribution']-b['q100_contribution']
   text+=f"| {name} | {level} | {delta:+.6f} | {tail:+.6f} | {delta-tail:+.6f} |\n"
 text+='\nĐây là phân rã đại số của scalar: Delta = Delta Q100 + Delta Q0–Q99. Không phải phép thử thay model, không loại mẫu, không đổi metric. Nếu Delta Q100 lớn hơn Delta scalar, phần Q0–Q99 đang bù lại theo chiều ngược. Không suy ra mean area hoặc mọi mẫu đều cải thiện chỉ từ phần Q0–Q99.\n\n'
 text+='\n## Kết luận từ audit\n\n'
 text+='- Scalar test audit khớp score đã lưu trong phạm vi sai số làm tròn (xem delta bảng trên). Vì thế phân rã Q100 giải thích trực tiếp score test hiện có.\n'
 text+='- Residual S68 tăng 2.223124 so với baseline; riêng Q100 tăng 2.247732, phần Q0–Q99 giảm 0.024608. Với S95, Q100 tăng 3.358396 trong khi Q0–Q99 giảm 0.615624. Phần tăng scalar tập trung ở cực đại, không phải bằng chứng mọi vùng dự báo đều rộng hơn.\n'
 text+='- Tại 4.8 s, residual mean Area68 là 6.218 m² và median là 5.529 m², thấp hơn baseline 6.637 / 5.877 m². Mean Area95 residual cũng thấp hơn ở mốc này (41.977 so với 49.612 m²). Ngược lại, tại 0.8 s mean Area68 residual cao hơn baseline (0.036 so với 0.009 m²). Kết luận phụ thuộc horizon và phép tổng hợp.\n'
 text+='- Các vùng chạm biên hiếm nhưng có thật; không dùng grid hiện tại để suy ra diện tích vô hạn hay xác suất ngoài grid. Reliability residual vẫn kém hơn ở Rmin, nên area nhỏ hơn ở nhiều mẫu không tự chứng minh uncertainty tốt hơn toàn diện.\n'
 text+='- Hướng tiếp theo có cơ sở hơn là kiểm tra input/raw log-sigma của outlier validation và thiết kế kiểm soát covariance trên train/validation. Cần đánh giá khả năng làm giảm coverage khi giới hạn sigma. Chưa thay parameterization, loss hoặc chạy training mới.\n'
 text+='\n## Diện tích và chạm biên theo horizon\n\n'  
 fig,axes=plt.subplots(2,4,figsize=(17,8))
 for row,level in enumerate(['0.68','0.95']):
  for j,(name,splits) in enumerate(data.items()):
   d=splits['test'];m=d['confidence'][level];t=d['metadata']['times_seconds']
   for ax,values in zip(axes[row],[m['mean_area_by_horizon'],m['quantiles_area_by_horizon']['50'],m['quantiles_area_by_horizon']['100'],m['boundary_touch_fraction_by_horizon']]):
    ax.plot(t,values,color=COLORS[j],label=name)
  for ax,title in zip(axes[row],['Mean area (m²)','Median area (m²)','Max area (m²)','Boundary-touch fraction']):ax.set(title=f'{level}: {title}',xlabel='Horizon (s)');ax.grid(alpha=.2)
 axes[0,0].legend();fig.suptitle('Full test confidence-area audit');fig.tight_layout();fig.savefig(OUT/'area_audit.png',dpi=170);plt.close(fig)
 text+='![Audit toàn test](area_audit.png)\n\n'
 for name,splits in data.items():
  for split,d in splits.items():
   text+=f'### {name} / {split}\n\n| Level | Horizon | Mean m² | Median m² | P95 m² | P99 m² | Max m² | Touch boundary % | Full-grid % |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
   for level,m in d['confidence'].items():
    for i,t in enumerate(d['metadata']['times_seconds']):
     q=m['quantiles_area_by_horizon'];text+=f"| {level} | {t:.1f} | {m['mean_area_by_horizon'][i]:.3f} | {q['50'][i]:.3f} | {q['95'][i]:.3f} | {q['99'][i]:.3f} | {q['100'][i]:.3f} | {m['boundary_touch_fraction_by_horizon'][i]*100:.4f} | {m['full_grid_fraction_by_horizon'][i]*100:.4f} |\n"
   ids=np.load(DEST/'full'/d['metadata']['label'].lower().replace('baseline','stable_mdn').replace('attention','stable_attention_mdn').replace('residual','residual_mdn')/split/'identifiers.npz')
   text+='\nOutlier 68% tại horizon đầu (xếp theo diện tích; cửa sổ gần nhau không độc lập):\n\n'
   for index in d['confidence']['0.68']['top_indices_per_horizon'][0][:5]:text+=f"- Index {index}: `{ids['sample_id'][index]}`, source `{ids['source'][index]}`.\n"
 text+='''
## Cách diễn giải

Đối chiếu scalar audit với test trước khi dùng đóng góp phân vị để giải thích kết quả cũ. Nếu delta đáng kể, kiểm tra phiên bản runtime/device và thứ tự RNG trước khi coi đây là tái hiện đúng. Q100 cho biết phần đóng góp của cực đại trong phép tổng hợp; không tự loại Q100 khỏi metric hoặc thay score đã công bố. Mean/median/P95/P99 giúp phân biệt cải thiện ở phần lớn mẫu và lỗi hiếm. Test chỉ dùng phân tích; mọi lựa chọn covariance/regularization tiếp theo phải chốt bằng train/validation, với một protocol mới được ghi rõ.
'''
 (OUT/'AREA_AUDIT.md').write_text(text);print(OUT/'AREA_AUDIT.md')
if __name__=='__main__':main()
