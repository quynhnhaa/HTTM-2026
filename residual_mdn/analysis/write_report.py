"""Build Vietnamese analysis report from persisted diagnostics."""
import json
from pathlib import Path
from compare_models import ROOT,OUT,JOBS

def main():
 d=json.loads((OUT/'diagnostics.json').read_text()); full=json.loads((OUT/'full_split_distribution.json').read_text())
 comparison=json.loads((ROOT/'results/comparisons/stable_three_models/comparison.json').read_text())
 text='''# Phân tích baseline stable, attention và residual MDN

## Kết luận có thể sử dụng trong đồ án

Residual CV + LSTM–MDN cải thiện NLL và sai số vị trí trung bình so với baseline stable trên test ở seed 2024. Tuy nhiên, Rmin giảm và điểm sharpness tăng. Attention đạt NLL/ADE/FDE tốt nhất trong ba model nhưng cũng đánh đổi reliability/sharpness. Chưa có model tốt nhất trên mọi tiêu chí; chưa có kiểm định nhiều seed.

Phân tích đầu ra cho thấy residual không tăng covariance đồng đều trên mọi mẫu. Một số outlier test có vùng confidence rất rộng, đặc biệt ở horizon ngắn. Cách tổng hợp sharpness hiện tại có sử dụng phân vị cực đại, nên nhạy với các trường hợp này. Đây là bằng chứng về một nguyên nhân có thể đóng góp; chưa phân rã đầy đủ mức chênh lệch S68/S95 chính thức.

## Kết quả test chính thức đã lưu

56.694 mẫu test; best chọn bằng full validation NLL; cùng seed/evaluator/MC/grid. Đây là các số từ evaluation.json, không thay bằng metric chẩn đoán mới.

| Chỉ số | Baseline | Attention | Residual |
|---|---:|---:|---:|
'''
 for row in comparison['results']:
  if row['metric'] in ['test_nll','minade20_m','minfde20_m','ravg_percent','rmin_percent','s68_m2_per_s','s95_m2_per_s','asaee_m_per_s']:
   text+=f"| {row['metric']} | {row['baseline']:.6f} | {row['attention']:.6f} | {row['residual']:.6f} |\n"
 text+='''
## Lịch sử học và metric validation

![Training và validation NLL](learning.png)

![Metric validation tại checkpoint định kỳ](validation_metrics.png)

Sau khoảng 2.000–2.500 epoch, đường validation NLL và displacement metric cải thiện chậm. Việc thêm 500 epoch không tạo thay đổi lớn. Các đường metric là checkpoint định kỳ, không phải metric test của best checkpoint. Sharpness residual trên validation định kỳ thấp hơn baseline; thứ hạng này không được giữ trên test. Vì vậy, chỉ xem validation trung bình hoặc một vài ví dụ sẽ bỏ sót trường hợp bất thường.

## Kiểm tra 8 mẫu validation cố định

Đã kiểm tra sample_id, X và y bằng array equality giữa cả ba model; đường CV cũng được đối chiếu với artifact residual. Best epoch lần lượt 2792 / 2977 / 2965. Không chọn lại các mẫu theo kết quả thuận lợi.

Mixture mean dùng cho hình là Σ pi_k mu_k. Sai số của đường này là chẩn đoán bổ sung, không phải minADE20/minFDE20 của repository. Đường nối các mean chỉ để đọc vị trí theo thời gian; không chứng minh một phân phối joint trên toàn quỹ đạo.

Phân rã covariance marginal:

- Within = Σ pi_k trace(Sigma_k).
- Between = Σ pi_k ||mu_k − mu_bar||².
- Total = Within + Between, tức trace của covariance GMM.

| Model | Within trung bình (m²) | Between trung bình (m²) | Sai số mixture mean trung bình (m) |
|---|---:|---:|---:|
'''
 for name,m in d['models'].items():text+=f"| {name} | {m['within_trace_mean_m2']:.4f} | {m['between_trace_mean_m2']:.4f} | {m['mean_position_error_diagnostic_m']:.4f} |\n"
 text+='''
![Covariance và sai số thực tế](uncertainty_decomposition.png)

Within của residual gần baseline trên 8 mẫu này. Attention có Gaussian cực rộng nhưng trọng số rất nhỏ ở một số mẫu: trung bình covariance không có trọng số vì thế khác hẳn covariance GMM có trọng số. Không thể kết luận chất lượng uncertainty từ sigma riêng lẻ.

![Diện tích confidence trên mẫu cố định](fixed_confidence_areas.png)

Diện tích GMM 68%/95% được chẩn đoán với grid [-18,18]², 361×361 điểm, 1.000 MC samples. Đây là 8 mẫu validation và RNG NumPy riêng; không phải chạy lại toàn bộ metric test hay tái hiện từng bit RNG Torch. Đường solid là mean, dashed là median. Không dùng các số này để thay thế bảng test.

## Mẫu tốt, khó và thất bại

Tất cả 8 mẫu đều có hình. Nhãn movement_class lấy nguyên từ manifest; đây không phải đánh giá tự động mức độ dễ/khó.

| Slot | Nhãn dữ liệu | Baseline mean error | Attention mean error | Residual mean error | Hình |
|---|---|---:|---:|---:|---|
'''
 for i,s in enumerate(d['samples']):
  vals=[d['models'][name]['per_sample_mean_position_error_m'][i] for name in ['Baseline','Attention','Residual']]
  text+=f"| {i} | {s['movement_class']} | {vals[0]:.3f} | {vals[1]:.3f} | {vals[2]:.3f} | [Xem](sample_{i:02d}.png) |\n"
 text+='''
- Slot 5–6 (standing): cả ba có sai số mean nhỏ, residual cải thiện nhẹ so với baseline.
- Slot 1 (strong_right): residual giảm mean error so với baseline; attention tốt ở các mốc đầu nhưng tăng lỗi gần cuối horizon.
- Slot 2 (light_right): residual có mean error cao hơn baseline, là ví dụ thất bại của cải tiến.
- Slot 3 (straight): attention tốt hơn rõ trên mẫu này; residual vẫn có lỗi đáng kể. Prior CV không đảm bảo dự báo tốt trên mọi mẫu đi thẳng.

![Trường hợp rẽ](sample_01.png)

![Trường hợp residual kém hơn](sample_02.png)

Các ellipse biểu diễn 68% của từng Gaussian bivariate, không phải vùng 68% của toàn GMM. Kích thước marker và opacity biểu diễn trọng số. Trục chung được giới hạn quanh observed/GT/CV/mixture mean để đọc đường dự báo; ellipse lớn bị cắt ở mép hình. Tham số đầy đủ vẫn có trong NPZ nguồn.

![Cùng mẫu qua checkpoint](checkpoint_evolution.png)

## Audit covariance toàn bộ validation và test

Chạy inference best checkpoint trên toàn bộ 19.148 validation và 56.694 test, không optimizer/training. Các số sau lấy trung bình trace GMM qua sáu horizon chính thức trên từng mẫu, rồi tổng hợp giữa các mẫu. Đây là diagnostic về covariance, không phải sharpness confidence area.

| Model | Split | Mean trace (m²) | Median trace | P99 trace | Max trace |
|---|---|---:|---:|---:|---:|
'''
 for name,splits in full.items():
  for split,m in splits.items():
   s=m['diagnostics']['total_trace_m2'];q=s['quantiles'];text+=f"| {name} | {split} | {s['mean']:.3f} | {q['50']:.3f} | {q['99']:.3f} | {q['100']:.3f} |\n"
 text+='''
Residual test có median trace thấp hơn baseline, nhưng max trace lớn hơn nhiều. Attention có median nhỏ nhưng mean bị kéo lên bởi outlier cực lớn. Đây là lý do cần xem cả phân vị và các trường hợp cực trị.

Đối chiếu grid/MC trên ba mẫu có total trace lớn nhất mỗi split:

| Model / split | Index | Source | Area68 @0.8s | Area95 @0.8s | Area68 @4.8s | Area95 @4.8s |
|---|---:|---|---:|---:|---:|---:|
'''
 for name,splits in full.items():
  for split,m in splits.items():
   for a in m['top3_grid_audit']:
    first,last=a['areas_m2_horizon_by_confidence'][0],a['areas_m2_horizon_by_confidence'][-1]
    text+=f"| {name}/{split} | {a['index']} | {a['source']} | {first[0]:.2f} | {first[1]:.2f} | {last[0]:.2f} | {last[1]:.2f} |\n"
 text+='''
Đáng chú ý: residual test index 33794 có Area68 tại 0.8 s khoảng 530 m²; baseline cùng index chỉ khoảng 0.109 m² trong phép chẩn đoán này. Các index 33793–33795 thuộc các cửa sổ gần nhau của cùng source, nên không xem chúng là bằng chứng độc lập về tần suất lỗi. Một số confidence area chạm trần 1.296 m² của grid: diện tích thực ngoài grid chưa được đo; metric bị giới hạn bởi miền tích phân.

## Công thức sharpness thực tế và giới hạn

Nguồn: stable_mdn/eval.py (build_confidence_set_mdn, estimate_sharpness) và stable_mdn/vis.py (plot_sharpness_over_time).

1. Với mỗi điểm grid z, confidence = tỷ lệ sample GMM có log density lớn hơn log density tại z.
2. Area_kappa(h) = tỷ lệ điểm grid có confidence ≤ kappa × 36 × 36.
3. Giữa các mẫu, tính các phân vị q=0,1,…,100 tại từng horizon.
4. Scalar hiện tại = (1/4.8) × Σ_h [mean_q Q_q(Area_kappa(h)) / t_h], với t_h=0.8,1.6,…,4.8 s.

Đường vẽ trong evaluator là Q50; scalar dùng mean của 101 phân vị, gồm Q100 (max), không phải median. Với N rất lớn, riêng max vẫn có trọng số 1/101; một mẫu cực trị có thể ảnh hưởng đáng kể dù tần suất nhỏ. Phép chia cho t_h cũng tăng tác động của diện tích bất thường ở horizon sớm. Vì vậy, scalar không đồng nhất với trung bình diện tích trên toàn bộ mẫu.

Bảng official giữ nguyên nhãn đơn vị và phép tổng hợp local hiện có. Phép chuẩn hóa thêm 1/(num_steps*dt) cần được rà soát khi giải thích thứ nguyên học thuật; tài liệu này không tự sửa metric hoặc gán cách tổng hợp đó cho paper. Trường sharpness_formula_version='corrected' nói về phần diện tích grid đã sửa, không chứng minh toàn bộ scalar đã được chuẩn hóa theo paper.

Chưa lưu toàn bộ per-sample confidence area của lần test trước, nên chưa thể phân rã chính xác đóng góp Q100 vào chênh lệch scalar đã công bố. Audit hiện tại chỉ xác nhận covariance và confidence-area outlier tồn tại. Không coi chúng là nguyên nhân duy nhất; reliability, thay đổi hình dạng mixture, MC noise và clipping grid còn ảnh hưởng.

## Bước cải tiến đề xuất sau phân tích

1. Nếu cần kết luận uncertainty chặt chẽ: chạy một audit evaluator riêng, lưu per-sample area theo horizon cùng ID, MC/grid/provenance; giữ evaluator chính thức hiện tại để so sánh. Báo riêng mean/median/P95/P99/max và tỷ lệ confidence region chạm biên. Chưa chạy audit full này trong bước phân tích hiện tại.
2. Nếu cần thử model mới: ưu tiên nghiên cứu covariance có giới hạn vật lý hoặc regularization chống outlier. Đây là giả thuyết xuất phát từ validation/outlier audit; giới hạn sigma, trọng số regularization và lựa chọn model phải được chốt bằng train/validation. Không tự chọn ngưỡng theo các index test ở trên, không sửa checkpoint/model đang có.
3. Nếu mục tiêu hoàn thiện đồ án: kết quả ba model hiện đủ để trình bày ablation về trade-off. Dùng baseline làm mốc, attention cho cải thiện displacement rõ hơn, residual cho thay đổi nhỏ với cùng số tham số. Không cần tiếp tục kéo dài cùng run chỉ vì chưa có một model thắng mọi metric.
4. Khi tuyên bố cải thiện ổn định, cần nhiều seed và protocol cố định. Mọi training mới cần được duyệt riêng.

## Tái tạo và nguồn

```bash
.venv/bin/python residual_mdn/analysis/compare_models.py
.venv/bin/python residual_mdn/analysis/scan_distribution.py
.venv/bin/python residual_mdn/analysis/write_report.py
```

Nguồn kết quả test: results/comparisons/stable_three_models/comparison.json. Nguồn hình: fixed_samples/inputs.npz và predictions/best.npz của từng run continuation500; lịch sử lấy cả run2500 và continuation500. Nguồn định danh nằm trong diagnostics.json; full_split_distribution.json chứa thống kê và index/source outlier. fixed_diagnostics.npz lưu diện tích chẩn đoán 8 mẫu. Model/checkpoint/config/loss/official metrics không được thay đổi bởi các script này.
'''
 if (OUT/'AREA_AUDIT.md').exists():
  text=text.replace('## Kết luận có thể sử dụng trong đồ án','Audit toàn bộ confidence area đã hoàn tất sau báo cáo ban đầu: [AREA_AUDIT.md](AREA_AUDIT.md). Audit tái hiện scalar test và xác nhận Q100 giải thích toàn bộ mức tăng sharpness residual so với baseline; phần Q0–Q99 giảm nhẹ. Các đoạn dưới mô tả phân tích ban đầu và giới hạn lúc chưa có per-sample area.\n\n## Kết luận có thể sử dụng trong đồ án',1)
 (OUT/'ANALYSIS.md').write_text(text)
 print(OUT/'ANALYSIS.md')
if __name__=='__main__':main()
