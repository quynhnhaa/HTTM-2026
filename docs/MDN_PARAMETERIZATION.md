# Tham số hóa MDN dùng chung: legacy và paper

## Thay đổi

Bộ giải mã chung nằm ở `base_mdn/utils/mdn_distribution.py`. Baseline, attention và mode-consistent truyền chính sách vào bộ giải mã này khi tính loss, đánh giá, sampling và lưu các mẫu cố định. Không đổi kiến trúc, K, optimizer, preprocessing hay định nghĩa metrics.

Cấu hình cũ không có `model_params.mdn_parameterization` được hiểu là `legacy`:

$$
\sigma=\exp(o_\sigma),\qquad \rho=\tanh(o_\rho).
$$

Các cấu hình mới dùng `paper`, theo dạng công thức trong paper trang 6:

$$
\sigma=\exp(o_\sigma)+1+\varepsilon_\sigma,\qquad
\rho=\tanh(o_\rho)\varepsilon_\rho.
$$

```json
"mdn_parameterization": {
  "mode": "paper",
  "epsilon_sigma": 0.000001,
  "epsilon_rho": 0.999
}
```

**Hai epsilon là lựa chọn triển khai, chưa được xác nhận là giá trị chính thức của paper.** Phần cộng 1 áp dụng vào độ lệch chuẩn, không phải cộng ma trận đơn vị vào covariance. Với tọa độ hiện tại, sigma có giới hạn dưới khoảng 1 mét; giới hạn này có thể làm phân phối rộng hơn và ảnh hưởng NLL/sharpness. Đây là thử nghiệm theo dạng tham số hóa paper, chưa phải chứng minh tái hiện chính xác paper hoặc bảo đảm metrics tốt hơn.

Covariance vẫn được dựng bằng sigma_x², sigma_y² và rho·sigma_x·sigma_y. Giới hạn tránh sigma gần 0 và rho sát ±1; nó không bảo đảm mọi run sẽ ổn định nếu raw output bị NaN/Inf hoặc exp bị tràn số. Không thêm clipping, jitter hay thay loss ngầm.

## Metadata và checkpoint cũ

- Run manifest, resolved config và checkpoint ghi lại chính sách và hai epsilon.
- Evaluator đọc chính sách trong checkpoint; checkpoint cũ thiếu metadata dùng legacy.
- Resume từ checkpoint có chính sách khác bị từ chối. Bắt đầu các run mới từ đầu.
- Visualization dùng các tham số đã giải mã và lưu trong từng run. Không diễn giải lại checkpoint legacy bằng công thức paper.
- So sánh attention với baseline mới cần cùng chính sách, seed đánh giá, số mẫu và cấu hình metrics.
- Cấu hình legacy giữ nguyên. Muốn dùng công thức mới, chọn các cấu hình `paper_*` dưới đây.

## Đã kiểm tra

Hai smoke run chỉ huấn luyện **1 epoch**, trên tập con nhỏ và lưới đánh giá thô:

- Baseline: `results/trained_models/base_mdn/imptc/smoke_paper_peds_imptc/runs/smoke_paper_baseline_seed2024`.
- Attention: `results/trained_models/attention_mdn/imptc/smoke_paper_attention_peds_imptc/runs/smoke_paper_attention_seed2024`.

Cả hai lưu best/last/final/epoch_0001, history, official validation metrics, 8 mẫu cố định và raw/decoded outputs. Các mẫu X/y/ID giống nhau giữa hai mô hình; X của mẫu cố định giống baseline cũ. Đánh giá NLL thử trên 32 mẫu test hoàn tất. Đã kiểm tra công thức, gradient hữu hạn, covariance với raw sigma=-100/rho=100, cấu hình sai bị từ chối, và decoded outputs legacy khớp dữ liệu cũ.

Đã kiểm tra việc tạo diagnostic bằng batch lỗi giả lập: `diagnostics/<split>_epoch_<epoch>_batch_<start>.json` ghi lỗi, khoảng sigma/rho/covariance và ID thật; `.npz` lưu X, y, raw output và decoded outputs. Việc bổ sung ID không đổi chọn dữ liệu hay RNG.

**Smoke test không chứng minh ổn định đủ 2500 epoch. Chưa chạy full training mới.**

## Bạn chạy full training

Chạy từ thư mục gốc repo, lần lượt hai lệnh:

```bash
.venv/bin/python base_mdn/train.py \
  -t imptc -c paper_peds_imptc.json \
  --run-id imptc_baseline_paper_seed2024 -l -p

.venv/bin/python attention_mdn/train.py \
  -t imptc -c paper_attention_peds_imptc.json \
  --run-id imptc_attention_paper_seed2024 -l -p
```

Hai cấu hình dùng K=3, seed=2024, 2500 epoch, batch=4096, Adam, lịch LR và dữ liệu như các cấu hình gốc. Checkpoint 1/5/10, mỗi 100 epoch, validation metrics mỗi 250 epoch, last mỗi epoch. Không train lại mode-consistent trong quy trình này.

Kết quả mới:

```text
results/trained_models/base_mdn/imptc/paper_peds_imptc/runs/imptc_baseline_paper_seed2024/
results/trained_models/attention_mdn/imptc/paper_attention_peds_imptc/runs/imptc_attention_paper_seed2024/
```

Run ID đã tồn tại sẽ bị từ chối khi train từ đầu. Nếu muốn thí nghiệm khác, dùng ID mới. Không resume từ run legacy.

## Đánh giá sau training

Các lệnh sau chạy evaluator chính thức trên toàn bộ test, có thể mất thời gian:

```bash
.venv/bin/python base_mdn/evaluate_run.py \
  --config paper_peds_imptc.json \
  --run-id imptc_baseline_paper_seed2024 --official

.venv/bin/python attention_mdn/evaluate.py \
  --config paper_attention_peds_imptc.json \
  --run-id imptc_attention_paper_seed2024 --official \
  --baseline-evaluation results/trained_models/base_mdn/imptc/paper_peds_imptc/runs/imptc_baseline_paper_seed2024/testing/evaluation.json
```

Mỗi run có `testing/evaluation.json`: test NLL, epoch best, chính sách phân phối, cấu hình đánh giá và official metrics. Attention ghi thêm chênh lệch với baseline mới nếu truyền `--baseline-evaluation`. Run paper không tự lấy metrics baseline legacy trong comparison.json.

## Visualization sau training

```bash
.venv/bin/python attention_mdn/visualize.py \
  --run-dir results/trained_models/attention_mdn/imptc/paper_attention_peds_imptc/runs/imptc_attention_paper_seed2024 \
  --baseline-run results/trained_models/base_mdn/imptc/paper_peds_imptc/runs/imptc_baseline_paper_seed2024

.venv/bin/python attention_mdn/visualize_learning.py \
  --run-dir results/trained_models/attention_mdn/imptc/paper_attention_peds_imptc/runs/imptc_attention_paper_seed2024 \
  --sample-index 0
```

Có thể chọn sample-index từ 0 đến 7. NLL có ở từng epoch; phân phối qua các epoch chỉ có ở những checkpoint được lưu. Figures nằm trong thư mục `figures` của run attention.
