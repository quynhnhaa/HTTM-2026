# Mode-consistent trajectory MDN (IMPTC)

Phương pháp này được triển khai trong thư mục riêng. Không sửa `base_mdn/` hoặc checkpoint baseline. Giải thích động cơ và hình minh họa: [MODE_CONSISTENT_MDN.md](../docs/mode_consistent_mdn/MODE_CONSISTENT_MDN.md).

Kết quả huấn luyện và so sánh với baseline M3: [RESULTS.md](RESULTS.md).

## Phương pháp

Mô hình vẫn nhận 32 bước × 4 đặc trưng IMPTC, dùng một LSTM có hidden size 8 và ba Gaussian hai chiều. `mode_head` tạo **một** bộ trọng số `pi[B, 3]` cho toàn bộ 48 bước. `trajectory_head` tạo các tham số `mu_x, mu_y, log_sigma_x, log_sigma_y, raw_rho` cho mỗi cặp bước–mode.

`forward` trả tensor `[B, 48, 18]` theo đúng layout MDN của repo, trong đó ba mode logit được lặp lại theo 48 bước. Nhờ đó tracker mẫu cố định và evaluator biên của repo dùng được mà không thay mã baseline. Hàm loss riêng tính:

$$
\mathcal L = -\frac{1}{48}\;\mathbb E_B\!\left[
\log \sum_{k=1}^{3}\pi_k
\exp\!\left(\sum_{h=1}^{48}\log \mathcal N_2(y_h;\mu_{h,k},\Sigma_{h,k})\right)
\right].
$$

NLL trong `history.csv` là **joint trajectory NLL chia cho 48**, khác NLL từng bước của baseline. Không so sánh trực tiếp hai cột NLL này. `evaluate.py` báo thêm `marginal_nll_per_position` để có một chẩn đoán cùng định nghĩa với baseline.

## Cấu hình và artefact

- Config đầy đủ: [configs/imptc/mode_consistent_peds_imptc.json](configs/imptc/mode_consistent_peds_imptc.json). Ngoài phương pháp, các thông số huấn luyện, seed, mức giảm dữ liệu, mốc lưu và metric được giữ như cấu hình M3 `default_peds_imptc.json`.
- Config kiểm tra nhanh: [configs/imptc/smoke_peds_imptc.json](configs/imptc/smoke_peds_imptc.json). Dữ liệu và lưới metric đều giảm mạnh; **không dùng kết quả smoke để so sánh khoa học**.
- Artefact run: `results/trained_models/mode_consistent_mdn/imptc/<config_name>/runs/<run_id>/` gồm `history.csv`, `metrics/`, `checkpoints/{best,last,final}.pt`, `fixed_samples/{inputs.npz,predictions/}` và manifest.
- Manifest mẫu cố định dùng cùng danh sách validation và checksum như baseline M3.

## Chạy

Từ thư mục gốc repo:

```bash
.venv/bin/python mode_consistent_mdn/train.py \
  --target imptc --configs smoke_peds_imptc.json \
  --run-id imptc_mode_consistent_smoke --gpu 0
```

Có thể kiểm tra lệnh lấy mẫu quỹ đạo trên một phần nhỏ test set (đầu ra ghi rõ `evaluation_limited.json`):

```bash
.venv/bin/python mode_consistent_mdn/evaluate.py \
  --run-id imptc_mode_consistent_smoke --config smoke_peds_imptc.json \
  --limit 32
```

Sau khi đã kiểm tra smoke và artefact, cấu hình huấn luyện đầy đủ là:

```bash
.venv/bin/python mode_consistent_mdn/train.py \
  --target imptc --configs mode_consistent_peds_imptc.json \
  --run-id imptc_mode_consistent_seed2024 --gpu 0
```

Tiếp tục một run chưa hoàn tất bằng `--run-id` cũ và `--resume <đường dẫn last.pt>` của chính run đó.

Đánh giá mô hình mới trên test, kèm kết quả lấy mẫu độc lập từ checkpoint baseline M3 để đối chiếu:

```bash
.venv/bin/python mode_consistent_mdn/evaluate.py \
  --run-id imptc_mode_consistent_seed2024 \
  --baseline-checkpoint results/trained_models/base_mdn/imptc/default_peds_imptc/runs/imptc_baseline_seed2024/checkpoints/best.pt
```

Sau khi có checkpoint, tạo hình từ **đầu ra thực** của cùng 8 mẫu cố định và metric theo epoch:

```bash
.venv/bin/python mode_consistent_mdn/visualize.py \
  --run-id imptc_mode_consistent_seed2024 --sample-index 0
```

Hình được ghi vào `runs/<run_id>/figures/`. Các ellipse biểu diễn **từng Gaussian** với bán trục bằng một độ lệch chuẩn, không phải vùng tin cậy 68% của toàn hỗn hợp.

Có thể tạo ba ca nghiên cứu từ **toàn bộ test set** theo tiêu chí công khai: ADE thấp nhất, độ bất định cao trong nhóm ADE trung bình, và ADE cao nhất (chỉ xét người đi hơn 1 m):

```bash
.venv/bin/python mode_consistent_mdn/case_study.py \
  --run-id imptc_mode_consistent_seed2024 --gpu 0
```

Kết quả là `testing/case_selection.json` và `figures/07_test_case_studies.png`. Đây là phân tích định tính, không phải metric gốc.

Thêm `--official` để chạy evaluator gốc trên **toàn bộ test set** bằng phân phối GMM biên của biến thể. Kết quả này được lưu riêng dưới khóa `repository_official_marginal_metrics`. Có thể dùng `--limit N` cho kiểm tra nhanh; JSON khi đó mang tên `evaluation_limited.json`.

## Cách đọc kết quả

- `repository_official_marginal_metrics`: thuật toán đánh giá gốc trên GMM 2D tại từng bước, gồm Ravg, Rmin, S68, S95, ASAEE và minADE/minFDE theo cách lấy mẫu hiện có của repo.
- `variant_trajectory_sampling`: lấy 20 quỹ đạo, mỗi quỹ đạo giữ một mode xuyên suốt 48 bước; tính minADE/minFDE tại sáu thời điểm chính thức. Đây là **phân tích bổ sung**, không được gọi là metric gốc.
- `baseline_independent_sampling`: cùng số mẫu, seed, sáu thời điểm và công thức lỗi; baseline lấy mẫu GMM riêng tại từng bước. Hai khóa này so sánh **hai hệ thống hoàn chỉnh** (mô hình, loss và quy tắc lấy mẫu); riêng chúng không tách được tác động của từng thành phần.
- `mean_mode_weights` và `argmax_mode_fraction` giúp xem mô hình có dồn xác suất vào một mode không; chỉ số này không chứng minh ba mode có nhãn hành vi cụ thể.

Trước huấn luyện dài, kiểm tra `history.csv`, metric JSON, checkpoint, `fixed_samples/inputs.npz` và dự báo cố định ở epoch 1. Không diễn giải kết quả trước khi có checkpoint và đánh giá đầy đủ.
