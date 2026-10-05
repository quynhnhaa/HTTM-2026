# sparsemax_mdn: MDN với trọng số trộn sparsemax

## Mục đích
Thay softmax trên K_max logit trọng số trộn của baseline MDN bằng sparsemax, để các thành phần không cần thiết có trọng số đúng bằng 0. Số thành phần hiệu dụng K(x, t) = |{k : pi_k(x, t) > 0}| do mô hình tự sinh theo từng mẫu và từng bước dự báo, chỉ từ một lần huấn luyện NLL thông thường (không thêm tham số, không thêm số hạng loss).

## Phạm vi khẳng định
- Không khẳng định đây là ý tưởng mới (chưa đối chiếu tài liệu).
- K(x, t) là số chỉ số đầu ra (head index) có trọng số dương, không phải số mode vật lý.
- Chưa có kết quả nào; README này không đưa ra kết luận về chất lượng.
- Giao thức, tiêu chí thành công và rủi ro được khai báo trước trong `docs/superpowers/specs/2026-10-04-sparsemax-mdn-design.md`.

## Thay đổi so với baseline (chỉ ba thứ)
1. Mô hình `SparsemaxMDN` (kế thừa `LSTM_Trajectory_Forecast`, cùng kiến trúc và khởi tạo): khối pi của đầu ra thô được thay bằng `log(pi)` với `pi = sparsemax(pi_logit)`, và hằng số hữu hạn `-1e9` khi `pi == 0`. Nhờ đó `decode_mdn_output` của baseline (softmax) trả về đúng trọng số sparsemax, mọi mã baseline (tracker, fixed-sample, `MDN_Forecaster`) dùng lại không đổi.
2. Loss `sparsemax_nll`: log-sum-exp có mặt nạ chính xác; thành phần có `pi == 0` không đóng góp và nhận gradient bằng đúng 0. Định nghĩa NLL không đổi (có test so với `NLL_MDN_loss` khi thay sparsemax bằng softmax).
3. Tracker `SparsemaxTracker`: thêm `mean_support_size`, `fraction_full_support`, `dead_components` (tập con cố định gồm 2048 mẫu validation đầu tiên) vào `history.csv`, thêm `support_size` vào file dự đoán mẫu cố định, và `architecture = 'sparsemax_mdn_v1'` vào checkpoint.

Lưu ý: `Categorical(probs)` của baseline kẹp xác suất 0 lên khoảng 1e-7 nên đánh giá chính thức có thể rò rỉ không đáng kể; điều này được ghi nhận, không che giấu.

`base_mdn/` không bị sửa; `base_hashes.py` kiểm tra SHA256 đầu và cuối mỗi lần chạy.

## Các tệp
- `sparsemax.py`, `model.py`, `loss.py`, `artifacts.py`, `train.py`, `analyze_k.py`, `review_smoke.py`, `base_hashes.py`
- `configs/imptc/sparsemax_k8_peds_imptc.json` (giống `default_peds_imptc.json`, chỉ khác `num_gaussians = 8` và `experiment_params` bổ sung), `configs/imptc/smoke_sparsemax_k8_peds_imptc.json`
- `tests/`, `reports/` (`BASE_SOURCE_HASHES.json`, `SMOKE.json`, `SMOKE.md`)

## Cách chạy
```bash
# unit test
.venv/bin/python -m unittest discover -s sparsemax_mdn/tests -t .
# kiểm tra base_mdn không đổi
.venv/bin/python -m sparsemax_mdn.base_hashes --check
# smoke (CPU)
.venv/bin/python -m sparsemax_mdn.train --smoke --gpu -1 --run-id sparsemax_smoke_seed2024
# review smoke (ghi reports/SMOKE.json; --full yêu cầu file này có status passed)
.venv/bin/python -m sparsemax_mdn.review_smoke
# huấn luyện đầy đủ: chỉ khi người dùng yêu cầu rõ ràng
.venv/bin/python -m sparsemax_mdn.train --full --gpu 0 --run-id ID
# phân tích K(x, t) (chỉ validation, thăm dò)
.venv/bin/python -m sparsemax_mdn.analyze_k --run-id ID --checkpoint best --gpu -1 [--limit N]
```
Kết quả nằm ở `results/trained_models/sparsemax_mdn/imptc/<config>/runs/<run-id>/`; phân tích ở `analysis/k_usage.json`.

Đánh giá một checkpoint (`sparsemax_mdn/evaluate.py`): tính NLL chính xác của hỗn hợp sparsemax và các metric chính thức của repo (minADE/FDE k=20, Ravg/Rmin, S68/S95, ASAEE) qua `MDN_Forecaster` không sửa. Mặc định là "clampedpi" (code gốc, `Categorical(probs=pi)` chặn trọng số 0 xuống khoảng 1.2e-7); `--exact-pi` dùng trọng số chính xác (π = 0 thì loại hẳn) chỉ trên instance forecaster; tên cờ nằm trong tên file kết quả `<run>/evaluation/<split>_<checkpoint>_<exactpi|clampedpi>[_limited].json`. `--limit N` chỉ cho validation (lấy N mẫu đầu, toàn bộ tập thay vì tập con ngẫu nhiên 50% lúc huấn luyện). Split `test` cần `--confirm-test-once`, không dùng với `--limit`, và chỉ được chạy một lần cho mỗi run (kiểm tra trước khi đọc dữ liệu hay checkpoint; file `test_evaluation_started.lock` giữ chỗ, nếu run lỗi giữa chừng thì người dùng phải quyết định xóa). Chỉ chạy test khi người dùng yêu cầu rõ ràng.
```bash
.venv/bin/python -m sparsemax_mdn.evaluate --run-id ID --split validation --limit 500 [--exact-pi] --gpu -1
.venv/bin/python -m sparsemax_mdn.evaluate --run-id ID --split test --confirm-test-once   # một lần duy nhất
```

Mỗi lần đánh giá còn ghi thêm `<split>_<checkpoint>_<cờ>[_limited]_persample.npz` (diện tích sharpness từng mẫu, sai số ADE theo mốc thời gian, tập tin cậy reliability, `sample_index`) và khối `sharpness_per_sample_summary` trong JSON.

Đánh giá sharpness từng mẫu cho checkpoint BASELINE softmax (`sparsemax_mdn/baseline_persample.py`): dùng cùng harness với `evaluate.py` (import lại các hàm, `MDN_Forecaster.evaluate` không sửa, seed 2024) để so sánh đuôi sharpness với sparsemax trên cùng một split. Thư mục run baseline chỉ đọc; kết quả ghi vào `results/trained_models/sparsemax_mdn/baseline_persample/<label>/<split>_best[_limited].json` và `..._persample.npz`. Split `test` cần `--confirm-test-baseline`, không dùng với `--limit`, chỉ một lần cho mỗi label (lock `test_evaluation_started.lock`, tạo trước khi đọc dữ liệu); với test, JSON có thêm `reproduction_vs_ablation_json` (so với `results/ablations/imptc_num_gaussians/ablation.json`, chỉ ghi lại chênh lệch).
```bash
.venv/bin/python -m sparsemax_mdn.baseline_persample --baseline-run results/trained_models/base_mdn/imptc/default_peds_imptc/runs/imptc_baseline_seed2024 --label K3 --split validation --limit 300 --gpu 0
```

## Ghi chú bản chụp hash (2026-10-05)
Bản chụp `reports/BASE_SOURCE_HASHES.json` được ghi lại để thêm đúng một file: `base_mdn/plot_training_history.py` (script vẽ đồ thị NLL của baseline do người dùng tạo ngày 2026-10-04, chỉ đọc history.csv, không train hay đánh giá). Mọi file khác của `base_mdn/` giữ nguyên hash như bản chụp trước (bản cũ ở `/tmp/sparsemax_hashes_before.json` khi ghi lại).
