# growing_mdn: học K bằng cách tăng dần số component

Biến thể độc lập, không sửa `base_mdn/`. Mô hình bắt đầu với K=1 và tăng dần số component của đầu MDN theo lịch (K 1 -> K_max). K tăng ở mỗi phase; sau khi chạy xong tất cả các phase, K được chọn một lần bằng `select_k.py` trên validation với quy tắc có phạt. Kiến trúc LSTM, hàm loss, optimizer, learning rate, tham số hóa sigma/rho và các chỉ số chính thức được giữ như baseline, trừ các khác biệt nêu ở mục "Khác biệt so với baseline".

Thiết kế: `docs/superpowers/specs/2026-10-04-growing-mdn-design.md`. Kế hoạch: `docs/superpowers/plans/2026-10-04-growing-mdn.md`.

## Giới hạn tuyên bố

- K được chọn trên validation bằng quy tắc có phạt. Đây không phải chứng minh K tối ưu toàn cục; kết quả phụ thuộc epsilon, lịch huấn luyện và seed.
- "Component k" chỉ là chỉ số head, không phải mode nhất quán về vật lý giữa các mẫu hay các timestep.
- Cần nhiều seed trước khi khẳng định sự khác biệt giữa các K.
- Hiện mới có smoke. Chưa có full training và chưa có kết luận nào về K tốt nhất.

## Khác biệt so với baseline

- LinearLR được khởi động lại ở mỗi phase. Phase 0 dùng lr 1e-3; từ phase 1 trở đi lr khởi động lại từ `lr_default * growth.restart_lr_factor` (0.1, tức 1e-4) với cùng start/end factor.
- Revision 1 (sửa lỗi split không liên tục): offset của mu khi split là `growth.split_delta_relative` (0.05) nhân với median sigma của component nguồn trên tập train theo từng bước dự báo và từng trục, thay cho offset tuyệt đối 0.05 m (khóa `split_delta` cũ đã bị bỏ). Sau mỗi split trainer so validation NLL trước/sau; nếu không hữu hạn hoặc `|thay đổi| > growth.max_split_nll_change` (0.01 cho cấu hình đầy đủ, 0.05 cho smoke) thì ghi `aborted: true`, manifest `aborted_split_discontinuity` và dừng. Kiểm tra trên trọng số thật: `python -m growing_mdn.check_split_on_checkpoint` (kết quả trong `reports/SPLIT_CHECK.json`).
- Chọn K dùng validation NLL đầy đủ (baseline dùng reduction 0.5).
- Trạng thái tốt nhất theo từng phase (`best_k##.pt`) là trạng thái dùng cho split và metric của phase. File metrics đặt tên theo epoch cuối của phase được tính từ trọng số của epoch tốt nhất.
- `final.pt` là trạng thái tốt nhất của K_max.
- `metric_every=0`: metric chính thức chỉ được tính ở cuối mỗi phase (K_max-K_init+1 lần) thay vì mỗi 250 epoch như baseline; điều này ảnh hưởng đồ thị "official metrics vs epoch" (có thể dựng lại đường mịn hơn từ các checkpoint định kỳ `epoch_####`).
- `best_epoch`/`best_validation_nll` trong manifest là cực tiểu trên mọi K, không phải của K được chọn.
- `resolved_config.json` giữ `num_gaussians` = K ban đầu.
- File `metrics/epoch_####` ở cuối phase tính từ trọng số epoch tốt nhất, còn `epoch_####.pt` giữ trọng số epoch cuối; khi vẽ phải lấy `best_epoch` từ `phase_summary.json`.
- `MDN_Forecaster` ghi các file rls/ss/plot dùng chung trong thư mục evaluation cấp config và mỗi phase ghi đè; `metrics/*.json` mới là nguồn chuẩn.
- BIC/AIC chưa được script nào tính (`parameter_count` và `validation_nll` có trong `phase_summary.json` nên có thể bổ sung).
- `history`/`metrics` có thể có dòng trùng sau resume (khử trùng theo epoch khi vẽ).
- Mô hình lồng nhau khởi tạo từ phase trước nhìn thấy nhiều epoch hơn trên các trọng số chung, nên so với baseline K=3/K=8 phải so theo tổng compute.

## Cấu trúc

- `model.py`: mô hình và việc mở rộng đầu MDN (head growth).
- `optim_state.py`: ánh xạ lại trạng thái Adam khi mở rộng head.
- `selection.py`: chấm điểm và quy tắc chọn K.
- `artifacts.py`: tracker lưu artifact (history, checkpoint, mẫu cố định, tham số MDN).
- `train.py`: trainer theo phase; hook kiểm thử `--stop-after-epoch`, `--stop-after-phase`.
- `select_k.py`: chọn K từ validation, ghi `selection.json`. Dùng tolerance đóng băng trong `resolved_config.json` của run, yêu cầu run `completed` và đủ mọi K, từ chối ghi đè `selection.json` hoặc chạy sau khi đã có kết quả test chính thức.
- `evaluate.py`: đánh giá. Cần đúng một trong `--official` / `--limit N`. `--official` (duy nhất nơi test split được nạp) chỉ cho K trong `selection.json` của đúng run này và chỉ một lần mỗi run (từ chối nếu đã có `evaluation_k*.json`). `--limit N` chạy trên N mẫu VALIDATION đầu tiên, ghi `evaluation_k##_validation_limited.json`, chỉ để kiểm tra đường ống, không phải kết quả.
- `review_smoke.py`: review smoke, fail-closed.
- `base_hashes.py`: kiểm tra hash `base_mdn/` (`--check`).
- `configs/imptc/`: `growing_peds_imptc.json` (full), `smoke_growing_peds_imptc.json`.
- `tests/`: 37 unit test.
- `reports/`: `SMOKE.md`, `SMOKE.json`, `BASE_SOURCE_HASHES.json`.

## Cách chạy

```bash
# Unit test
.venv/bin/python -m unittest discover -s growing_mdn/tests -t .

# Smoke (K 1->3, 4 epoch, subset train nhỏ, 8 mẫu validation cố định)
.venv/bin/python -m growing_mdn.train --smoke --run-id <run_id>

# Review smoke
.venv/bin/python growing_mdn/review_smoke.py

# Full run: CHỈ sau khi người dùng duyệt rõ ràng (2600 epoch, K 1 -> 12)
.venv/bin/python -m growing_mdn.train --full --run-id <run_id>

# Chọn K trên validation
.venv/bin/python -m growing_mdn.select_k --run-id <run_id>

# Đánh giá test chính thức: một lần, sau khi quy tắc đã đóng băng
.venv/bin/python -m growing_mdn.evaluate --run-id <run_id> --official
```

## Trạng thái kiểm chứng (smoke)

Smoke: K 1->3, 4 epoch, subset train nhỏ, 8 mẫu validation cố định; metric chỉ tính trên 8 mẫu đó, không phải validation đầy đủ. Bốn kiểm tra resume (dừng sau epoch 1, sau epoch 2, sau phase 0 rồi resume, và resume ở phase cuối) tái hiện đúng lần chạy liền mạch, max |validation NLL diff| = 0.0 trên CPU. Delta NLL tại điểm split (split tương đối, Revision 1) là -0.0089 (1->2) và -0.0056 (2->3) trên mô hình gần như chưa huấn luyện, phụ thuộc dữ liệu; tính liên tục trên trọng số đã hội tụ được kiểm trong `reports/SPLIT_CHECK.json` và bằng unit test hồi quy. Hash `base_mdn/` không đổi.

## Tolerance chọn K (đề xuất, chờ người dùng xác nhận trước full run)

`epsilon_nll=0.02`, `ravg_tolerance_pp=0.5`, `sharpness_tolerance_ratio=0.10`.

## Giới hạn đã biết

- `finish_phase` lưu `phase_k##.pt` trước `phase_summary.json`. Resume từ `last.pt` an toàn; resume từ một checkpoint `phase_k##` sau khi crash trong khoảng đó có thể làm thiếu K trong summary.
- Khi resume chạy lại phần kết thúc phase, `metrics/history.csv` có thể có dòng trùng.
- Chưa có test tự động cho các nhánh fail-closed của `review_smoke.py`.
