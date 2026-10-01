# Kết quả LSTM–Attention–MDN so với baseline M3 trên IMPTC

## Giao thức

- Hai mô hình dùng IMPTC, đầu vào 32 × 4, 48 bước dự báo, 3 Gaussian 2D, seed 2024, cùng NLL biên và evaluator chính thức của repo.
- Baseline M3 lấy từ `results/comparisons/imptc_m1_vs_m3/comparison.json`, mục `label: m3`, checkpoint tốt nhất epoch 1961.
- Attention MDN dùng checkpoint tốt nhất theo validation NLL ở epoch 1740; 7890 tham số, so với 8224 của baseline.
- Run attention **dừng sớm ở epoch 1764 do NLL không hợp lệ**. Lần chạy lại từ `last.pt` tái hiện cùng lỗi. Một batch có `log_sigma` thấp tới khoảng -21,65, làm ma trận hiệp phương sai mất tính xác định dương trong float32. Checkpoint best trước lỗi vẫn được đánh giá trên toàn bộ test. Không thay mô hình, loss hay siêu tham số giữa run.

## NLL

| Chỉ số | Baseline M3 | Attention MDN |
| --- | ---: | ---: |
| Best validation NLL / vị trí ↓ | -1,150 | **-1,223** |
| Test NLL / vị trí ↓ | -1,092 | **-1,178** |
| Số tham số | 8224 | **7890** |

NLL cùng định nghĩa ở cả hai mô hình. Chỉ số test của attention tính trên toàn bộ 56.694 mẫu.

## Metric chính thức trên test

Evaluator gốc của repo chạy trên toàn bộ 56.694 mẫu, dùng cùng seed 2024 và GMM ba thành phần tại sáu mốc dự báo. Mũi tên chỉ hướng tốt hơn theo định nghĩa metric.

| Chỉ số | Baseline M3 | Attention MDN | Thay đổi |
| --- | ---: | ---: | ---: |
| minADE20 ↓ | 0,464 m | **0,446 m** | giảm 0,018 m (3,9%) |
| minFDE20 ↓ | 0,607 m | **0,556 m** | giảm 0,051 m (8,4%) |
| Ravg ↑ | 97,47% | **97,88%** | tăng 0,41 điểm % |
| Rmin ↑ | 94,06% | **95,52%** | tăng 1,45 điểm % |
| S68 ↓ | **1,358 m²/s** | 1,968 m²/s | rộng hơn 0,610 m²/s |
| S95 ↓ | 6,061 m²/s | **5,971 m²/s** | hẹp hơn 0,090 m²/s |
| ASAEE ↓ | 0,235 m/s | **0,229 m/s** | giảm 0,006 m/s |
| Thời gian suy luận / batch ↓ | **0,289 ms** | 0,531 ms | chậm hơn 0,242 ms |

**Kết luận:** checkpoint attention tốt hơn baseline M3 về test NLL, minADE20, minFDE20, Ravg, Rmin, S95 và ASAEE; S68 kém hơn và suy luận chậm hơn. Chênh lệch S95 rất nhỏ nên không nên diễn giải quá mạnh. Run attention không ổn định tới epoch 2500, vì vậy chưa thể gọi đây là một phương pháp huấn luyện hoàn chỉnh, ổn định. Một thí nghiệm tiếp theo nên xử lý sụp phương sai từ đầu rồi huấn luyện lại toàn bộ, thay vì sửa checkpoint giữa run.

## Tiến trình dự báo qua các epoch

- [GIF dự báo mẫu cố định 0 qua 21 checkpoint](../results/trained_models/attention_mdn/imptc/attention_peds_imptc/runs/imptc_attention_seed2024/figures/05_learning_progress_sample0.gif)
- [Montage các mốc huấn luyện](../results/trained_models/attention_mdn/imptc/attention_peds_imptc/runs/imptc_attention_seed2024/figures/05_learning_progress_sample0_montage.png)

Các dự báo được capture ở epoch 1, 5, 10, mỗi 100 epoch và best epoch 1740; chỉ NLL được ghi ở mọi epoch.

## Hình và artefact

- [NLL train/validation](../results/trained_models/attention_mdn/imptc/attention_peds_imptc/runs/imptc_attention_seed2024/figures/01_train_validation_nll.png)
- [Metric validation theo epoch](../results/trained_models/attention_mdn/imptc/attention_peds_imptc/runs/imptc_attention_seed2024/figures/02_official_metric_history.png)
- [Cùng 8 mẫu validation so với baseline](../results/trained_models/attention_mdn/imptc/attention_peds_imptc/runs/imptc_attention_seed2024/figures/03_fixed_samples_baseline_vs_attention.png)
- [Trọng số attention](../results/trained_models/attention_mdn/imptc/attention_peds_imptc/runs/imptc_attention_seed2024/figures/04_temporal_attention.png)
- [Run manifest](../results/trained_models/attention_mdn/imptc/attention_peds_imptc/runs/imptc_attention_seed2024/run_manifest.json)
- [Đánh giá test](../results/trained_models/attention_mdn/imptc/attention_peds_imptc/runs/imptc_attention_seed2024/testing/evaluation.json)

Bản đồ attention là trung bình của 8 mẫu cố định. Trọng số nhìn chung phân tán; không suy ra mô hình có hành vi chọn mốc lịch sử sắc nét chỉ từ hình này.
