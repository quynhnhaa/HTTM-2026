# So sánh test: baseline stable và attention stable

Toàn bộ 56.694 mẫu test IMPTC; seed 2024; evaluator stable_v2_ecdf_rng.

Best checkpoint chọn bằng full validation: baseline epoch 2792, attention epoch 2977.

| Chỉ số | Hướng | Baseline stable | Attention stable | Attention − baseline |
|---|---|---:|---:|---:|
| Test NLL / vị trí | ↓ | -0.926869 | -1.107224 | -0.180354 |
| minADE20 (m) | ↓ | 0.468000 | 0.436000 | -0.032000 |
| minFDE20 (m) | ↓ | 0.612000 | 0.541000 | -0.071000 |
| Ravg (%) | ↑ | 98.181587 | 98.189501 | +0.007914 |
| Rmin (%) | ↑ | 94.626415 | 92.677356 | -1.949060 |
| S68 (m²/s) | ↓ | 0.847620 | 3.583653 | +2.736033 |
| S95 (m²/s) | ↓ | 5.735489 | 6.897894 | +1.162405 |
| ASAEE (m/s) | ↓ | 0.236905 | 0.232492 | -0.004413 |
| Số tham số | ↓ | 8224.000000 | 7890.000000 | -334.000000 |
| Inference (ms/batch) | ↓ | 0.256383 | 0.519312 | +0.262928 |

## Giao thức và giới hạn

- Sigma floor 0.01 m, rho limit 0.999, exp guard 1000 m ở cả hai.
- NLL trung bình trên tất cả 48 future timestep; ADE/FDE evaluator dùng sáu mốc 0.8–4.8 s, K=20.
- Evaluator sắp mẫu theo mật độ riêng tại từng timestep; không coi đây là 20 quỹ đạo có liên kết theo thời gian.
- Reliability dùng empirical CDF đúng threshold; sharpness giữ phép tổng hợp local và diện tích grid đã sửa.
- Cả hai đã train 2500 epoch rồi continuation 500 epoch, LR mới 1e-5 đến 1e-7 và giữ trạng thái Adam.
- Đây là một seed; chưa có ablation tách đóng góp attention khỏi decoder. Không khẳng định ý nghĩa thống kê.
- Sharpness cần đọc cùng reliability. Thời gian inference là trung bình batch của evaluator, chưa phải benchmark độc lập với warm-up.
- Không so trực tiếp với metric legacy khác phiên bản evaluator. Không chọn checkpoint hoặc quyết định train tiếp bằng test.

Nguồn: hai file evaluation.json được ghi trong comparison.json.
