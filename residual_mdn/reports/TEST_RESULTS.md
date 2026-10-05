# So sánh test IMPTC: baseline stable, attention và residual

Toàn bộ 56.694 mẫu test, seed 2024, evaluator stable_v2_ecdf_rng. Best chọn theo full validation NLL.

| Model | Best epoch |
|---|---:|
| Baseline stable | 2792 |
| Attention stable | 2977 |
| Residual MDN | 2965 |

| Chỉ số | Baseline stable | Attention stable | Residual MDN | Residual − baseline |
|---|---:|---:|---:|---:|
| Test NLL / vị trí ↓ | -0.926869 | -1.107224 | -0.964047 | -0.037177 |
| minADE20 (m) ↓ | 0.468000 | 0.436000 | 0.457000 | -0.011000 |
| minFDE20 (m) ↓ | 0.612000 | 0.541000 | 0.592000 | -0.020000 |
| Ravg (%) ↑ | 98.181587 | 98.189501 | 98.236488 | +0.054901 |
| Rmin (%) ↑ | 94.626415 | 92.677356 | 93.054186 | -1.572230 |
| S68 (m²/s) ↓ | 0.847620 | 3.583653 | 3.070744 | +2.223124 |
| S95 (m²/s) ↓ | 5.735489 | 6.897894 | 8.478262 | +2.742772 |
| ASAEE (m/s) ↓ | 0.236905 | 0.232492 | 0.242489 | +0.005584 |
| Số tham số | 8224.000000 | 7890.000000 | 8224.000000 | +0.000000 |
| Inference (ms/batch) | 0.256383 | 0.519312 | 0.343075 | +0.086692 |

## Diễn giải và giới hạn

- Residual giữ kiến trúc/số tham số baseline; chỉ cộng prior CV, vận tốc trung bình 5 bước cuối, vào mean.
- Cả ba dùng covariance stable và lịch 2500 epoch + continuation500, Adam được giữ với LR continuation1e-5 đến1e-7.
- NLL dùng 48 timestep; ADE/FDE repo dùng sáu mốc và sắp sample riêng mỗi timestep, không phải sampling quỹ đạo chung.
- Đọc sharpness cùng reliability; chưa thể kết luận toàn diện chỉ bằng NLL/ADE/FDE.
- Đây là một seed; không suy ra ý nghĩa thống kê. Thời gian evaluator chưa phải benchmark độc lập có warm-up.
- Không so trực tiếp với legacy evaluator. Window velocity đã chốt trước run, không chọn bằng test.

Nguồn evaluation.json được ghi trong comparison.json.
