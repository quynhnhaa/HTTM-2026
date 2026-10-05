# So sánh base_mdn với K = 1, 2, 3, 5, 8

Nguồn: kết quả test đã lưu tại `results/ablations/imptc_num_gaussians/ablation.json`. Không chạy lại training hoặc evaluation.

## Protocol đã kiểm tra

IMPTC test, seed2024; best checkpoint theo minimum validation NLL. Cả5run completed2500epoch. Đã đọc checkpoint và xác nhận K/epoch; resolved_config của5run chỉ khác model_params.num_gaussians. LSTM hidden8, input32, output48, Adam1e-3, batch4096, train/validation reduction0.5; test batch128, best-of20samples, confidence MC1000, mesh±18m/0.1m, 6horizons. Metadata không chỉ định covariance mới: decoder legacy.

## Kết quả

| Metric | K=1 | K=2 | K=3 | K=5 | K=8 |
|---|---:|---:|---:|---:|---:|
| Test NLL ↓ | -0.344557 | -0.926523 | -1.092095 | -1.131680 | **-1.206566** |
| minADE20 (m) ↓ | 0.530000 | 0.473000 | 0.464000 | 0.460000 | **0.452000** |
| minFDE20 (m) ↓ | 0.648000 | 0.610000 | 0.607000 | 0.603000 | **0.590000** |
| Ravg (%) ↑ | 86.498770 | 95.799200 | 97.470900 | 97.906238 | **98.494310** |
| Rmin (%) ↑ | 76.530920 | 90.245317 | 94.064592 | 95.946626 | **96.587223** |
| S68 ↓ | 3.819278 | 5.949270 | 1.357518 | 2.200876 | **0.902178** |
| S95 ↓ | 8.140516 | 11.835262 | 6.061026 | 7.627523 | **5.348166** |
| ASAEE ↓ | 0.292837 | **0.232817** | 0.235288 | 0.243113 | 0.239394 |
| Best epoch | 1961 | 1961 | 1961 | 2473 | 2473 |
| Số tham số | 3040 | 5632 | 8224 | 13408 | 21184 |

## Nhận xét

- K=8 tốt nhất7/8metric chất lượng; K=2 tốt nhất ASAEE. K=8 cũng có validation NLL thấp nhất trong các kết quả này.
- K=3 là mốc gọn hơn:8224tham số; K=8 có21184tham số (~2.58lần). Không có K tốt nhất theo mọi tiêu chí chất lượng và chi phí.
- K=5 cải thiện NLL/ADE/FDE/reliability so với K=3, nhưng sharpness và ASAEE kém hơn. Tăng K không bảo đảm mọi metric cải thiện đơn điệu.
- K=1 kém rõ về NLL và reliability; K=2 cải thiện lớn, nhưng S68/S95 cao trong lần chạy này. Chưa suy ra nguyên nhân chỉ từ bảng metric.
- Một seed và tập5giá trị chưa chứng minh K=8 tối ưu toàn cục;8là giá trị lớn nhất đã thử. Không dùng bảng test này để chọn/tune tiếp phương pháp Bayesian; các quyết định phát triển dùng train/validation.

## So với các biến thể mới

Baseline K=8 đã có sẵn và cần được dùng cùng K=3 làm mốc tham chiếu cho gated/Bayesian. Đề xuất trước đó về chạy baseline K=8 có thể tận dụng artifact hiện có trước. Tuy nhiên, các biến thể mới dùng full-validation selection thay vì reduction0.5, nên đây chưa phải đối chứng hoàn toàn đồng nhất về protocol training/selection.

K=8 baseline so với gated K=8 đã test:

| Metric | Base K=8 | Gated K=8 |
|---|---:|---:|
| Test NLL ↓ | -1.206566 | -1.165655 |
| minADE20 (m) ↓ | 0.452000 | 0.453000 |
| minFDE20 (m) ↓ | 0.590000 | 0.590000 |
| Ravg (%) ↑ | 98.494310 | 98.948896 |
| Rmin (%) ↑ | 96.587223 | 97.656542 |
| S68 ↓ | 0.902178 | 1.047979 |
| S95 ↓ | 5.348166 | 5.262806 |
| ASAEE ↓ | 0.239394 | 0.240322 |

Gated tốt hơn Ravg/Rmin/S95, kém hơn NLL/ADE/S68/ASAEE, FDE bằng nhau trong số liệu lưu. Vì vậy việc gated tốt hơn baseline K=3 không đủ để quy lợi ích cho gate. Bayesian full run vẫn đang chạy; chưa có test để so sánh.
