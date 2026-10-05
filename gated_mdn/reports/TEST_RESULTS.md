# Full test: base_mdn, attention_mdn và gated_mdn

Full test gated_mdn hoàn tất thành công (exit code 0), 56.694 mẫu IMPTC, seed 2024. Dùng best.pt epoch 2474 được chọn bằng full validation NLL; test không dùng để chọn checkpoint.

| Metric | base_mdn | attention_mdn | gated_mdn |
|---|---:|---:|---:|
| Test NLL ↓ | -1.092095 | -1.177682 | -1.165655 |
| minADE20 (m) ↓ | 0.464000 | 0.446000 | 0.453000 |
| minFDE20 (m) ↓ | 0.607000 | 0.556000 | 0.590000 |
| Ravg (%) ↑ | 97.470900 | 97.881050 | 98.948896 |
| Rmin (%) ↑ | 94.064592 | 95.516845 | 97.656542 |
| S68 ↓ | 1.357518 | 1.967861 | 1.047979 |
| S95 ↓ | 6.061026 | 5.971077 | 5.262806 |
| ASAEE ↓ | 0.235288 | 0.228821 | 0.240322 |
| Số tham số | 8224 | 7890 | 21191 |

## K đã học

K_max = 8; K deterministic = 8; expected K = 7,991841. Cả tám deterministic gate đều bằng 1. Không thành phần nào bị loại. Run này chưa chứng minh tự tìm được K tối ưu hoặc giảm độ phức tạp.

## Nhận xét

- So với base_mdn: NLL, ADE, FDE, Ravg, Rmin, S68 và S95 tốt hơn; ASAEE kém hơn. Số tham số tăng từ 8.224 lên 21.191.
- So với attention_mdn: Ravg, Rmin, S68, S95 tốt hơn; NLL, ADE, FDE và ASAEE kém hơn.
- Không có mô hình tốt nhất trên tất cả metric.
- Không thể quy cải thiện cho cơ chế học gate: K tăng từ 3 lên 8, head lớn hơn, có stochastic gating khi train và protocol validation thay đổi. Cần đối chứng base_mdn K=8 không gate với cùng protocol trước khi đánh giá đóng góp riêng của gate.
- Gate chưa đóng có thể do mức phạt nhỏ, lợi ích giữ thành phần đối với NLL, hoặc tương tác gate với mixture logits; đây là giả thuyết, không phải nguyên nhân đã xác nhận.
- S68/S95 được giữ đúng công thức evaluator legacy của repository; không phải metric thay thế. Reliability ↑ được hiểu theo score chính thức, không phải tỷ lệ coverage thô.

## Nguồn

- Baseline: `results/comparisons/imptc_m1_vs_m3/comparison.json` (m3).
- Attention: `results/trained_models/attention_mdn/imptc/attention_peds_imptc/runs/imptc_attention_seed2024/testing/evaluation.json`.
- Gated: `results/trained_models/gated_mdn/imptc/gated_peds_imptc/runs/gated_k8_seed2024_tmux/testing/evaluation.json`.

Chưa chạy thêm training hoặc đối chứng.
