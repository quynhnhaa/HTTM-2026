# crps_mdn: NLL + CRPS

Kiến trúc baseline giữ nguyên (LSTM hidden 8, K = 3, giải mã legacy). Chỉ đổi **hàm mất mát khi train**:

    loss = NLL + lambda * (CRPS_x + CRPS_y),   lambda = 1.0 (công bố trước, không chỉnh)

Phân phối biên của hỗn hợp Gaussian 2 chiều trên mỗi trục là hỗn hợp Gaussian 1 chiều (cùng trọng số, trung bình và độ lệch chuẩn của trục đó; rho không tham gia). CRPS của hỗn hợp 1 chiều có công thức đóng (Grimit et al. 2006), chính xác và khả vi, không cần lấy mẫu. Công thức ở `loss.py`. Validation và chọn checkpoint tốt nhất dùng NLL thuần, nên so sánh được với baseline.

Spec: `docs/superpowers/specs/2026-10-04-crps-mdn-design.md` (cấu hình và tiêu chí công bố trước, rủi ro).

## Chạy
    .venv/bin/python -m unittest discover -s crps_mdn/tests -t .
    .venv/bin/python -m crps_mdn.train --smoke --run-id crps_smoke_seed2024 --gpu -1
    .venv/bin/python crps_mdn/review_smoke.py            # ghi reports/SMOKE.json (cổng cho --full)
    .venv/bin/python -m crps_mdn.train --full --run-id crps_seed2024 --gpu 0    # chỉ khi người dùng yêu cầu

`history.csv` có thêm `train_objective` (NLL + lambda * CRPS) và `train_crps`; `train_nll` là NLL thuần.
Chưa có script đánh giá riêng; `shared_decoder_mdn/evaluate.py` có thể mở rộng khi cần (checkpoint cùng kiến trúc baseline).
Không hỗ trợ resume.
