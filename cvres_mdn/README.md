# cvres_mdn: dự báo phần dư so với ngoại suy vận tốc không đổi

Giữ nguyên baseline (LSTM hidden 8, K = 3, hàm mất mát NLL, Adam, lịch learning rate, metric chính thức). Chỉ đổi **cách tạo tâm của các Gaussian**:

    tâm[b, k, i] = đầu ra của mạng[b, k, i] + v_cuối[b] * dt * (k + 1)      (cho mọi thành phần i, cả hai trục)

`v_cuối` là vận tốc (cột 2 và 3 của đầu vào) tại bước quan sát cuối, dt = 0.1. Các khối σ, ρ, π không đổi. Mô hình **không thêm tham số nào**; state dict có cùng khoá với baseline.

**Cảnh báo:** vì cùng khoá, nạp checkpoint này vào lớp baseline thường sẽ cho kết quả sai mà không báo lỗi (thiếu phần cộng ngoại suy). Luôn đánh giá bằng `cvres_mdn.evaluate` / `CVResidualMDN`.

Spec và số liệu đã đo: `docs/superpowers/specs/2026-10-04-cvres-mdn-design.md`, `reports/CV_PRIOR_CHECK.json` (chỉ validation).

## Chạy
    .venv/bin/python -m unittest discover -s cvres_mdn/tests -t .
    .venv/bin/python cvres_mdn/analyze_cv.py                 # kiểm tra giả định dữ liệu, ghi CV_PRIOR_CHECK.json
    .venv/bin/python -m cvres_mdn.train --smoke --run-id cvres_smoke_seed2024 --gpu -1
    .venv/bin/python cvres_mdn/review_smoke.py               # ghi reports/SMOKE.json (cổng cho --full)
    .venv/bin/python -m cvres_mdn.train --full --run-id cvres_seed2024 --gpu 0     # chỉ khi người dùng yêu cầu
    .venv/bin/python -m cvres_mdn.evaluate --config cvres_peds_imptc --run-id cvres_seed2024 --split validation --gpu 0
    # test: thêm --split test --confirm-test-once (chỉ một lần, chỉ khi người dùng yêu cầu)
