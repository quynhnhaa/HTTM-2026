# Trạng thái hiện tại: công thức legacy

Đã lưu phiên bản công thức theo dạng paper trong commit `29485b8`.
Phiên bản hiện tại quay về công thức cũ cho mọi phương pháp:

$$
\sigma=\exp(o_\sigma),\qquad \rho=\tanh(o_\rho).
$$

Không còn phần cộng 1 vào sigma hoặc hệ số co rho. Các cấu hình train paper đã được gỡ khỏi phiên bản hiện tại. Hạ tầng metadata, ghi log và bảo vệ checkpoint vẫn được giữ lại. Cấu hình không ghi chính sách được hiểu là legacy.

Checkpoint/result của các run paper và legacy trong `results/` không bị xóa hoặc sửa. Để tránh giải mã sai, code hiện tại từ chối cấu hình/checkpoint paper. Khi cần tái hiện hoặc đánh giá run paper, sử dụng mã ở commit `29485b8` trong checkout/worktree riêng. Tài liệu và các cấu hình paper đầy đủ cũng nằm trong commit đó.

## Train mới bằng công thức cũ

Chạy từ thư mục gốc repo. Dùng run ID mới, vì các run legacy trước đây đã tồn tại:

```bash
.venv/bin/python base_mdn/train.py \
  -c default_peds_imptc.json \
  --run-id imptc_baseline_legacy_rerun_seed2024 -l -p

.venv/bin/python attention_mdn/train.py \
  -c attention_peds_imptc.json \
  --run-id imptc_attention_legacy_rerun_seed2024 -l -p
```

Không có training nào được bắt đầu khi khôi phục công thức. Công thức cũ có thể gặp lại vấn đề covariance gần suy biến; không áp dụng thay đổi ổn định hóa khác trong lần khôi phục này.
