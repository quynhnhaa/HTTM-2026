# LSTM–Attention–MDN trên IMPTC

Biến thể này ở trong thư mục riêng; mã và checkpoint baseline M3 không thay đổi. [Kết quả thực nghiệm](RESULTS.md) ghi rõ run dừng sớm do lỗi số học và phép đánh giá checkpoint tốt nhất.

Baseline chỉ đưa trạng thái LSTM cuối vào lớp tuyến tính để xuất cùng lúc 48 GMM. Ở đây LSTM trả lại toàn bộ 32 trạng thái quan sát. Mỗi mốc dự báo có một query học được, kết hợp trạng thái cuối rồi đọc 32 trạng thái bằng temporal cross-attention. Một MLP dùng chung xuất 18 tham số cho GMM ba thành phần ở từng mốc. Mô hình giữ loss NLL biên, dữ liệu 32 × 4, đầu ra `[B,48,18]`, seed và evaluator gốc.

`model_params.attention_heads=2` và `decoder_width=192` là tham số của kiến trúc mới; số tham số thực tế được ghi trong kết quả test. Giả thuyết: truy cập trực tiếp các đoạn lịch sử giúp dự báo các chuyển động thay đổi hướng mà trạng thái cuối nén chưa đủ. Hiệu quả phải được đo, không suy từ kiến trúc.

Chạy từ thư mục gốc repo:

```bash
.venv/bin/python attention_mdn/train.py --target imptc --configs smoke_attention_peds_imptc.json --run-id imptc_attention_smoke --gpu 0
.venv/bin/python attention_mdn/evaluate.py --run-id imptc_attention_smoke --config smoke_attention_peds_imptc.json --limit 32
.venv/bin/python attention_mdn/train.py --target imptc --configs attention_peds_imptc.json --run-id imptc_attention_seed2024 --gpu 0
.venv/bin/python attention_mdn/evaluate.py --run-id imptc_attention_seed2024 --official
```

`visualize_learning.py` tạo GIF và montage của dự báo trên cùng một mẫu validation qua các checkpoint đã lưu:

```bash
.venv/bin/python attention_mdn/visualize_learning.py \
  --run-id imptc_attention_seed2024 --sample-index 0
```

GIF/montage nằm trong `runs/<run_id>/figures/`. Dự báo cố định được lưu ở epoch 1, 5, 10, mỗi 100 epoch và best; biểu đồ NLL mới có dữ liệu từng epoch. Do đó animation đi qua mọi checkpoint có dự báo lưu sẵn, không phải đủ từng epoch.

`testing/evaluation.json` chứa NLL test, metric chính thức và delta so với baseline M3 trong `results/comparisons/imptc_m1_vs_m3/comparison.json`. Smoke chỉ kiểm tra pipeline và artefact, không dùng để kết luận khoa học.

## Huấn luyện với tham số hóa sigma/rho theo dạng paper

Xem [cấu hình, kiểm tra và lệnh train mới](../docs/MDN_PARAMETERIZATION.md). Các run cũ tiếp tục dùng legacy; hai run mới dùng cùng epsilon và lưu kết quả riêng.
