# Stable MDN: bản riêng để review trước training

Thư mục này được copy từ `base_mdn`. Baseline và artifact cũ được giữ nguyên.
Chưa chạy training. Đây là variant đề xuất, chưa có bằng chứng cải thiện kết quả.

## Parameterization nghĩa là gì?

Mạng sinh số thực raw; parameterization là cách đổi các số đó thành tham số
hợp lệ của GMM. Kiến trúc LSTM, layout output, mean, mixture weights và
marginal NLL được giữ nguyên. Chỉ sigma/rho của mode stable thay đổi.

| Tham số | Legacy | Stable |
|---|---|---|
| sigma | exp(raw) | exp(min(raw, log(1000))) + 0.01 |
| rho | tanh(raw) | 0.999 * tanh(raw) |
| pi | softmax(logits) | như legacy |
| mu | raw mean | như legacy |

Sàn 0.01 m (1 cm) là lựa chọn ban đầu để review, không phải hằng số paper
hoặc giá trị đã tối ưu. Margin rho tránh trị tuyệt đối bằng 1 do làm tròn.
Giới hạn phần exp ở 1000 m tránh overflow; không phải sàn sigma 1 m của paper.
Các giới hạn này không bảo đảm mọi lỗi số học hoặc divergence sẽ biến mất.
Guard trên exp làm gradient sigma bằng 0 khi raw vượt ngưỡng trên.
Metadata parameterization được lưu vào checkpoint; train/evaluate/capture
cùng dùng decoder. Checkpoint legacy tiếp tục được đọc bằng legacy;
mode paper vẫn bị từ chối. Không resume checkpoint legacy sang stable.

## NLL thay đổi từ đâu sang đâu?

Trong batch, cả hai vẫn tính:

```text
batch_nll = -sum(log p(y[i,h] | X[i])) / (B * 48)
```

Chỉ tổng hợp epoch được sửa, cho cả train và validation:

```text
Cũ: epoch_nll = sum(batch_nll[j]) / số_batch
Mới: epoch_nll = sum(B[j] * batch_nll[j]) / sum(B[j])
```

Ví dụ batch size 4, 4, 2 và NLL 1, 2, 3:
trung bình cũ là 2; trung bình mới là (4*1 + 4*2 + 2*3)/10 = 1.8.
Loss dùng backward không bị đổi bởi sửa tổng hợp; chọn best validation có thể đổi.

## Cách ly baseline và cấu hình

- Entry points dùng namespace kết quả `results/trained_models/stable_mdn/`.
- Config copy nguyên vẫn dùng legacy, để làm đối chứng với logging mới.
- `configs/imptc/stable_peds_imptc.json` bật mode stable rõ ràng.
- Fixed sample manifest vẫn tham chiếu manifest baseline để giữ cùng sample.
- Config stable dùng toàn bộ validation (`eval_data_reduction=1.0`), cùng tập
  cho validation NLL và metric; chọn best không dựa trên subset thay đổi.
- Config legacy copy nguyên vẫn giữ reduction cũ; muốn đối chứng cùng giao thức
  phải tạo config legacy với validation đầy đủ trước khi chạy.
- Calibration tính empirical CDF trực tiếp: `P(confidence <= threshold)`.
  Giữ tập threshold 0–1 của repo; không tuyên bố giống nguyên xi paper 0.01–0.99.
- `evaluate()` và `save_examples()` dùng seed evaluation riêng rồi phục hồi toàn
  bộ RNG Python/NumPy/Torch CPU/CUDA kể cả khi có exception; evaluation không
  làm lệch RNG training hoặc resume.
- Metric/report mang `evaluator_version=stable_v2_ecdf_rng`; không trộn với
  metric baseline cũ. Sharpness và thủ tục sampling ADE/FDE vẫn giữ như bản copy.
- Sửa thêm logger mặc định trong testing để hoạt động khi logging bị tắt.
- Chưa sửa `attention_mdn`; so sánh stable baseline với stable attention cần
  một variant attention riêng có cùng decoder và giao thức.

Lệnh tham khảo cho lúc đã được yêu cầu train (chưa thực thi):

```bash
.venv/bin/python stable_mdn/train.py --configs stable_peds_imptc.json --run-id imptc_stable_seed2024
.venv/bin/python stable_mdn/evaluate_run.py --config stable_peds_imptc.json --run-id imptc_stable_seed2024 --official
```

Trước full training phải review logging/checkpoint/fixed samples và chạy smoke test
có kiểm tra artifact theo yêu cầu. Không chạy `--configs all`: các config copy gồm
nhiều dataset và thí nghiệm không thuộc phạm vi variant IMPTC này.
