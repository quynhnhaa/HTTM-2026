# kappa_mdn: K là một tham số của mô hình (cách 1)

Thư mục riêng, không sửa `base_mdn/` (có hash kiểm tra). Thiết kế và giao thức công bố trước: [spec](../docs/superpowers/specs/2026-10-04-kappa-mdn-design.md).

## Ý tưởng
Số thành phần K là số nguyên nên không có gradient. Ta làm mịn nó thành một số thực `kappa` cho **từng bước dự báo** (48 số, khởi tạo 12.0). Thành phần thứ `j` có một cổng mềm `sigmoid((kappa - j + 0.5) / tau)`: mở khi `j <= kappa`, đóng khi `j > kappa`. Cổng nhân vào trọng số π của thành phần đó (không chuẩn hoá lại khi train, để đóng cổng làm tăng NLL). `kappa = 1 + 15·sigmoid(kappa_logit)` luôn nằm trong [1, 16]. Xem mục 7 của spec: bản v1 thất bại vì κ sụp về 1.

- **Huấn luyện:** cổng mềm, loss = NLL chính xác + `lambda` nhân số cổng mở kỳ vọng (lambda = 0.01 nat mỗi thành phần). `tau` giảm tuyến tính từ 1.0 (epoch 1) xuống 0.1 (epoch 1250), rồi giữ nguyên.
- **Dự đoán và validation:** cổng cứng, K(t) = làm tròn `kappa[t]` (ít nhất 1); các thành phần `j > K(t)` có trọng số đúng bằng 0.
- `kappa` dùng learning rate gấp 10 lần; mọi thứ khác theo baseline (2500 epoch, batch 4096, seed 2024, K_max = 16).

K là **một số học được cho mỗi bước dự báo, chung cho mọi mẫu** (không phải K theo từng mẫu). K_max, lambda và lịch của tau là hằng số công bố trước.

## Cách chạy
```bash
.venv/bin/python -m unittest discover -s kappa_mdn/tests -t .             # 18 test
.venv/bin/python -m kappa_mdn.train --smoke --run-id kappa_smoke_seed2024 --gpu -1
.venv/bin/python kappa_mdn/review_smoke.py                                # ghi reports/SMOKE.json
.venv/bin/python -m kappa_mdn.train --full --run-id kappa_k16_seed2024 --gpu 0   # chỉ chạy khi người dùng đồng ý
```
`--full` bị từ chối nếu chưa có `reports/SMOKE.json` với trạng thái `passed`.

## Artifact mỗi epoch
`history.csv`: NLL thuần, hạng phạt, objective, validation NLL, `kappa` nhỏ nhất/trung bình/lớn nhất, K trung bình, tỉ lệ bước ở K_max, tau, số cổng mở mềm. Dự đoán cố định lưu thêm `kappa` và `support_size`.

## Giới hạn
Chưa có kết quả huấn luyện đầy đủ. Cổng mềm khi huấn luyện khác cổng cứng khi dự đoán nên có khoảng cách khi `tau` còn lớn. Kết quả phụ thuộc vào lambda. Không xử lý quá khớp của thành phần hẹp. Cổng cứng đóng cho trọng số đúng bằng 0, nên vấn đề sự kiện hiếm của sparsemax cũng có thể xảy ra.
