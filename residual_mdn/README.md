# Residual LSTM–MDN

Variant riêng của baseline stable, 8224 tham số. Không thay các thư mục model cũ.

## Thay đổi duy nhất về mô hình

Input IMPTC giữ [x,y,vx,vy] trong hệ tọa độ ego. Lấy trung bình vx/vy của 5
bước quan sát cuối; đây là 5 vận tốc của 5 khoảng 0.1 s, không dùng tương lai.

```text
v_hat = mean(X[:, -5:, 2:4])
CV[h] = X[:, -1, :2] + v_hat * h * 0.1   (h=1..48)
mu[h,k] = CV[h] + delta_mu[h,k]
```

LSTM/linear head giống baseline; hai nhóm mean output được hiểu là residual.
Không thêm tham số. Forward trả mean vị trí cuối cùng, nên loss, evaluator và
fixed capture dùng đúng phân phối vị trí mà không cần sửa decoder.
Sigma, rho, pi giữ như stable_mdn, K=3. Target vẫn là vị trí; marginal NLL không
đổi. Không thêm MSE hoặc regularization. Số bước velocity=5 là lựa chọn đã chốt
trước run, chưa tối ưu; không chọn lại bằng test.

## Giao thức

- Full validation, sample-weighted epoch NLL và evaluator stable_v2_ecdf_rng.
- Cùng seed2024, Adam, batch4096, train reduction0.5, fixed samples và lịch
  2500 epoch như baseline stable.
- Nhánh continuation riêng 500 epoch, LR1e-5 đến1e-7, giữ Adam và best parent.
- Raw output trong tracker là output forward đã cộng CV; residual gốc và CV
  được lưu bổ sung vào fixed samples để phân tích, không làm thay đổi loss.
- Metric repo dùng 6 timestep cho ADE/FDE; không coi sample rank là quỹ đạo chung.
- Đánh giá test sau khi đã chốt protocol, không dùng test để chọn hyperparameter.

Các file model.py, train.py, evaluate.py, continue_training.py dùng pipeline
chung từ stable_mdn. Test unit không training; smoke IMPTC là run riêng.
