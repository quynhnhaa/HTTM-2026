# Giải thích minADE₂₀, minFDE₂₀ và ASAEE

Ghi chú từ cuộc trao đổi, đối chiếu với `stable_mdn/eval.py`, `stable_mdn/vis.py` và cấu hình IMPTC. Chúng ta đã tìm hiểu sâu minADE và ASAEE; minFDE được bổ sung dưới đây để hoàn chỉnh bộ ba metric sai số vị trí. Sharpness mới được nhắc để phân biệt với ellipse, chưa được giải thích đầy đủ.

## 1. Dữ liệu và đầu ra mô hình

- Một **mẫu dữ liệu** là một chuỗi quan sát quá khứ cùng ground truth tương lai. Trong baseline đang thảo luận: 32 bước quan sát, 48 bước dự báo, mỗi bước 0.1 giây.
- LSTM → MDN dự đoán một GMM 2D tại mỗi bước tương lai. Ground truth tương lai dùng tính loss khi huấn luyện và sai số khi đánh giá, không dùng làm đầu vào dự đoán.
- Với M = 3, mỗi GMM có ba thành phần Gaussian. Mỗi thành phần có trọng số π, mean μ = (μx, μy), độ lệch chuẩn σx, σy và tương quan ρ.
- Covariance được dựng thành:

```text
Σ = [[σx²,       ρ σx σy],
     [ρ σx σy,       σy²]]
```

Ba Gaussian cùng đóng góp vào một phân bố; không mặc định là ba quỹ đạo hay ba hướng đi. Ellipse của chúng có thể chồng nhau. Các ellipse biểu diễn đường bao từ covariance, không trực tiếp là metric sharpness của toàn bộ GMM.

Evaluator baseline đánh giá tại sáu mốc: **0.8, 1.6, 2.4, 3.2, 4.0, 4.8 giây**, tương ứng chỉ số bước `[7, 15, 23, 31, 39, 47]`.

## 2. minADE₂₀: sai số trung bình tốt nhất trong 20 dự đoán

Với một mẫu dữ liệu:

1. Có 20 dự đoán vị trí tại các mốc đánh giá.
2. Với mỗi dự đoán k, tính khoảng cách Euclid đến ground truth tại từng mốc.
3. Lấy trung bình theo các mốc → một giá trị ADE cho dự đoán k.
4. Lấy nhỏ nhất trong 20 giá trị ADE → minADE₂₀ của mẫu dữ liệu.
5. Lấy trung bình minADE₂₀ trên toàn bộ mẫu dữ liệu để báo cáo.

Gọi eₙ,ₖ,ⱼ là khoảng cách Euclid của mẫu dữ liệu n, dự đoán k, mốc j:

```text
ADE(n, k) = (1/J) × Σⱼ e(n, k, j)
minADE₂₀(n) = minₖ ADE(n, k)
minADE₂₀ toàn tập = (1/N) × Σₙ minADE₂₀(n)
```

**Thứ tự là trung bình theo thời gian trước, lấy min sau. Không lấy min riêng tại từng bước rồi mới trung bình.**

Ví dụ hai dự đoán có sai số ở hai bước là `[1, 9]` và `[8, 2]`: ADE của cả hai bằng 5, nên minADE bằng 5; không phải `(1 + 2)/2 = 1.5`.

**Ý nghĩa:** trong 20 dự đoán, dự đoán tốt nhất gần ground truth tới mức nào trên các mốc tương lai. Đơn vị mét, càng thấp càng tốt. Đây là đánh giá best-of-20 dùng ground truth để chọn; khi triển khai thực tế chưa biết tương lai nên không thể chọn theo cách đó.

**Chi tiết riêng của repo:** các điểm lấy mẫu được sắp theo mật độ riêng tại từng timestep trước khi tính sai số. Vì vậy, không nên gọi đây là 20 quỹ đạo được lấy mẫu với liên kết thời gian. Code tính `euc_errors.mean(axis=2).min(axis=0)`.

## 3. minFDE₂₀: sai số cuối tốt nhất trong 20 dự đoán

Với mỗi mẫu dữ liệu, chỉ xét mốc cuối được đánh giá, ở baseline là **4.8 giây**:

```text
minFDE₂₀(n) = minₖ e(n, k, J)
minFDE₂₀ toàn tập = (1/N) × Σₙ minFDE₂₀(n)
```

Không lấy trung bình theo thời gian. Tính 20 khoảng cách tại mốc cuối, lấy nhỏ nhất, rồi trung bình trên tập dữ liệu.

**Ý nghĩa:** điểm cuối tốt nhất trong 20 dự đoán cách vị trí thật cuối bao nhiêu mét. Càng thấp càng tốt.

Dự đoán được chọn bởi minFDE có thể khác dự đoán được chọn bởi minADE. Một dự đoán có điểm cuối tốt chưa chắc tốt ở các mốc trước đó. Code tính `euc_errors[:, :, -1].min(axis=0)`.

## 4. ASAEE theo evaluator của repo

### 4.1. Tìm điểm dự đoán đại diện

Một điểm trên mặt phẳng có hai tọa độ `(x, y)`. Mật độ tại điểm đó là một số riêng:

```text
pₜ(x, y) = Σₖ πₖ,ₜ × Normal₂D((x, y); μₖ,ₜ, Σₖ,ₜ)
```

Evaluator tạo một lưới gồm L điểm tọa độ và dùng cùng lưới cho các mốc tương lai. Cùng tọa độ có thể có mật độ khác nhau tại các mốc vì GMM thay đổi theo thời gian. Có thể hình dung bảng mật độ kích thước L × J.

Ý tưởng là tìm điểm có mật độ cao nhất của **toàn bộ hỗn hợp**, không chỉ lấy mean của Gaussian có π lớn nhất hay mean hỗn hợp.

Repo thực hiện gián tiếp bằng confidence map. Với các mẫu sₘ lấy từ GMM:

```text
cₜ(z) = (1/M) × số mẫu sₘ có pₜ(sₘ) > pₜ(z)
điểm đại diện = điểm lưới có cₜ(z) nhỏ nhất
```

Điểm có mật độ cao có ít mẫu với mật độ cao hơn nó. Đây là xấp xỉ trên lưới bằng Monte Carlo; các điểm có thể đồng hạng nên không đảm bảo cực đại mật độ chính xác. Ground truth không tham gia lựa chọn điểm đại diện.

Mô hình xuất 48 GMM nhưng ASAEE baseline chỉ chọn điểm tại sáu mốc đánh giá → **sáu điểm đại diện cho mỗi mẫu dữ liệu**. Các điểm được chọn riêng từng mốc, không bị ép tạo thành quỹ đạo liên tục.

### 4.2. Tính sai số và tổng hợp

1. Với mỗi mẫu n và mốc tⱼ, tính khoảng cách Euclid giữa điểm đại diện và ground truth cùng mốc:

```text
eₙ(tⱼ) = sqrt((x̂ₙ(tⱼ) − xGTₙ(tⱼ))² + (ŷₙ(tⱼ) − yGTₙ(tⱼ))²)
```

2. Trung bình trên tất cả mẫu tại từng mốc:

```text
AEE(tⱼ) = (1/N) × Σₙ eₙ(tⱼ)
```

3. Chia AEE cho thời gian tương ứng, rồi tổng hợp đúng theo code:

```text
ASAEE = [Σⱼ AEE(tⱼ)/tⱼ] / (H × Δt)
```

H là số bước dự báo đầy đủ; Δt là thời gian mỗi bước. Với baseline H = 48, Δt = 0.1, mẫu số là **4.8**, không phải số mốc J = 6.

Ví dụ minh họa:

| Thời gian (s) | AEE (m) | AEE / thời gian |
|---:|---:|---:|
| 0.8 | 0.16 | 0.20 |
| 1.6 | 0.32 | 0.20 |
| 2.4 | 0.48 | 0.20 |
| 3.2 | 0.64 | 0.20 |
| 4.0 | 0.80 | 0.20 |
| 4.8 | 0.96 | 0.20 |

Theo code: ASAEE = (6 × 0.20)/4.8 = **0.25**.

### 4.3. Ý nghĩa và giới hạn diễn giải

ASAEE cho biết sai số của điểm dự đoán đại diện, có chuẩn hóa theo thời gian dự báo. Cùng một sai số khoảng cách, sai số ở tương lai gần đóng góp lớn hơn vì chia cho thời gian nhỏ hơn. Càng thấp càng tốt khi so sánh cùng giao thức đánh giá.

ASAEE không đo sai số vận tốc, cũng không cho biết uncertainty có được hiệu chỉnh tốt hay không. Để đánh giá chất lượng phân bố cần xem thêm reliability và sharpness.

**Lưu ý đơn vị:** repo báo ASAEE là `m/s`, nhưng công thức còn chia tổng AEE/t cho HΔt; xét đơn vị trực tiếp cho ra m/s². Đây là sự không nhất quán giữa công thức và nhãn đơn vị trong evaluator. Giữ nguyên kết quả baseline, ghi rõ “ASAEE theo evaluator của repo”, và không diễn giải ASAEE = 0.25 thành “mỗi giây mô hình lệch 0.25 m”.

## 5. So sánh ba metric

| Metric | Dự đoán dùng để đánh giá | Tổng hợp | Thấp hơn tốt hơn |
|---|---|---|---|
| minADE₂₀ | Tốt nhất trong 20 theo ADE | Trung bình theo mốc → min → trung bình theo mẫu | Có |
| minFDE₂₀ | Tốt nhất trong 20 tại mốc cuối | Min ở mốc cuối → trung bình theo mẫu | Có |
| ASAEE của repo | Điểm đại diện gần mode GMM ở từng mốc | AEE theo mốc → chia thời gian → tổng hợp theo code | Có |

minADE/minFDE dùng ground truth để chọn dự đoán tốt nhất. ASAEE chọn điểm đại diện từ phân bố trước, rồi mới dùng ground truth để đo sai số.

## 6. Nguồn mã để đối chiếu

- `stable_mdn/eval.py`: sampling và sắp hạng theo timestep; tính minADE/minFDE; chọn điểm đại diện; tính khoảng cách; `build_confidence_set_mdn`.
- `stable_mdn/vis.py`: `plot_aee_over_time`, công thức ASAEE.
- `stable_mdn/configs/imptc/default_peds_imptc.json`: Δt, forecast horizon và các mốc đánh giá.
