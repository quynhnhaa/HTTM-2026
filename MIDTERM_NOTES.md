# Ghi chú bài giữa kỳ: LSTM-MDN cho dự báo quỹ đạo xác suất

## 1. Yêu cầu môn học

Nhóm gồm ba sinh viên. Bài giữa kỳ yêu cầu tìm hiểu lý thuyết và demo một ứng dụng của mô hình hóa chuỗi thời gian nhiều chiều dựa trên GMM kết hợp với một mô hình xử lý chuỗi thời gian như HMM hoặc DNN.

Bài làm cần thể hiện:

- Gaussian Mixture Model;
- một mô hình xử lý chuỗi thời gian;
- dữ liệu chuỗi thời gian nhiều chiều;
- một ứng dụng cụ thể;
- dataset và code demo;
- phần lý thuyết;
- kết quả thực nghiệm và trực quan hóa;
- khả năng giải thích quá trình training và kết quả.

## 2. Paper và implementation chính

- Paper: **Reliable Probabilistic Human Trajectory Prediction for Autonomous Applications**.
- Hội nghị: ECCV Workshop 2024.
- Paper: <https://arxiv.org/abs/2410.06905>
- Official repository: <https://github.com/kav-institute/mdn_trajectory_forecasting>

Paper và official repository là nguồn tham chiếu chính. Những phát biểu về thuật toán, output, metric và preprocessing phải được kiểm chứng từ paper hoặc code thực tế.

## 3. Bài toán thực nghiệm

Thực nghiệm chính sử dụng IMPTC:

```text
Past multivariate trajectory
        |
        v
      LSTM
        |
        v
       MDN
        |
        v
pi, mu_x, mu_y, sigma_x, sigma_y, rho
        |
        v
2D GMM at each future timestep
        |
        v
Probabilistic future trajectory forecast
```

Mục tiêu đầu tiên là train lại baseline từ đầu với configuration của paper/repository càng sát càng tốt.

Baseline ban đầu không tự ý thay đổi:

- architecture;
- loss;
- optimizer;
- learning rate;
- số Gaussian;
- preprocessing;
- metric chính thức;
- các hyperparameter quan trọng.

Experiment bổ sung chỉ được xem xét sau khi baseline đã chạy và được đánh giá đúng.

## 4. Nội dung lý thuyết cần giải thích

### 4.1 Chuỗi thời gian nhiều chiều

Mỗi pedestrian trajectory là một chuỗi vector theo thời gian. Theo implementation hiện tại, input mặc định chứa bốn đặc trưng trên mỗi timestep. Ý nghĩa chính xác của từng chiều phải được liên hệ với preprocessing thực tế trước khi đưa vào báo cáo.

### 4.2 Vai trò của LSTM

LSTM xử lý chuỗi quan sát và tạo biểu diễn chứa thông tin chuyển động trong quá khứ. Cần giải thích hidden state, phụ thuộc theo thời gian và lý do dùng output cuối của LSTM trong implementation.

### 4.3 Vai trò của MDN

MDN không trực tiếp trả về một tọa độ duy nhất. Tầng đầu ra sinh tham số của một hỗn hợp Gaussian hai chiều cho mỗi future timestep.

Với component `k`:

```text
pi_k     = softmax(alpha logits)
mu_k     = [mu_x, mu_y]
sigma_x  = exp(raw_sigma_x)
sigma_y  = exp(raw_sigma_y)
rho      = tanh(raw_rho)
```

Covariance:

```text
Sigma_k = [[sigma_x^2,              rho * sigma_x * sigma_y],
           [rho * sigma_x * sigma_y,              sigma_y^2]]
```

Các phép biến đổi `softmax`, `exp` và `tanh` đảm bảo lần lượt tổng trọng số bằng một, độ lệch chuẩn dương và hệ số tương quan nằm trong khoảng hợp lệ.

### 4.4 Hàm loss

Baseline sử dụng negative log-likelihood của ground-truth position dưới mixture distribution. Cần giải thích trực giác: phân phối gán xác suất cao cho ground truth sẽ có NLL thấp hơn.

### 4.5 Giới hạn diễn giải

Implementation tạo một GMM hai chiều ở mỗi future timestep. Không được tự gọi đây là một joint GMM duy nhất cho toàn bộ future trajectory nếu chưa chứng minh được cấu trúc phụ thuộc xuyên thời gian từ code.

## 5. Kế hoạch lưu artifact trước full training

Phải thiết kế và kiểm tra logging trước khi chạy dài để tránh phải train lại.

Thông tin cần xem xét lưu:

- epoch;
- train NLL;
- validation NLL;
- learning rate;
- metric chính thức của repository/paper;
- checkpoint định kỳ;
- best checkpoint và final checkpoint;
- fixed sample indices/identifiers;
- observed trajectory;
- ground-truth future trajectory;
- prediction;
- raw MDN output hoặc `pi`, `mu`, `sigma_x`, `sigma_y`, `rho` đã decode.

Định dạng artifact cần được quyết định trước khi train, ưu tiên định dạng dễ tái sử dụng để vẽ hình mà không phải chạy lại model không cần thiết.

## 6. Fixed validation/test samples

Chọn khoảng 5-10 sample cố định trước full training và lưu index hoặc identifier ổn định.

Tại mỗi checkpoint được chọn:

- observed trajectory không đổi;
- ground truth không đổi;
- preprocessing không đổi;
- model prediction và GMM distribution là phần duy nhất thay đổi.

Các mốc mong muốn ban đầu gồm epoch 1, 5, 10, các mốc định kỳ hợp lý, best và final. Cần cân đối mốc lưu với cấu hình 2.500 epoch của baseline và chi phí evaluation thực tế.

## 7. Visualization bắt buộc

### Hình 1 — Training và validation loss

- Trục x: epoch.
- Trục y: NLL.
- Mục tiêu: đánh giá hội tụ, khoảng cách train-validation và dấu hiệu overfitting.

### Hình 2 — Metric chính thức theo epoch

Chỉ sử dụng metric thực sự có trong paper/repository. Hình cho thấy chất lượng dự báo và/hoặc uncertainty thay đổi trong quá trình học.

### Hình 3 — Past, ground truth và prediction

Vẽ trên mặt phẳng x-y:

- observed/past trajectory;
- ground-truth future;
- predicted future.

Hình phải giúp người xem hiểu trực tiếp input, output và sai lệch của bài toán trajectory prediction.

### Hình 4 — GMM contour hoặc covariance ellipse

Tại một future timestep, thể hiện:

- từng `pi_k`;
- từng tâm `mu_k`;
- covariance ellipse hoặc Gaussian contour từ `Sigma_k`;
- ground-truth point.

Đây là hình liên kết trực tiếp implementation với lý thuyết GMM.

### Hình 5 — Uncertainty theo forecast horizon

Vẽ phân phối hoặc vùng uncertainty tại nhiều future timestep bằng output thực tế. Không áp đặt giả định uncertainty phải tăng theo thời gian.

### Hình 6 — Một fixed sample qua nhiều epoch

So sánh cùng một sample tại các checkpoint. Cần dùng chung giới hạn trục và cách biểu diễn để thay đổi của prediction dễ quan sát và không gây hiểu nhầm.

## 8. Phân tích case sau training

Chọn ít nhất:

- **Good case:** dự báo chính xác;
- **Difficult/uncertain case:** phân phối rộng hoặc có nhiều khả năng đáng chú ý;
- **Failure case:** sai số lớn hoặc phân phối không phản ánh đúng ground truth.

Không chỉ cherry-pick kết quả đẹp. Failure case được dùng để phân tích limitation, dữ liệu đầu vào, multimodality và chất lượng uncertainty.

Tiêu chí chọn case cần được định nghĩa từ metric hoặc đặc điểm phân phối thay vì chỉ dựa vào cảm nhận hình ảnh.

## 9. Metric cần nghiên cứu từ repository

Repository hiện có các nhóm metric/đánh giá sau, nhưng công thức và cách aggregate phải được đọc kỹ trước khi trình bày:

- best-of-K `minADE@K`;
- best-of-K `minFDE@K`;
- AEE/ASAEE;
- reliability;
- sharpness;
- inference time.

Không được đánh đồng accuracy với uncertainty calibration. Một dự báo có sai số vị trí thấp chưa chắc có uncertainty đáng tin cậy, và ngược lại.

## 10. Trình tự thực hiện dự kiến

1. Đọc paper và mapping từng khái niệm sang source code.
2. Kiểm tra dataset IMPTC đã tiền xử lý và cấu trúc sample.
3. Xác nhận chính xác input feature, tensor shape và coordinate system.
4. Audit training, evaluation, checkpoint và visualization hiện có.
5. Thiết kế bổ sung logging và fixed-sample capture tối thiểu.
6. Chạy smoke test ngắn để kiểm chứng artifact.
7. Chốt baseline protocol và seed/reproducibility information.
8. Chạy full baseline.
9. Sinh visualization và chọn case theo tiêu chí rõ ràng.
10. Phân tích kết quả, limitation và chuẩn bị demo/thuyết trình.

## 11. Phân công nhóm

Chưa chốt. Có thể cân nhắc ba mảng có liên kết chặt chẽ:

1. Lý thuyết GMM/MDN và hàm NLL.
2. LSTM, dữ liệu chuỗi thời gian, preprocessing và training pipeline.
3. Evaluation, reliability/sharpness, visualization và demo.

Mỗi thành viên vẫn cần hiểu được pipeline tổng thể để trả lời câu hỏi khi thuyết trình.

## 12. Trạng thái hiện tại

- Đã xác định paper và official repository.
- Đã xác định IMPTC là dataset chính.
- Đã thiết lập môi trường Python cục bộ và kiểm tra GPU.
- Chưa chuẩn bị dataset IMPTC.
- Chưa thay đổi training pipeline cho logging/fixed samples.
- Chưa chạy full training.
- Chưa chốt protocol thực nghiệm cuối cùng.

