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

## 12. Trạng thái đã kiểm tra ngày 03/10/2026

- IMPTC đã có dữ liệu tiền xử lý và báo cáo audit trong `reports/data_audit/`.
- Pipeline đã có history, checkpoint best/periodic/final, resume và capture mẫu cố định.
- Baseline M3 đã có kết quả; Attention-MDN có checkpoint best epoch 1740, nhưng run legacy dừng do divergence ở epoch 1764.
- Công thức legacy hiện nhất quán giữa loss, evaluator và capture trong `base_mdn`: `sigma = exp(raw_sigma)`, `rho = tanh(raw_rho)`.
- Các run paper vẫn được giữ riêng; decoder hiện từ chối `mode="paper"`. Không đổi metadata checkpoint paper thành legacy để sử dụng lại.
- Một số vấn đề về tổng hợp NLL, validation và calibration còn cần xử lý trước thực nghiệm mới.
- Các bước dưới đây là đề xuất; chưa phải quyết định triển khai hoặc cho phép chạy training.

## 13. Hướng đồ án được khuyến nghị

**Tên đề tài đề xuất:** Dự báo quỹ đạo người đi bộ bằng LSTM–MDN và đánh giá cải tiến với temporal attention.

Lấy baseline M3 và Attention-MDN làm trọng tâm. Bổ sung một ablation để xác định phần đóng góp của temporal attention, thay vì mở rộng nhiều kiến trúc cùng lúc.

**Câu hỏi nghiên cứu:** Temporal attention có cải thiện dự báo xác suất từ lịch sử chuyển động không, và cải thiện đó đánh đổi gì về sharpness, tốc độ suy luận và độ ổn định training?

Đóng góp phù hợp với phạm vi giữa kỳ:

1. Giải thích và tái lập baseline LSTM–MDN trên IMPTC theo implementation chính thức.
2. Xây dựng và đánh giá variant temporal attention.
3. Tách đóng góp của attention khỏi thay đổi decoder bằng ablation.
4. Phân tích chất lượng phân phối bằng NLL, reliability và sharpness cùng displacement metrics.

## 14. Thứ tự thực hiện tiếp theo

### Bước 1: Chuẩn hóa logging và evaluation

Các vấn đề đã phát hiện trong mã hiện tại:

- NLL mỗi batch là trung bình trên `B × 48` vị trí, đúng với marginal NLL của baseline.
- NLL epoch đang lấy trung bình các batch mà không cân theo số mẫu. Cần tính `sum(batch_nll × batch_size) / sum(batch_size)` để batch cuối không bị tăng trọng số.
- Validation hiện lấy ngẫu nhiên 50% mỗi lần gọi. Nên dùng toàn bộ validation hoặc một subset cố định, có manifest, để chọn best checkpoint.
- Validation NLL và metric có thể dùng hai subset khác nhau vì evaluator gọi lại `get_eval_data()`.
- Calibration trong `vis.py` bị lệch bin so với empirical CDF tại đúng threshold. Cần kiểm chứng và sửa ở phiên bản evaluator riêng.
- Checkpoint lưu RNG trước evaluation; evaluation tiếp tục dùng RNG để chọn subset và sampling. Cần bảo đảm resume tại mốc evaluation không làm thay đổi diễn tiến chạy liên tục.

Giữ kết quả và evaluator legacy để đối chiếu; đặt tên phiên bản cho evaluator sửa lỗi. Không ghi đè hoặc trộn số liệu giữa các phiên bản.

Checkpoint hiện có có thể evaluate lại. Tuy nhiên, việc thay đổi validation và cách tổng hợp NLL có thể thay đổi checkpoint tốt nhất; không tự coi best cũ là best theo giao thức mới. Nếu không có đủ checkpoint để chọn lại chính xác, ghi hạn chế hoặc thực hiện run mới sau khi được yêu cầu.

### Bước 2: Làm MDN ổn định khi training

Tạo variant riêng có sàn sigma nhỏ và giới hạn rho tránh covariance gần suy biến. Giá trị cụ thể phải được chọn có căn cứ về đơn vị dữ liệu, độ ổn định và ảnh hưởng tới phân phối; không mặc định dùng sàn trên 1 mét như công thức paper.

Áp dụng cùng parameterization mới cho baseline và attention để so công bằng. Giữ nguyên các run legacy làm mốc tham chiếu. Không áp dụng decoder mới vào checkpoint train bằng công thức cũ.

Mục tiêu là hoàn thành training và giữ NLL/covariance hợp lệ. Chưa có bằng chứng variant ổn định sẽ cải thiện accuracy hoặc calibration; đây là giả thuyết cần kiểm nghiệm.

Trước run dài phải review logging, checkpoint và fixed-sample capture; thực hiện smoke test ngắn theo yêu cầu được cho phép và xác nhận artifact đầy đủ. Không chạy full training khi chưa có yêu cầu rõ ràng.

### Bước 3: Ablation để xác định đóng góp của attention

| Mô hình | Vai trò | Câu hỏi |
|---|---|---|
| LSTM–MDN baseline M3 | Mốc tham chiếu | Kiến trúc gốc đạt kết quả nào? |
| LSTM + horizon embedding + MLP, không attention | Đối chứng decoder | Decoder mới có đủ tạo cải thiện không? |
| LSTM + temporal attention + cùng MLP | Variant attention | Attention đóng góp thêm bao nhiêu? |

Giữ cùng dữ liệu, split, input 32 bước, output 48 bước, K=3, loss, optimizer, lịch learning rate và ngân sách training. Dùng cùng parameterization trong từng nhóm so sánh.

Đối chứng không attention nên giữ future queries/horizon embedding, LayerNorm và MLP tương ứng; cách thay context phải được mô tả rõ. Cân bằng số tham số gần nhau nếu thực tế cho phép và báo số tham số chính xác, vì matching head không tự bảo đảm matching capacity.

Attention hiện tại thay nhiều yếu tố cùng lúc: temporal attention, future queries, LayerNorm, MLP và chia sẻ head giữa horizon. Chưa thể quy toàn bộ cải thiện cho attention nếu thiếu đối chứng này.

### Bước 4: Phân tích kết quả và trực quan hóa

- Báo NLL, minADE/minFDE theo giao thức repository, Ravg, Rmin, S68, S95, số tham số và thời gian suy luận.
- Ghi rõ ADE của evaluator repo dùng sáu mốc 0,8–4,8 giây và sắp sample theo mật độ riêng tại từng timestep. Không mô tả đây là 20 quỹ đạo có liên kết theo thời gian.
- Nếu đánh giá cả 48 bước hoặc sampling quỹ đạo khác, báo riêng dưới tên giao thức bổ sung.
- Phân tích nhóm đứng yên, đi thẳng và rẽ; không dùng nhãn chuyển động tương lai làm input của baseline.
- Xem từng forecast horizon để hiểu vì sao FDE cải thiện nhưng S68 có thể tăng.
- Dùng cùng fixed samples qua các checkpoint và cùng giới hạn trục.
- Có case tốt, khó/bất định và thất bại, với tiêu chí chọn rõ ràng.
- Đánh giá sharpness cùng reliability; vùng nhỏ hơn không tự động tốt hơn nếu mô hình quá tự tin.

Nếu đủ tài nguyên, chạy ba seed cho các mô hình chính và báo trung bình, độ phân tán. Nếu chỉ một seed, giới hạn kết luận ở thí nghiệm đó; không khẳng định cải thiện bền vững hoặc có ý nghĩa thống kê.

## 15. Bằng chứng hiện có và giới hạn

Kết quả legacy đã lưu trong repository:

| Chỉ số test | Baseline M3 | Attention-MDN |
|---|---:|---:|
| NLL mỗi vị trí, thấp hơn tốt hơn | -1,092 | -1,178 |
| minADE20 (m) | 0,464 | 0,446 |
| minFDE20 (m) | 0,607 | 0,556 |
| Ravg (%) | 97,47 | 97,88 |
| Rmin (%) | 94,06 | 95,52 |
| S68 theo evaluator local (m²/s) | 1,358 | 1,968 |
| S95 theo evaluator local (m²/s) | 6,061 | 5,971 |
| Số tham số | 8.224 | 7.890 |

Nguồn:

- `results/comparisons/imptc_m1_vs_m3/comparison.json`.
- `results/trained_models/attention_mdn/imptc/attention_peds_imptc/runs/imptc_attention_seed2024/testing/evaluation.json`.
- `attention_mdn/RESULTS.md` và run manifest tương ứng.

Attention có kết quả tốt hơn ở nhiều chỉ số trong run này nhưng S68 rộng hơn, suy luận chậm hơn và training legacy chưa hoàn tất. Chênh lệch S95 nhỏ, không nên diễn giải mạnh. Reliability dùng evaluator hiện có còn vấn đề bin; số liệu sau sửa cần được báo riêng.

## 16. Phạm vi và phân công đề xuất

Ưu tiên hoàn thành pipeline, variant ổn định, ablation attention và phân tích kết quả. Chưa ưu tiên thêm Transformer, diffusion hoặc social attention vì sẽ mở rộng dữ liệu, kiến trúc và số thí nghiệm.

Nếu phần chính đã hoàn tất và còn thời gian, có thể cân nhắc dự báo residual quanh chuyển động vận tốc không đổi. Đây là một thí nghiệm riêng, không thay baseline hoặc gộp đồng thời với ablation attention.

Phân công gợi ý cho ba thành viên:

| Thành viên | Phần chính | Sản phẩm |
|---|---|---|
| 1 | GMM, MDN, NLL và parameterization | Giải thích công thức, tensor shape, ổn định số |
| 2 | LSTM, temporal attention và ablation | Kiến trúc, cấu hình so sánh, quản lý run |
| 3 | Evaluation và visualization | Bảng metric, calibration, sharpness, case study và demo |

Mỗi người cần hiểu pipeline tổng thể. Chốt protocol và review artifact cùng nhau trước full run.

## 17. Tiêu chí hoàn thành đồ án

- Giải thích đúng vai trò LSTM, MDN và GMM theo từng timestep.
- Phân biệt baseline implementation với mô tả paper và các variant local.
- So sánh mô hình trên cùng giao thức, checkpoint chọn bằng validation.
- Có artifact và hình trực quan tái sử dụng được.
- Có ablation hoặc nêu rõ giới hạn nếu chưa thực hiện được.
- Trình bày trung thực đánh đổi và failure cases.
- Không khẳng định attention là nguyên nhân cải thiện duy nhất khi chưa có đối chứng.
