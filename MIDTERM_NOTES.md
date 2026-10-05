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

## Phân tích sau test ba model — 2026-10-03

Báo cáo và bộ hình: [ANALYSIS.md](residual_mdn/reports/analysis/ANALYSIS.md). Script tái tạo ở `residual_mdn/analysis/`.

- Residual cải thiện test NLL/ADE/FDE so với baseline stable ở một seed; Rmin và sharpness scalar kém hơn. Attention tốt nhất về NLL/displacement trong ba model nhưng không thắng tất cả metric uncertainty.
- Đã đối chiếu chính xác X/y/sample_id của 8 mẫu validation giữa ba model, vẽ mean/CV/ellipse Gaussian và cùng mẫu qua checkpoint. Sai số mixture mean trên hình là diagnostic, không thay minADE20 của repo.
- Inference toàn bộ validation/test xác nhận có outlier covariance. Residual median total trace test thấp hơn baseline nhưng max lớn hơn nhiều. Chẩn đoán grid trên test index 33794 cho Area68 @0.8s khoảng 530 m² so với 0.109 m² của baseline; chưa phân rã đầy đủ đóng góp vào sharpness official.
- Scalar sharpness local trung bình 101 phân vị, có cả max; đường vẽ là median. Grid [-18,18]² giới hạn diện tích ở 1296 m². Cần giải thích normalization/đơn vị theo code, không mặc nhiên coi scalar là median hay toàn bộ công thức của paper.
- Hướng tiếp theo: audit per-sample confidence area nếu cần kết luận uncertainty chặt chẽ, rồi cân nhắc covariance có giới hạn/regularization chọn bằng train/validation. Không chọn ngưỡng theo test. Không có training mới trong bước này.

### Audit confidence area đầy đủ — 2026-10-03

[AREA_AUDIT.md](residual_mdn/reports/analysis/AREA_AUDIT.md) và `results/comparisons/confidence_area_audit/full/` lưu diện tích/cờ chạm biên/ID cho cả ba model trên toàn validation và test. Audit tái hiện scalar sharpness test cũ trong sai số làm tròn, không đổi metric.

Residual: Q100 chiếm 80.40% S68 và 52.48% S95 trên test. Delta S68 so với baseline +2.223124 = Q100 +2.247732 và Q0–Q99 −0.024608. Delta S95 +2.742772 = Q100 +3.358396 và Q0–Q99 −0.615624. Vì vậy, phần tăng scalar sharpness tập trung ở cực đại; không kết luận residual dự báo rộng hơn trên mọi mẫu. Tại 4.8s, mean/median Area68 residual đều thấp hơn baseline. Rmin vẫn giảm nên chưa có cải thiện uncertainty toàn diện. Hướng tiếp theo là audit input/raw log-sigma của outlier validation rồi cân nhắc kiểm soát covariance, lựa chọn bằng train/validation; chưa chạy training mới.

### Truy nguyên outlier validation — 2026-10-03

Báo cáo: [validation_outliers/REPORT.md](residual_mdn/reports/analysis/validation_outliers/REPORT.md). Chọn hợp top2 diện tích theo model/level/horizon, được 18 mẫu validation; lưu X/y/raw/decoded parameters và ID. Không chọn theo test, không sửa model/dataset.

Velocity toàn validation khớp diff(position)/0.1 tới khoảng 4.44e-16. Một số input đổi tốc độ mạnh ở đuôi train/validation, nhưng chưa có căn cứ kết luận dữ liệu hỏng. Attention index13232 có speed≈1.38 m/s (khoảng phân vị62 train), trong khi component pi≈0.997 có sigma≈57.27/29.44 m: lỗi uncertainty không chỉ xuất hiện vì tốc độ cao. Residual index5469 có component sigma≈204m nhưng pi≈0.0035; component pi≈0.9947 có sigma≈1.97/3.85m. Không gán sharpness chỉ cho max sigma. Phép cộng CV đã được xác nhận không trực tiếp đổi sigma/rho/pi; thay đổi quá trình học gián tiếp chưa được chứng minh nhân quả. Hướng ablation kiểm soát covariance vẫn cần đánh giá cả reliability, không chỉ thu nhỏ ellipse. Chưa chạy training mới.

### S68 của attention gốc so với base gốc — audit replay 2026-10-03

Báo cáo: [LEGACY_S68_ANALYSIS.md](attention_mdn/reports/LEGACY_S68_ANALYSIS.md). Audit best base1961/attention1740 trên56.694 test×6horizon, decoder legacy, replay đúng thứ tự MC của evaluator; S68/S95 khớp score cũ trong sai số làm tròn. Attention chỉ rộng hơn ở27.85% cặp mẫu–horizon, nhưng mean/P99 cao hơn; median ở4.8s là5.032m² so với5.480m² baseline. Delta S68 +0.610343 gồm Q100 +0.512407 và Q0–Q99 +0.097936: 83.95% delta từ phần maximum. Mẫu4881 @4.8s có A68 replay794.739m² attention vs11.297m² base; component pi0.931 có sigma11.678/12.975m. Không phải Gaussian rộng nhưng pi gần0. NLL/displacement/reliability tốt hơn không đảm bảo sharpness cùng tốt vì chúng đo các khía cạnh khác và tổng hợp khác. Chưa chứng minh attention layer gây covariance rộng; không chọn hyperparameter bằng test, không bỏ metric/cực trị. Chưa sửa model hay train thêm.

## Variant học component hoạt động: gated_mdn — 2026-10-03

Đã triển khai riêng tại [gated_mdn/README.md](gated_mdn/README.md), không sửa source base_mdn (đối chiếu SHA256). LSTM/MDN decoder legacy, K_max5 với4 global hard-concrete gates và1 component luôn giữ; objective marginal NLL +0.01×expected active K. Cơ chế sparsity dựa trên Louizos et al. ICLR2018, adaptation gate component GMM của project; không tuyên bố phương pháp hoàn toàn mới. Gate là global/shared, không phải coherent trajectory mode hay K đổi theo mẫu. Eval deterministic và export loại head rows inactive; chưa export thì model vẫn có head K_max đầy đủ.

Các lựa chọn K_max/lambda chưa tối ưu, có thể toàn bộ gates vẫn hoạt động. Best chọn unpenalized validation NLL, không ép chọn sparse checkpoint kém chất lượng. Giữ optimizer/LR/preprocessing/covariance baseline; dùng full validation và sample-weighted epoch NLL cho variant, đây là khác biệt protocol selection so với base cũ cần báo rõ. Full metrics dùng evaluator legacy. NLL/objective/penalty, expected K/deterministic K và gate probabilities được ghi riêng, checkpoint có gate/optimizer/RNG/loader state;8 fixed sample IDs giữ nguyên baseline.

[SMOKE.md](gated_mdn/reports/SMOKE.md):4 unit tests qua, gồm compact distribution equivalence và stochastic optimizer/RNG continuation. Smoke3 epochs,189train/epoch và full19.148 validation NLL;7 official metric branches kiểm tra trên8 fixed validation samples, MC64/grid1m, không phải full metrics. Limited test16/export smoke qua. Gates cập nhật nhưng deterministic K vẫn5; chưa có kết luận tìm được K tối ưu hay cải thiện metric. Chưa full training. Để tách capacity và sparsity khi đánh giá cần đối chứng ungated K5 cùng protocol ngoài baseK3; chưa train đối chứng này. Không dùng test để lựa chọn penalty/K_max.

### Đổi trần component theo yêu cầu — K_max8

Cấu hình hiện tại của gated_mdn dùng K_max8 (7 learned gates +1 protected component), giữ lambda0.01 và các lựa chọn khác. K_max5 trước đó chỉ là kiểm tra cơ chế, không phải kết quả chọn K; smoke cũ được giữ và báo cáo lưu SMOKE_K5.md/json. Smoke K8 mới `gated_k8_smoke_seed2024` đã qua3 epochs; full validation NLL, fixed8 metric smoke, limited test16 và compact export qua, base source hashes không đổi. Unit test mới xác nhận head8 có thể export đúng5 component và giữ distribution tương đương; đây là test cơ chế với gate gán thủ công, không phải đã học ra K5. Tổng5 unit tests qua. Smoke hiện vẫn active8; expectedK≈7.649 khác số component nguyên. Chưa full training. K5 sau một full run (nếu xảy ra) là learned active count dưới K_max8/lambda đã chọn, không chứng minh K5 tối ưu toàn cục; nếu active8 thì chưa loại trừ tác động của ceiling/penalty hoặc hội tụ. Đối chứng capacity tương ứng là ungatedK8 với cùng protocol.

### Full gated K8 đã được duyệt và khởi động

Run `gated_k8_seed2024_tmux`, tmux `gated_k8`; cấu hình2500epochs đã qua smoke K8. Đã review epoch1 checkpoint, gate log, history.csv và8 fixed artifacts, base source hashes không đổi. Chi tiết theo dõi ở [RUN_K8.md](gated_mdn/reports/RUN_K8.md). Chưa có kết quả full training/test; không tự queue test hay đối chứng ungatedK8.

### Tài liệu giải thích learned K với10 hình

[GIAI_THICH_HOC_K.md](gated_mdn/docs/GIAI_THICH_HOC_K.md) giải thích gate global vs pi local, hard-concrete/expected L0, train stochastic vs eval deterministic, đếmK và compact export. Hình01–08/10 là minh họa, hình09 từ snapshot run thật đếnepoch209; ởsnapshot đó activeK8, chưa pruning. Snapshot không phải kết quả cuối/live. Tài liệu nêu best chọn unpenalized validation NLL, gate/logits có thể bù nhau và learned active K không phải K tối ưu toàn cục. Linkảnh và source đã kiểm tra; training không bị gián đoạn.

## Biến thể Bayesian trên trọng số mixture

Đã triển khai thử `bayesian_mdn/` độc lập: global Beta stick-breaking + bounded conditional mixture logits; LSTM/covariance legacy giữ nguyên. Smoke 3 epoch, chưa full train. Xem `bayesian_mdn/README.md` và `bayesian_mdn/reports/SMOKE.json`. K theo 99% global mass đã bằng 7 ngay tại prior, không được diễn giải là học thành công K=7. Inference plug-in mean sticks chưa phải posterior predictive; cần kiểm chứng trước tuyên bố cải thiện calibration hoặc tối ưu K.


### Bayesian posterior predictive và diagnostics K (smoke bổ sung)

Đã hoàn tất smoke `bayesian_k8_predictive_qmc_smoke_seed2024`: epoch0 snapshot, full-validation conditional usage/responsibility, posterior predictive qua64scrambled Sobol/Beta inverse CDF, best theo predictive NLL và comparison16/64/256/seeds. 6tests pass, artifactreview pass, prediction tái hiện bằng bank đã lưu; modelstate cuối bằng smoke cũ. K99% vẫn7 từ epoch0, không kết luận học giảm K. Báo cáo4hình: `bayesian_mdn/reports/PREDICTIVE_SMOKE.md`. Chưa full train.


Full Bayesian run đã được khởi chạy sau review đạt: tmux `bayesian_k8`, run `bayesian_k8_qmc_seed2024_tmux`, 2500epoch. Đã xác nhận epoch1 checkpoint, usage epoch0/1 và tiến trình train hoạt động; chưa có kết quả cuối.


Theo yêu cầu người dùng đã dừng Bayesian K8 tại last completed epoch749 (đang metrics750), giữ artifact và manifest stopped_by_user. Sau7tests/smoke3epoch/artifactreview đạt, khởi chạy K_max10 độc lập, full2500epoch từ đầu: tmux `bayesian_k10`, run `bayesian_k10_qmc_seed2024_tmux`, config `bayesian_k10_peds_imptc.json`. Không ép K cuối=8; giữ prior/ngưỡng/hidden/covariance cũ. Chi tiết: `bayesian_mdn/reports/K10_RUN.md`.


Theo yêu cầu đã chạy thêm gated_mdn K_max10, chỉ đổi K so với gated K8; lambda0.01/legacy covariance/LSTMhidden8 giữ nguyên. 5tests và smoke3epoch/fullvalidation/artifact/export review đạt trước full. Full2500epoch mới: tmux `gated_k10`, run `gated_k10_seed2024_tmux`, config `gated_k10_peds_imptc.json`, result folder độc lập. Chi tiết `gated_mdn/reports/RUN_K10.md`. Không ép hoặc khẳng định sẽ học K=8.


## Variant học K: growing_mdn — 2026-10-04

Thiết kế: `docs/superpowers/specs/2026-10-04-growing-mdn-design.md`; kế hoạch: `docs/superpowers/plans/2026-10-04-growing-mdn.md`; hướng dẫn: `growing_mdn/README.md`.

Đã triển khai `growing_mdn/` độc lập (không sửa `base_mdn/`): tăng dần head MDN K 1 -> K_max, ánh xạ lại trạng thái Adam, chấm điểm và quy tắc chọn K trên validation, tracker artifact, trainer theo phase có hook dừng/resume, `select_k.py`, `evaluate.py` (`--official` chỉ cho K trong `selection.json`, một lần mỗi run, là nơi duy nhất nạp test split; `--limit` chạy trên validation), `review_smoke.py`, `base_hashes.py`.

Đã kiểm chứng: 37 unit test; smoke K 1->3, 4 epoch, subset train nhỏ, 8 mẫu validation cố định (metric chỉ từ 8 mẫu, không phải validation đầy đủ); bốn kiểm tra resume tái hiện lần chạy liền mạch với max |validation NLL diff| 0.0 trên CPU; hash `base_mdn/` không đổi (ba file xóa sẵn có trong working tree không do biến thể này). Delta NLL tại split trên dữ liệu thật -0.0090 (1->2) và -0.0055 (2->3) trên mô hình chưa huấn luyện, phụ thuộc dữ liệu.

Khác baseline cần nêu: LinearLR khởi động lại mỗi phase; `metric_every=0` (metric chính thức chỉ ở cuối mỗi phase); K được chọn một lần sau mọi phase bởi `select_k.py` (tolerance đóng băng trong `resolved_config.json`); chọn K dùng validation NLL đầy đủ (baseline reduction 0.5); trạng thái tốt nhất theo phase dùng cho split/metric; `final.pt` là trạng thái tốt nhất của K_max; mô hình lồng nhau thấy nhiều epoch hơn trên trọng số chung nên so K=3/K=8 theo tổng compute; cần nhiều seed; "component k" là chỉ số head, không phải mode vật lý nhất quán.

Tolerance `epsilon_nll=0.02`, `ravg_tolerance_pp=0.5`, `sharpness_tolerance_ratio=0.10` mới là đề xuất, chờ người dùng xác nhận. Chưa chạy full training (2600 epoch, K 1 -> 12) và chưa có kết quả nào về K; cần người dùng duyệt rõ ràng trước khi chạy. Giới hạn đã biết xem trong `growing_mdn/README.md`.

### Cập nhật growing_mdn (2026-10-04, sau khi chạy thật)
Run v1 `growing_k12_seed2024` bị dừng: tách với độ lệch tâm tuyệt đối 0.05 m làm validation NLL từ −0.0672 lên 133.68 (σ của model K = 1 đã train chỉ cỡ 1e-3 m ở các bước đầu). Revision 1 (spec): độ lệch tâm tỉ lệ với σ trung vị của thành phần nguồn, đặt lại LR 1e-4 ở các phase sau, chốt dừng nếu |ΔNLL| > 0.01. Kiểm chứng trên trọng số thật của v1: tách tương đối đổi NLL 0.00026 (K 1→2) và 0.00002 (K 2→3). Run v2 `growing_k12_v2_seed2024` đã xong: cả 11 lần tách qua chốt (|ΔNLL| lớn nhất 0.0016); `select_k` chọn K = 11 (K tốt nhất theo NLL là 12) trên validation. Giới hạn: NLL chưa bão hòa đến trần K = 12; mỗi K chỉ train 200 epoch nên không so trực tiếp với baseline; đến epoch 800 v1 còn thấp hơn v2 (−0.817 so với −0.723) nên LR đặt lại thấp hơn và độ lệch theo σ bị lẫn với nhau. Đây là phân tích về K (chọn mô hình), không phải cải tiến chính. Chưa chạy test. Chi tiết: `docs/growing_mdn/GROWING_MDN.md`.

## Cải tiến chính: sparsemax_mdn — 2026-10-04
Thiết kế và giao thức công bố trước: `docs/superpowers/specs/2026-10-04-sparsemax-mdn-design.md`; tài liệu giải thích: `docs/sparsemax_mdn/SPARSEMAX_MDN.md`; code: `sparsemax_mdn/` (không sửa `base_mdn/`).

Ý tưởng: thay softmax bằng sparsemax cho trọng số π của MDN (K_max = 16), nên K(x,t) = số π > 0 thay đổi theo mẫu và theo bước, một lần train, loss NLL và mọi thứ khác như baseline. Không tuyên bố là phương pháp mới; trích dẫn Martins và Astudillo (2016) cần kiểm tra. Run `sparsemax_k16_seed2024` (2500 epoch, seed 2024) đã xong. Kết quả trên **validation, một seed**: K trung bình 3.98 (2 đến 9), không thành phần chết, trần 16 không bị chặn; NLL best −1.204 (baseline K = 3: −1.150, K = 8: −1.250); so với baseline (trung bình qua epoch 1250–2500) độ tin cậy tốt hơn K = 3 (Ravg 97.51 so với 96.85, Rmin 95.12 so với 92.54) và gần K = 8 (97.88, 95.43), minFDE nhỉnh hơn; **độ sắc nét chưa kết luận được**: S95 chính thức cao hơn ở 8/10 mốc epoch nhưng dao động rất mạnh, còn chẩn đoán bằng hàm của repo trên 6000 mẫu validation (`sparsemax_mdn/diagnose_sharpness.py`) cho thấy sparsemax không kém sắc nét hơn K = 3 ở giá trị điển hình, chỉ có vài mẫu dự đoán cực rộng (đuôi). Giả thuyết "K lớn hơn ở mẫu khó" không được ủng hộ (Spearman −0.09 và −0.25). Đối chiếu tiêu chí công bố trước: đạt một phần (độ sắc nét chưa kết luận), K thích nghi đạt, quan hệ với độ khó không đạt. Chưa chạy test; chưa so nhiều seed; sparsemax không xử lý mối lo quá khớp của thành phần hẹp. Có `sparsemax_mdn/evaluate.py` (đánh giá validation, chế độ trọng số chính xác, test chỉ một lần với `--confirm-test-once`).

### Kết quả test của sparsemax_mdn (một lần, 2026-10-04)
Test 56 694 mẫu, checkpoint best (epoch 1961), code đánh giá chính thức: sparsemax NLL −1.134, Ravg 97.98, Rmin 96.08, minADE 0.456, minFDE 0.584, S68 1.474, S95 6.103, ASAEE 0.2413; K trung bình 3.99, không thành phần chết. Baseline K = 3: −1.092, 97.47, 94.06, 0.464, 0.607, 1.358, 6.061, 0.2353. Baseline K = 8: −1.207, 98.49, 96.59, 0.452, 0.590, 0.902, 5.348, 0.2394. Kết luận (một seed): sparsemax tốt hơn K = 3 về NLL, độ tin cậy, ADE/FDE nhưng kém sắc nét hơn một chút; **không vượt K = 8** (kém NLL, Ravg, Rmin, S68, S95). Chưa thể gọi là cải tiến về độ chính xác; điểm có thể nêu là K thích nghi (khoảng 4 thành phần hoạt động). Điểm sharpness chính thức là trung bình 101 phân vị nên mẫu lớn nhất chiếm khoảng 1% trọng số (nhạy với đuôi); trung vị theo mẫu của sparsemax là S95 4.00, S68 0.55 trong khi điểm chính thức là 6.10 và 1.47. Số liệu từng mẫu của baseline trên test chưa tính. Tập test đã dùng, không dùng để chỉnh phương pháp nữa. Chi tiết: `docs/sparsemax_mdn/SPARSEMAX_MDN.md` mục 6.7.

Giải thích S68/S95 của sparsemax cao hơn K = 3 trên test (không phải sai công thức): điểm sharpness chính thức là trung bình 101 phân vị theo mẫu, mẫu rộng nhất ở mỗi mốc chiếm khoảng 1% điểm. Trên test, 61% điểm S68 và 30% điểm S95 của sparsemax đến từ phân vị 100 (một mẫu mỗi mốc, diện tích tới 985 và 1296 m², bằng toàn bộ lưới). Trên cùng 4000 mẫu validation với công thức chính thức, sparsemax sắc hơn K = 3 (S95 4.43 so với 4.81; S68 0.625 so với 0.687). Chưa có số liệu từng mẫu của K = 3 trên test nên chưa chứng minh trực tiếp ở test. Chi tiết: `docs/sparsemax_mdn/SPARSEMAX_MDN.md` mục 6.9.

Cập nhật (baseline K = 3 trên test, số liệu từng mẫu): metric chính thức của baseline tái tạo đúng (sai khác 0.0) so với `ablation.json`. Trên test, phần "thân" (phân vị 0–99) của sparsemax sắc hơn K = 3 (S68 0.569 so với 0.627; S95 4.26 so với 4.46), trung bình theo mẫu và phân vị 99 cũng thấp hơn, và K = 3 có nhiều mẫu rộng bất thường hơn (S95 trên 20: 258 mẫu so với 68). Chênh lệch điểm chính thức (S68 1.474 so với 1.358) đến hoàn toàn từ mẫu rộng nhất ở mỗi mốc (phân vị 100, chiếm 1% điểm): 0.905 so với 0.730. Chưa tính số liệu từng mẫu của K = 8.
