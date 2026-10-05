# Gated LSTM–MDN: học số component hoạt động

Variant độc lập, chỉ import `base_mdn` để đọc LSTM, decoder legacy, data loader, evaluator và tracker. Không sửa/copy đè baseline. Không dùng stable_mdn hay attention_mdn làm backbone.

## Phương pháp

Bắt đầu với `K_max=8`. Giữ LSTM hidden8, input32×4, forecast48×2, MDN layout mu_xK, mu_yK, log_sigma_xK, log_sigma_yK, raw_rhoK, logitsK. Legacy covariance giữ nguyên sigma=exp(raw), rho=tanh(raw).

Component0 được bảo vệ luôn có gate1, tránh mixture rỗng. Bảy component còn lại có gate hard-concrete **dùng chung cho mọi mẫu/timestep**. Đây là learned global subset, không phải K thay đổi theo từng pedestrian. Component index cũng không được coi là mode quỹ đạo chung.

Với u~Uniform(0,1), beta=2/3, gamma=-0.1, zeta=1.1:

```text
s = sigmoid((log(u) - log(1-u) + log_alpha) / beta)
z = clamp(s * (zeta-gamma) + gamma, 0, 1)
P(z>0) = sigmoid(log_alpha - beta*log(-gamma/zeta))
E[K_active] = 1 + sum P(z_k>0), k=1..K_max-1
pi_k = softmax(a_k + log(z_k))
objective = marginal_NLL + lambda_l0 * E[K_active]
```

Train lấy một vector gate ngẫu nhiên mỗi batch; vector đó dùng chung batch/horizon. Eval/export dùng gate xác định `clamp(sigmoid(log_alpha)*(zeta-gamma)+gamma,0,1)`. Eval gate không phải xác suất mở và expected K không phải số nguyên K đã prune. Hard-concrete có mass ở0/1; không cần dùng ngưỡng tùy ý trên pi để quyết định tắt. Gate0 được bảo vệ là lựa chọn thích nghi cho mixture trong project này, không phải yêu cầu từ paper.

Gate bằng0 được mã hóa bằng logit sentinel hữu hạn -1e9, cho pi bằng0 trong float32 mà vẫn lưu raw output hữu hạn. Gaussian inactive vẫn được tính trong model chưa export; gating không tự sửa covariance không hợp lệ của decoder legacy. Đây **không** phải hướng sửa sigma đang tạm dừng.

Nguồn cơ chế gate: [Louizos, Welling, Kingma — Learning Sparse Neural Networks through L0 Regularization, ICLR2018](https://arxiv.org/abs/1712.01312). Paper áp dụng sparsity cho network; việc gate component GMM ở đây là adaptation, không phải thuật toán mới hoàn toàn hay bằng chứng hiệu quả sẵn có trên IMPTC.

## Cấu hình và tính công bằng

- Giữ Adam, LR1e-3 với LinearLR tới1e-7/2500epochs, train reduction0.5, batch4096, preprocessing/coordinates và official evaluator legacy như base config hiện tại.
- K_max8 thay cho K3; thêm7 gate parameters và penalty. Đây là thay đổi phương pháp đã được yêu cầu.
- `lambda_l0=0.01`, probability mở ban đầu0.95 và K_max8 là lựa chọn thử nghiệm trước run, **chưa tối ưu**. Một run không đảm bảo gate sẽ đóng hay metric tốt hơn. Gate bị đóng vẫn có thể mở lại khi train; không xóa trọng số giữa run.
- Full deterministic validation (`eval_data_reduction=1.0`) thay cho random50% để best checkpoint có điểm so sánh ổn định. Đây là khác biệt protocol selection so với run base cũ, phải báo rõ. Trong biến thể này dùng sample-weighted epoch NLL, không trung bình đều batch lớn/nhỏ. Công thức batch marginal NLL vẫn là tổng negative log_prob chia B×48.
- Best được chọn bằng **unpenalized validation NLL** của deterministic gates; không tự lấy sparse checkpoint dù NLL kém. Objective/penalty được ghi riêng. Có thể best giữ K_max; không tuyên bố đã tìm K tối ưu toàn cục.
- So sánh có ý nghĩa cần cả base K3 và đối chứng ungated K8 với cùng protocol; nếu không có K8 đối chứng, không thể tách ảnh hưởng thêm capacity và sparsity. Chưa train đối chứng mới.
- Không dùng test chọn lambda/K_max. L0 component gating không đảm bảo xử lý outlier S68 hay ổn định covariance legacy. Underfitting/coverage giảm và component chết sớm là các khả năng phải kiểm tra.

## Artifact

Output ở `results/trained_models/gated_mdn/imptc/<config>/runs/<run>/`:

- history.csv: train/validation **NLL**, objective, penalty, LR, sample count, expected K và deterministic active K.
- gate_history.jsonl: gates/probabilities/active_indices từng epoch.
- best, last, periodic và final checkpoints gồm log_alpha, optimizer, scheduler, RNG, loader ordering, config và gate report.
- fixed_samples: giữ 8 ID/X/y của manifest baseline; NPZ gồm raw/decoded pi/mu/sigma/rho/covariance và gate state.
- metrics: evaluator legacy định kỳ; smoke ghi split `fixed_validation_smoke`, không phải full validation metrics.

NLL stochastic lúc train và deterministic lúc validation khác cách lấy gate; không diễn giải mọi khoảng cách giữa chúng là overfitting. E[K] là expectation trên hard-concrete, deterministic K tính số gate eval dương.

## Chạy và kiểm tra

```bash
.venv/bin/python -m unittest discover -s gated_mdn/tests -v
.venv/bin/python gated_mdn/train.py --smoke --run-id <smoke_run_moi>
```

Smoke3 epochs, 189train samples/epoch, full19.148 validation NLL. Official metric branches kiểm tra trên8 fixed validation samples, MC64, grid1m; không dùng để kết luận chất lượng model.

Full run K_max8 đã được duyệt và khởi động trong tmux `gated_k8`; chi tiết ở [reports/RUN_K8.md](reports/RUN_K8.md). Lệnh để tạo một full run mới:

```bash
.venv/bin/python gated_mdn/train.py --full --run-id <run_moi>
```

Resume chỉ từ checkpoint epoch cuối đã hoàn tất của cùng run/config để tránh CSV/metrics/best bị rewind:

```bash
.venv/bin/python gated_mdn/train.py --full --run-id <run> --resume <run>/checkpoints/last.pt
```

Test best checkpoint bằng deterministic gates; full official test chỉ dùng sau khi chốt experiment:

```bash
.venv/bin/python gated_mdn/evaluate.py --run-id <run> --limit 16
.venv/bin/python gated_mdn/evaluate.py --run-id <run> --official
```

Export bỏ thực sự các hàng head inactive và hấp thụ log-gate của component sống vào bias logits. LSTM giữ nguyên; distribution eval tương đương trong sai số float. Export chỉ phục vụ inference, không resume variant từ compact checkpoint.

```bash
.venv/bin/python gated_mdn/export.py --checkpoint <run>/checkpoints/best.pt --output <duong_dan_moi>.pt
```

Training model vẫn có đầy đủ K_max head rows; chỉ sau export mới có thể nói số tham số head giảm. Smoke K_max8 không kết luận K tốt nhất; xem reports/SMOKE.md. Smoke K_max5 trước đây được lưu riêng trong reports/SMOKE_K5.md.

Tài liệu giảng giải tiếng Việt với10 hình: [Model học K như thế nào?](docs/GIAI_THICH_HOC_K.md).


## Full run K_max=10

Đã thêm cấu hình `gated_k10_peds_imptc.json`, chỉ đổi K_max8→10; full2500epoch từ đầu với lambda0.01 và thư mục độc lập. Smoke và artifact review đạt. Xem [RUN_K10.md](reports/RUN_K10.md); theo dõi bằng `tmux attach -t gated_k10`. Run K8 và Bayesian vẫn được giữ nguyên.
