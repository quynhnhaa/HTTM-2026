# Partial Bayesian conditional LSTM–MDN

Biến thể nghiên cứu độc lập, import read-only encoder/decoder và evaluator từ `base_mdn`. Không sửa baseline. Đây là **Bayesian trên các biến trọng số toàn cục**, không phải toàn bộ trọng số LSTM Bayesian, và không phải triển khai nguyên bản thuật toán DP-GMM cho clustering.

## Mô hình xác suất được triển khai

- LSTM hidden=8, K_max=8, horizon=48; mean và covariance giữ decoder legacy (exp sigma, tanh rho).
- Bảy biến toàn cục độc lập: v_k ~ Beta(1, alpha), alpha=1. Thành phần cuối nhận phần dư. Đây là truncated stick-breaking, sum w_k=1.
- Posterior variational: q(v_k)=Beta(a_k,b_k); a,b dương qua softplus. Khởi tạo q=prior.
- w_k=v_k prod_{j<k}(1-v_j); w_8=prod_{j<8}(1-v_j).
- Trọng số có điều kiện: pi_k(X,h,v) proportional to w_k(v) exp(tanh(logit_k(X,h))). Giới hạn local logit ở [-1,1] là thay đổi so với baseline; ngăn phần local bù trọng số toàn cục một cách không giới hạn. Nó cũng có thể giảm khả năng mô tả các tình huống hiếm.
- LSTM và Gaussian head có tham số điểm theta. Likelihood là product của marginal mixtures từng sample/timestep, không phải joint mixture cho toàn quỹ đạo.
- Một mẫu v reparameterized cho mỗi minibatch, dùng chung các sample và timestep: ước lượng Monte Carlo của expected log likelihood.

Objective chuẩn hóa:

```text
L = mean_{B*48}[-log p(y_nh | X_n, v, theta)] + KL(q(v)||p(v)) / (N_ref*48)
```

N_ref là số trajectory được lấy mỗi epoch theo train_data_reduction (smoke 189, cấu hình full 94797), được lưu ở history. Do mỗi epoch chọn ngẫu nhiên từ pool lớn hơn, đây là lựa chọn cỡ dữ liệu tham chiếu cho objective; không tuyên bố ELBO của toàn bộ pool chưa giảm. Không thêm hệ số KL tùy tiện để ép giảm K. KL có thể rất nhỏ sau chuẩn hóa, không bảo đảm sparsity.

## Inference và cách đọc K

Inference và lựa chọn best checkpoint hiện dùng **posterior predictive marginal**, với 64 điểm tích phân scrambled Sobol (randomized QMC) và Beta inverse CDF. Mean/covariance chỉ phụ thuộc X, không phụ thuộc v, nên tích phân có thể viết:

```text
pi_predictive(X,h) ≈ mean_s softmax(log w(v_s) + bounded_logits(X,h))
p_predictive(y|X,h) = sum_k pi_predictive_k(X,h) N_k(y|X,h)
```

Phải trung bình **pi sau chuẩn hóa**, không trung bình logits hoặc dùng mean sticks thay thế. Mixture marginal vẫn có 8 Gaussian, không biến thành 64×8. Đây là xấp xỉ tích phân hữu hạn, không phải tích phân chính xác. Các điểm Sobol không iid. Inverse Beta CDF dùng SciPy float64 trên CPU cho inference, rồi chuyển về dtype/device model; training vẫn dùng Beta.rsample và gradient bình thường. Mean/covariance legacy không đổi.

Seed và các điểm uniform gốc được giữ chung qua checkpoint, và bộ 16 điểm là prefix của 64/256. Điều này giúp so sánh posterior thay đổi liên tục mà không bị đổi bộ mẫu do rejection sampling. RNG inference được cô lập; snapshot lưu **posterior_weight_bank** để tái hiện prediction. Khi cần gọi model trực tiếp, dùng `with model.inference(draws=64,seed=2024): ...`; `model.eval()` riêng vẫn trả plug-in như trước, và metadata phân biệt hai mode.

Full validation ghi cả posterior-predictive NLL lẫn plug-in NLL. Train NLL là một mẫu stochastic ước lượng expected negative log likelihood; nó không cùng phép tính với posterior predictive NLL. Không cộng KL vào validation predictive NLL rồi gọi đó là ELBO. `validation_objective` cũ đã bỏ để tránh diễn giải sai.

`effective_k_global_mass` là số thành phần có mean global weights lớn nhất đủ đạt 99% tổng mass. Đây là diagnostic; phụ thuộc ngưỡng và prior, chưa phải K tối ưu hoặc số thành phần đã bị loại. Báo cáo này không đồng nhất với K theo conditional pi của từng quỹ đạo. Không tự compact hoặc loại thành phần: forward vẫn tính cả 8.

Ngay tại prior alpha=1, mean weights = [0.5,0.25,0.125,0.0625,0.03125,0.015625,0.0078125,0.0078125]. Theo 99% mass, K diagnostic đã bằng 7 **trước training**. Vì vậy K=7 sau smoke không chứng minh model học giảm K.

Không có nhãn cố định cho các modes từ baseline; stick-breaking áp dụng thứ tự thành phần, có thể tạo thiên lệch tối ưu. Prior concentration, ngưỡng mass và bound local vẫn là lựa chọn thiết kế. Cần đối chứng K=8 không Bayesian cùng protocol để đánh giá đóng góp riêng.

Full validation còn ghi mean pi và mean responsibility (membership khi biết ground truth), từng timestep và trung bình toàn split; histogram K theo từng sample/timestep ở ngưỡng95%/99%; K từ mean conditional pi. Chỉ mean pi không đủ chứng minh một component vô ích vì có thể phục vụ tình huống hiếm. Không sử dụng các diagnostics này để tự prune.

Epoch0 có checkpoint `initial.pt`, posterior weights, full-validation usage và fixed predictions. So sánh với epoch sau bằng biểu đồ thực tế trong [báo cáo predictive smoke](reports/PREDICTIVE_SMOKE.md).

## Đã kiểm tra

- 7 unit tests (bao gồm kiểm tra K10 và đuôi trọng số underflow): prior/KL/normalized sticks, gradient stochastic, tái lập state, predictive density so với trung bình các mixture thủ công, RNG/batch invariance/common integration points và diagnostics K.
- Smoke 3 epoch trên IMPTC, full validation 19148 trajectory; official metrics trên 8 fixed validation samples, MC=64 và mesh=1m chỉ để kiểm tra pipeline.
- Initial epoch0, periodic1/2/3, best/last/final; optimizer/scheduler/RNG/loader state; raw outputs và decoded pi,mu,sigma,rho,covariance cho fixed samples; posterior weights và validation usage JSONL; bank posterior được lưu cùng fixed predictions.
- Test giới hạn 16 mẫu chỉ kiểm tra inference, không dùng chọn cấu hình.
- So sánh budget16/64/256 và seeds2024/2025/2026 trên 8 fixed validation samples; không dùng test chọn budget.
- Smoke mới tái hiện nguyên trạng training state của smoke trước; logging/inference không đổi tiến trình học.
- Run K8 đã dừng theo yêu cầu ở last completed epoch749; artifact giữ lại. Full K_max10 đã khởi chạy sau smoke mới đạt. Chưa có kết quả cuối để so sánh.

```bash
.venv/bin/python -m unittest bayesian_mdn.test_model
.venv/bin/python -m bayesian_mdn.review_smoke
.venv/bin/python -m bayesian_mdn.plot_usage
.venv/bin/python -u bayesian_mdn/train.py --smoke --run-id YOUR_NEW_SMOKE_ID
# Full training chỉ chạy sau khi review artifact và được duyệt:
.venv/bin/python -u bayesian_mdn/train.py --full --run-id YOUR_NEW_FULL_ID
```

Best checkpoint theo unpenalized **posterior-predictive full validation NLL**, draws64/seed2024 được lưu trong config và checkpoint. Evaluator kiểm tra metadata; không âm thầm chuyển checkpoint inference cũ sang phương pháp mới. Full cấu hình 2500 epoch, Adam1e-3, LinearLR, batch4096; validation full khác baseline cũ chọn random50%. Official metric mỗi250epoch; periodic checkpoint mỗi100 và1/5/10. Resume chỉ latest completed epoch cùng config và run. Legacy covariance vẫn có nguy cơ số học; không tự thay covariance khi lỗi.

Nguồn lý thuyết: [Blei & Jordan (2006), Variational inference for Dirichlet process mixtures](https://www.cs.columbia.edu/~blei/papers/BleiJordan2006.pdf). Conditional modulation và diagnostic K ở đây là lựa chọn áp dụng của thí nghiệm, không phải thuật toán từ bài báo đó hoặc bảo đảm thống kê của DP mixture.


## Full run K8 đã dừng

```bash
tmux attach -t bayesian_k8
```

Run: `results/trained_models/bayesian_mdn/imptc/bayesian_peds_imptc/runs/bayesian_k8_qmc_seed2024_tmux`.
Log: `results/trained_models/bayesian_mdn/launches/bayesian_k8_qmc_seed2024_tmux.log`.
Launcher lưu mã kết thúc vào `.exit_code`; tmux hiển thị output trực tiếp và giữ pane sau khi kết thúc. Đã kiểm tra initial/epoch1 checkpoint, full-validation usage và inference metadata ngay sau launch. Không queue test hoặc một run khác tự động.


## Full run K_max=10 đang chạy

K8 giữ lại như run chưa hoàn tất. Cấu hình mới `bayesian_k10_peds_imptc.json` chỉ đổi K_max10; full2500epoch từ đầu, run và log độc lập. Xem [K10_RUN.md](reports/K10_RUN.md) và [K10_SMOKE.json](reports/K10_SMOKE.json).

```bash
tmux attach -t bayesian_k10
```

Kết quả: `results/trained_models/bayesian_mdn/imptc/bayesian_k10_peds_imptc/runs/bayesian_k10_qmc_seed2024_tmux`.
K_max10 không bảo đảm kết quả K=8; không điều chỉnh prior/threshold để ép số đó. Numerical floor khi log weight giúp tránh underflow của stick tail dài; không thay covariance legacy.
