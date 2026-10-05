# Chuyển Bayesian từ K_max=8 sang K_max=10

Theo yêu cầu, đã dừng run K8 và khởi chạy run K10 từ đầu, với thư mục độc lập. Không ghi đè kết quả K8, không sửa base_mdn.

## Run K8 được giữ lại

- Run ID: `bayesian_k8_qmc_seed2024_tmux`.
- Thư mục: `results/trained_models/bayesian_mdn/imptc/bayesian_peds_imptc/runs/bayesian_k8_qmc_seed2024_tmux`.
- Last completed epoch: **749**. Dừng trong official validation metrics ở epoch750; không tính750 là epoch hoàn tất.
- Last/best/periodic checkpoints, log và usage history giữ lại. Manifest đánh dấu stopped_by_user; exit143 là dừng bằng SIGTERM, không phải hoàn tất2500epoch.

## Cấu hình mới

- File full: `bayesian_mdn/configs/imptc/bayesian_k10_peds_imptc.json`.
- File smoke: `bayesian_mdn/configs/imptc/smoke_bayesian_k10_peds_imptc.json`.
- Chỉ thay num_gaussians từ8 sang10 trong cấu hình, giữ hidden8, prior concentration1, threshold99%, local logit bound1, covariance legacy, optimizer/lịch LR và inference64QMC.
- Full2500epoch từ đầu; không load checkpoint K8 vào head K10.
- Bộ9Beta sticks cho10Gaussian. K_max10 là giới hạn, không phải10model.
- Log của trọng số có floor numerical float32.tiny để tránh log(0) nếu tích phần dư ở đầu ra dài bị underflow; không đổi prior/ngưỡng để ép K cuối.

## Smoke đã đạt

- 7unit tests, gồm K10 shape60, gradient và tình huống đuôi weight underflow.
- 3epoch IMPTC; full validation19148mẫu; metrics trên8fixed validation samples.
- initial/best/last/final/periodic checkpoints, RNG/optimizer/scheduler/loader state và6fixed prediction archives đầy đủ; bank posterior shape64×10.
- Limited test16mẫu inference thành công, không dùng chọn cấu hình.
- Hash baseline unchanged.

| Epoch | Validation predictive NLL | K global99% | K mean conditional99% |
|---|---:|---:|---:|
| 1 | 6.380237 | 7 | 7 |
| 2 | 6.352795 | 7 | 7 |
| 3 | 6.338752 | 7 | 7 |

**Initial K99% cũng bằng7** vì prior geometric. Smoke không đủ xác định K học được. Tăng K_max lên10 không bảo đảm K cuối bằng8; ghi/report K theo dữ liệu thực tế.

## Full run mới

- Tmux: `bayesian_k10`.
- Run ID: `bayesian_k10_qmc_seed2024_tmux`.
- Kết quả: `results/trained_models/bayesian_mdn/imptc/bayesian_k10_peds_imptc/runs/bayesian_k10_qmc_seed2024_tmux`.
- Log: `results/trained_models/bayesian_mdn/launches/bayesian_k10_qmc_seed2024_tmux.log`.

```bash
tmux attach -t bayesian_k10
```

Báo cáo máy đọc: [K10_SMOKE.json](K10_SMOKE.json).
