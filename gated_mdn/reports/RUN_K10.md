# Full run gated MDN K_max=10

Đã khởi chạy theo yêu cầu, từ đầu và không ghi đè gated K8/Bayesian K10/base_mdn.

- Config: `gated_mdn/configs/imptc/gated_k10_peds_imptc.json`.
- Tmux: `gated_k10`.
- Run ID: `gated_k10_seed2024_tmux`.
- Results: `results/trained_models/gated_mdn/imptc/gated_k10_peds_imptc/runs/gated_k10_seed2024_tmux`.
- Log: `results/trained_models/gated_mdn/launches/gated_k10_seed2024_tmux.log`.
- Full2500epoch, seed2024, LSTM hidden8, K_max10, 9learnable hard-concrete gates và1protected component, legacy covariance, marginal NLL +0.01expected active K.
- Đã kiểm tra config chỉ đổi num_gaussians8→10 so với gated K8. Không đổi prior gate, optimizer/lịch LR hoặc mức phạt để ép kết quả K.
- Best theo unpenalized full validation NLL. Checkpoint1/5/10/mỗi100, best/last/final; metric mỗi250; 8fixed IDs như trước.
- Model có26377tham số; nếu compact cả10component thì plain head26368tham số. Không compact tự động trong training.

## Kiểm tra trước full

5unit tests pass; smoke3epoch với189train samples/epoch và đầy đủ19148validation NLL. Cả7metric branches trên8fixed validation samples (MC64/mesh1m), limited test16, compact export likelihood equivalence, gate parameter updates, checkpoint/RNG/optimizer/loader state và base source hashes đều đạt. Báo cáo [SMOKE_K10.json](SMOKE_K10.json).

Smoke còn10gate mở, expected K≈9.549. Không phải kết quả model hội tụ. `eval K` đếm gate>0; **không phải** K theo ngưỡng99% như Bayesian. Fullrun sẽ ghi K thực tế, không bảo đảm8hoặc giảm dưới10.

```bash
tmux attach -t gated_k10
```

Log hiện trực tiếp qua tee và vẫn lưu file; session tiếp tục khi ngắt SSH. Chưa tự queue fulltest hoặc một run khác.
