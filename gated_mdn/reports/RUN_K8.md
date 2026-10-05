# Full training K_max8

User đã duyệt full run. Run `gated_k8_seed2024_tmux` đang chạy trong tmux session `gated_k8`. Epoch1 checkpoint/log/fixed capture đã review: model có7 learned gate parameters, optimizer/RNG/loader ordering được lưu,8 sample IDs và decoded shape [8,48,8,2] khớp. Base source hashes không đổi.

- Config: `gated_mdn/configs/imptc/gated_peds_imptc.json`.
- 2500epochs, seed2024, K_max8, hidden8, legacy sigma/rho, lambda L0=0.01.
- Adam1e-3, LinearLR tới1e-7, batch4096, train subset94797/epoch.
- Full19148 validation NLL, best chọn unpenalized deterministic validation NLL.
- Gate states/objective/NLL/LR mỗi epoch; periodic checkpoints epoch1,5,10 và mỗi100; best/last/final;8 fixed samples.
- Full validation legacy metrics mỗi250epochs. Không tự queue test hoặc run đối chứng.
- Expected K là expectation; eval K là integer gates dương. Chưa kết luận K tối ưu hay metric tốt hơn.

Theo dõi:

```bash
tmux attach -t gated_k8
tail -f results/trained_models/gated_mdn/launches/gated_k8_seed2024_tmux.log
```

Artifacts: `results/trained_models/gated_mdn/imptc/gated_peds_imptc/runs/gated_k8_seed2024_tmux/`. Launcher ghi exit code ở `results/trained_models/gated_mdn/launches/gated_k8_seed2024_tmux.exit_code` khi kết thúc. Ngắt SSH không dừng tmux. Nếu legacy covariance gặp lỗi, run lưu failure và đánh dấu diverged; không tự sửa parameterization hoặc restart model khác.
