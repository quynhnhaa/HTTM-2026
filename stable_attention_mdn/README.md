# Stable Attention MDN

Kiến trúc AttentionMDN copy nguyên từ attention_mdn/model.py (7890 tham số).
Trainer, decoder, data loader, artifact tracker và evaluator được import từ
stable_mdn: cùng sigma floor 0.01 m, exp guard 1000 m, rho limit 0.999;
cùng NLL có trọng số theo mẫu, full validation, calibration ECDF và RNG riêng.

Full config giữ cùng dữ liệu, seed 2024, input 32/output 48/K=3, Adam, LR,
batch 4096, train reduction 0.5, 2500 epoch, lịch checkpoint và metric như
baseline stable. Chỉ kiến trúc và tên experiment/model namespace khác.

Smoke config: 3 epoch, 189 mẫu train/epoch, batch128, full validation,
MC64; chỉ ADE/FDE và reliability trong loop. Không dùng số smoke so chất lượng.
Giữ nguyên base_mdn và attention_mdn cũ. Kết quả nằm trong namespace riêng
results/trained_models/stable_attention_mdn/.

Training và evaluation dùng best checkpoint với metadata parameterization.
Không tự so với baseline legacy. --baseline-evaluation chỉ nhận báo cáo cùng
phiên bản evaluator stable_v2_ecdf_rng và cùng giao thức.
