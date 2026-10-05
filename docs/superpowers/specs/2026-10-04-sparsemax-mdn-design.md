# Sparsemax MDN (per-sample learned K) — Design Spec

Date: 2026-10-04. Status: direction approved by the user in chat ("chạy sparsemax thuần trước, tạo một thư mục khác"); this spec fixes the protocol before any code or training.

## 1. Goal and claim boundary
Replace the softmax over the K_max mixture-weight logits of the baseline MDN by **sparsemax**, so that components that are not needed for a given sample and forecast step receive weight exactly 0. The effective number of components `K(x, t) = |{k : pi_k(x, t) > 0}|` then emerges per sample and per step from a single ordinary NLL training run: no extra parameters, no extra loss term, no selection among several trained variants.

This is **not** claimed to be new (not checked against the literature). It is a minimal, theoretically motivated change of the mixture-weight normalizer. "Learns K" means: K(x, t) is produced by the trained model per input, below the cap K_max.

## 2. Baseline facts (verified in code)
- Output head: `fc` -> `[B, 48, 6K]`, blocks `mu_x, mu_y, log sigma_x, log sigma_y, rho_pre, pi_logit`, each of size K; legacy decoder: sigma = exp, rho = tanh, pi = softmax; Sigma = [[sx^2, rho sx sy],[rho sx sy, sy^2]]. Mixtures are per forecast step (not a joint mixture over the trajectory).
- Baseline protocol (`base_mdn/configs/imptc/default_peds_imptc.json`): Adam lr 1e-3, LinearLR 1.0 -> 1e-4 over 2500 epochs, batch 4096, train_data_reduction 0.5, eval_data_reduction 0.5, seed 2024, LSTM hidden 8; official metrics: minADE/FDE (k=20), Ravg/Rmin, S68/S95, ASAEE. Baseline results exist for K = 1, 2, 3, 5, 8 (single seed) in `results/ablations/imptc_num_gaussians/ablation.json`.

## 3. Method
- Only the pi block changes: `pi = sparsemax(pi_logit)` over the K_max axis, per sample and per forecast step (Martins & Astudillo 2016; citation to be verified before use). mu, sigma, rho, LSTM, loss definition (mean negative log-likelihood of the mixture), optimizer, schedule are unchanged.
- Implementation constraint: keep `base_mdn` untouched and reuse its decoding/evaluation code. The model returns raw output in the baseline layout with the pi block replaced by `log(pi)` for components with `pi > 0` and a finite sentinel (-1e9) for `pi == 0`, so that the baseline `decode_mdn_output` (softmax) recovers exactly the sparsemax weights and every baseline consumer (tracker, fixed-sample capture, `MDN_Forecaster`) works unchanged. Training/validation NLL is computed with an exact masked log-sum-exp (components with `pi == 0` contribute exactly nothing and receive no gradient for that sample); the baseline `Categorical(probs)` clamps zero probabilities to about 1e-7, so official evaluation may include a negligible leakage that is documented, not hidden.
- Sparsemax is implemented with the standard sort/cumsum formula (support size `k(z) = max{k : 1 + k z_(k) > sum_{j<=k} z_(j)}`, threshold `tau = (sum_{j<=k(z)} z_(j) - 1) / k(z)`, `p = max(z - tau, 0)`), differentiable by autograd; unit tests check sum-to-one, non-negativity, exact zeros, invariance to adding a constant to the logits, agreement with a reference implementation on random inputs, and finite gradients.

## 4. Pre-declared configuration (no tuning, no selection among runs)
- `K_max = 8` (the largest K with a baseline, which was also the best baseline K on 7 of 8 test metrics; the K = 3 baseline is the second comparator).
- Pure sparsemax (alpha = 2). No sigma floor, no extra regularizer, no loss change. These are possible later additions only if the results show a need, and would be reported as separate experiments.
- Everything else equals `default_peds_imptc.json` with `num_gaussians = 8`: 2500 epochs, same optimizer/schedule/batch/reductions, seed 2024, the 8 fixed validation samples, official metrics every 250 epochs, checkpoints at epochs 1, 5, 10 and every 100.
- One training run. The checkpoint with the best (reduced) validation NLL is used for evaluation, exactly as the baseline.

## 5. Success criteria (declared before training)
Primary: compared with the baseline at K = 3 and K = 8 under the same budget, the official metrics (validation first; test only once at the end for the final run, after the user asks) show non-inferior or better NLL, Ravg and Rmin, and non-worse sharpness (S68/S95). Secondary (adaptive-K hypotheses): mean K(x, t) is clearly below K_max, K varies across samples and horizons, and K(x, t) is associated with sample difficulty (e.g. larger minADE or larger predictive spread). If these do not hold, that is reported as is.

Limits to state with the results: a single seed (baseline seed variance is unknown, so small differences are not claims), the baseline numbers come from separate runs, K(x, t) is a head-index count not a physical mode, and `alpha`, `K_max` are design constants fixed in advance.

## 6. Artifacts and analyses
Per epoch: train/validation NLL, learning rate, **mean support size K(x, t) on validation** and the fraction of samples/steps at each K, plus the baseline artifacts (periodic/best/last/final checkpoints, fixed-sample predictions with pi, mu, sigma, rho, official metrics every 250 epochs). A read-only analysis script reports: distribution of K(x, t) overall and per forecast step; components never active on validation (dead components); association between K(x, t) and per-sample error (validation only); the same fixed samples across checkpoints. The test split is evaluated once, only on explicit user request.

## 7. Risks (to monitor, not to hide)
Dead components (zero gradient where pi = 0), NLL spikes when active components have tiny density at the true point, narrow-sigma components that fit training clusters (the baseline shows validation-train gap growing with K: 0.10 at K = 1 to 0.17 at K = 8; test reliability improved with K, so overfit-driven unreliability is a hypothesis, not an observation). Sparsemax does not address these; a sigma floor or other regularizer would be a separate, disclosed experiment.

## 8. Evaluation caveats recorded before the run (from the code review)
1. The official metrics (`MDN_Forecaster`) build the mixture with `Categorical(probs=pi)`, whose `.logits` are `log(clamp(pi, eps=1.19e-7))`. For components with `pi == 0` the density is therefore scored with weight about 1.2e-7 instead of exactly 0 in `log_prob`-based metrics (reliability confidence sets, k-sample probabilities, sharpness grid, ASAEE argmin); sampling is exact (zero-probability components are never sampled). Training and validation NLL of this method use the exact masked log-sum-exp. After training, the saved checkpoints allow a read-only re-evaluation with exact weights to check how much this matters; until then official metrics are reported as "base code applied as-is".
2. The baseline's own training/validation NLL also uses the eps-clamped `Categorical`, so for any baseline softmax weight below 1.2e-7 the two NLLs are not strictly the same quantity. When comparing NLL with the baseline, state this.

## 9. Revision 1 (2026-10-04): K_max = 16
The user asked for a cap above 8 so that K is not truncated by a low cap. The first run (`sparsemax_k8_seed2024`, K_max = 8) was stopped by the user after 48 epochs; its artifacts are kept (manifest `stopped_by_user`). In that run the mean support size fell to about 2.2 of 8 within 20 epochs and no sample used all 8 components after the first epochs, so the cap was not obviously binding, but a larger cap lets the model decide freely.
- Pre-declared change: `K_max = 16` (config `sparsemax_k16_peds_imptc.json`; everything else identical, including seed 2024 and the baseline protocol). Section 4 above is superseded for K_max only.
- Cap check, reported with the results: the fraction of (sample, forecast step) pairs whose support equals K_max, per epoch (`fraction_full_support` in `history.csv`). If it is not negligible, the cap is binding and the result is reported as a lower bound on the unconstrained K, not as a learned K.
- Baselines for comparison remain K = 3 and K = 8 (no baseline exists at K = 16, so capacity of the cap differs from both; the number of parameters of the sparsemax model at K_max = 16 is 41 920 vs 8 224 and 21 184, although most components are expected to be inactive).
