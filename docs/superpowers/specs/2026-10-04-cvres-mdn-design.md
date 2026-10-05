# CV-residual MDN: predict the offset from a constant-velocity extrapolation (pre-declared, 2026-10-04)

## 1. Goal and claim boundary
Change the **output parametrisation** of the baseline, not its loss or capacity. The mixture means are expressed as a constant-velocity (CV) extrapolation of the observed trajectory plus a network-predicted offset. Loss (exact NLL), architecture (LSTM hidden 8 -> linear head), K = 3, sigma/rho/pi decoding, optimizer, schedule, preprocessing and metrics are unchanged. The model has exactly the same parameters as the baseline (its state dict has the same keys), so any gain cannot come from capacity.
Residual-over-a-kinematic-prior is a standard idea in trajectory prediction; it is not claimed to be new here, and it is not claimed to be better before the results exist.

## 2. Verified data facts (measured on the full validation split, 19148 samples, `reports/CV_PRIOR_CHECK.json`)
- Input X [32, 4]: columns 0-1 are positions in the ego frame, the last observed position is (0, 0); columns 2-3 are velocities in m/s, equal to (pos[t] - pos[t-1]) / dt with dt = 0.1 (correlation 1.0000 on both axes, magnitude ratio 0.1).
- Target y [48, 2]: future positions in the same frame (origin at the last observed position); y[0] correlates 0.995 / 0.999 (x / y) with the last observed displacement.
- CV extrapolation: y_cv[k] = v_last * dt * (k + 1), k = 0..47, with v_last = X[:, -1, 2:4].
- Mean Euclidean error on validation: CV line ADE 0.635 m vs the baseline K = 3 mixture mean ADE 0.512 m (CV 0.079 / 0.530 / 1.613 m at 0.8 / 2.4 / 4.8 s against 0.040 / 0.401 / 1.401 m for the baseline). So the trained baseline already beats the plain CV line (by about 19% in ADE), but the CV line carries most of the displacement (about 3.7 m mean at 4.8 s versus a 1.6 m CV error).

## 3. Method (exact definition)
Raw head output [B, 48, 6K] with blocks of K: mu_x, mu_y, log sigma_x, log sigma_y, rho_pre, pi_logit (unchanged layout). The model adds the CV term to the first two blocks:
- mu_x[b, k, i] = raw_mu_x[b, k, i] + v_last_x[b] * dt * (k + 1)
- mu_y[b, k, i] = raw_mu_y[b, k, i] + v_last_y[b] * dt * (k + 1)
for every component i. Every other block is untouched. dt = `delta_t` of the config. The velocity columns are listed in the config (`cv_residual.velocity_columns` = [2, 3]).

## 4. Pre-declared configuration (no tuning, one run)
Everything equals `default_peds_imptc.json` (K = 3, hidden 8, 2500 epochs, batch 4096, Adam 1e-3 + same LinearLR, reductions 0.5, seed 2024), plus `cv_residual.velocity_columns = [2, 3]`.

## 5. Success criteria (declared before training)
Compared with baseline K = 3, same protocol and seed: NLL, Ravg, Rmin not worse, and S68, S95 not worse; validation first (training-time official metrics and a full-validation evaluation). The test split is used once and only when the user asks (baseline test numbers: `results/ablations/imptc_num_gaussians/ablation.json`, reproduced exactly). A negative result is reported as such.

## 6. Risks (monitor and report)
- The baseline already beats the CV line, so the gain may be small or zero.
- The velocity estimate at the last step is noisy; errors grow with horizon and the network must correct them.
- S68/S95 are dominated by a few hard samples (turns, stops) where the CV line does not help.
- A checkpoint of this model loaded into the plain baseline class produces silently wrong means (same keys, no offset): always evaluate with `cvres_mdn.evaluate` / `CVResidualMDN`.
- A single seed: small differences are not conclusions.
