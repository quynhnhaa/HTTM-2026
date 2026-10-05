# CRPS MDN: NLL + CRPS objective — Design Spec (pre-declared, 2026-10-04)

## 1. Goal and claim boundary
Change the **training objective** of the baseline, not its architecture or capacity: LSTM (hidden 8) -> MDN with K = 3, legacy decoder, per-step bivariate mixtures, unchanged. The objective becomes

  loss = NLL + lambda * (CRPS_x + CRPS_y),  averaged over samples and forecast steps.

Motivation (hypothesis, not proven): NLL punishes overconfidence only through the log density at the single observed point and is sensitive to a few outliers; CRPS is a proper score that rewards calibration and sharpness together and grows only linearly in the error, so the extra term may improve reliability (Ravg, Rmin) and the tail behaviour behind S68/S95. Not claimed to be new (literature not checked); not claimed to be better before the results exist. The capacity experiments (shared decoder, hidden 32/64) are NOT part of this design.

## 2. Method (exact definition)
The marginal of a bivariate Gaussian mixture on one axis is a 1-D Gaussian mixture with the same weights pi_i, means mu_i (mu_x or mu_y) and standard deviations sigma_i (sigma_x or sigma_y); rho does not enter. For a 1-D Gaussian mixture F = sum_i pi_i N(mu_i, sigma_i^2) and an observation y, the closed form is (Grimit et al. 2006):

  CRPS(F, y) = sum_i pi_i A(y - mu_i, sigma_i^2) - 1/2 sum_i sum_j pi_i pi_j A(mu_i - mu_j, sigma_i^2 + sigma_j^2),
  A(m, s^2) = m (2 Phi(m / s) - 1) + 2 s phi(m / s),

with Phi, phi the standard normal cdf and pdf. It is exact and differentiable (no sampling). It is computed per axis and per forecast step and the two axes are summed. Units are metres, like the data.
Because only marginals enter, the correlation rho is trained by the NLL term only.
- Training objective: NLL + lambda * (CRPS_x + CRPS_y), with NLL computed exactly as the baseline (`NLL_MDN_loss`).
- Validation loss and best-checkpoint selection: the pure NLL (as in the baseline), so the numbers are comparable.
- With lambda = 0 the objective equals the baseline exactly (tested).

## 3. Pre-declared configuration (no tuning, one run)
- lambda = 1.0 (equal weight of the two scores in their natural units: nats and metres). A declared constant, not tuned; its effect is not explored unless the user asks, and then as a labelled sensitivity analysis, not as a way to pick the result.
- Everything else equals `default_peds_imptc.json`: K = 3, hidden 8, 2500 epochs, batch 4096, Adam 1e-3 with the same LinearLR, reductions 0.5, seed 2024, official metrics every 250 epochs, same fixed samples.

## 4. Success criteria (declared before training)
Compared with baseline K = 3 under the same protocol (same seed): NLL, Ravg, Rmin not worse, and S68, S95 not worse. Validation first (training-time official metrics at epoch 2500 plus a full-validation evaluation). The test split is used once, only after the user asks; the baseline test numbers are in `results/ablations/imptc_num_gaussians/ablation.json` (reproduced exactly by `shared_decoder_mdn.evaluate --baseline-run`). If the criteria are not met, that is reported as a negative result.

## 5. Risks (monitor and report)
- The CRPS term may trade NLL for reliability: pure validation NLL can be worse than the baseline; reported as such.
- Scale mismatch: CRPS in metres has a different magnitude from the NLL; lambda = 1 may over- or under-weight it. Not tuned.
- Marginal-only: CRPS ignores the correlation between x and y.
- A single seed: small differences are not conclusions.
- Not an architecture change, so any gain cannot come from capacity (a comparison advantage over the hidden-size runs), but also may simply be small.
