# Growing MDN (split-and-grow K) — Design Spec

Date: 2026-10-04. Status: design approved in chat; spec awaiting review.

## 1. Goal
Learn the number of Gaussian components K for the IMPTC baseline instead of fixing K=3, by growing K from 1 by splitting components, then selecting K with a penalized validation rule. Independent of gated_mdn, bayesian_mdn, attention_mdn, residual_mdn and mode_consistent_mdn. Starting point is `base_mdn`, which must not be modified.

Success: a K-vs-validation curve from warm-started nested models, a K chosen by a pre-declared rule, and one test evaluation of that K with the official metrics, compared against baseline K=3 and K=8.

## 2. Verified facts about base_mdn
- `fc = Linear(8 -> 6*K*48)`; output reshaped to `[B, 48, 6K]`. Last dim is six blocks of size K: `mu_x, mu_y, log sigma_x, log sigma_y, rho_pre, pi_logit`. Flat row index of `fc` = `t*6K + block*K + k`, so changing K changes indexing; growth must remap rows.
- Per component: 6 blocks x 48 steps x (8 weights + 1 bias) = 2592 parameters (matches 8224 -> 21184 for K 3 -> 8).
- Distribution: `MixtureSameFamily(Categorical(pi), MultivariateNormal(mu, Sigma))`, independent per timestep; decoder legacy (sigma=exp, rho=tanh, pi=softmax). No joint trajectory mixture is claimed.
- `cfg.model_params['num_gaussians']` is read by `ExperimentTracker.capture_fixed_predictions`, `MDN_Forecaster`, `evaluate_run`; it must track the current K.
- Baseline protocol: Adam 1e-3, LinearLR 1.0 -> 1e-4 over 2500 epochs, batch 4096, train/eval reduction 0.5, seed 2024, LSTM hidden 8.

## 3. Layout (isolated, same pattern as gated_mdn)
New directory `growing_mdn/`, importing `base_mdn` modules through `sys.path` (data loader, ConfigLoader, tracker, `MDN_Forecaster`, `build_mdn_distribution`). No file in `base_mdn` is edited; SHA256 of `base_mdn` sources is recorded before and after each run.
- `model.py`: subclass of `LSTM_Trajectory_Forecast` with `grow(j)`.
- `train.py`: phase loop.
- `artifacts.py`: tracker subclass recording K, phase and growth events.
- `configs/imptc/*.json`: config copied from baseline with growth parameters added.
- `tests/`, `README.md`, `reports/`.

## 4. Splitting component j (K -> K+1)
- Rebuild `fc.weight` and `fc.bias` with the new indexing: copy all existing components, append a copy of component j's six rows at every timestep.
- pi: both copies get `logit_j - ln 2`, so the total weight of j is unchanged.
- Symmetry breaking: add `+delta` / `-delta` to the `mu_x` and `mu_y` biases of the two copies (delta = 0.05 in model coordinate units); sigma and rho unchanged.
- Adam: remap `exp_avg`/`exp_avg_sq` with the same index map; new rows inherit from the source row; step count preserved; LSTM state untouched. If the smoke test shows instability, fall back to zero-initialized moments for new rows and record the change.
- Invariant (unit test): validation NLL immediately after a split differs from before by less than 1e-3.

## 5. Choosing the component to split
After each phase, on the training set, for each component k compute the responsibility-weighted mean of `-log N_k(y)` over all samples and timesteps. Split the component with the largest score. All scores are logged. Component index is a head index, not a consistent physical mode across samples or timesteps; reports must say so.

## 6. Schedule
- Phase 0: K=1, 400 epochs. Each later phase: split one component, fine-tune 200 epochs. Stop growing at K_max=12. Total about 2600 epochs, comparable to the 2500-epoch baseline.
- LinearLR is restarted in every phase with the baseline start/end factors. This is a deviation from baseline and is stated in all reports.
- All phases run to K_max; K is selected afterwards, so the K curve is complete and not sensitive to an early-stop threshold.
- Best checkpoint per phase is chosen by validation NLL (full validation, sample-weighted, as in gated_mdn; this differs from baseline reduction 0.5 and is reported).

## 7. K selection rule (declared before any test evaluation)
Selected K = smallest K whose validation NLL is within epsilon of the best across phases, subject to validation Ravg/Rmin not worse and S68/S95 not larger than the best-NLL K. Epsilon is fixed in the config before viewing test results. BIC/AIC are reported but not used as the rule (timesteps within a trajectory are correlated). Multiple seeds are used to report K variability. Test data is used once, for the selected K, plus baseline K=3 and K=8 for comparison.

## 8. Artifacts
Per AGENTS.md: epoch, train/validation NLL, learning rate, official metrics, periodic/best/final checkpoints per phase, the 8 fixed samples from `base_mdn/configs/imptc/fixed_samples_seed2024.json`, observed/ground truth/predicted trajectories and raw plus decoded MDN parameters. Additional: K and phase in history, `growth_history.jsonl` (component scores, split choice, NLL before/after split), phase-end checkpoints. Official metrics come from `MDN_Forecaster`, unchanged.

## 9. Verification before any full run
Unit tests: row remapping of `fc` for several K; pi sums unchanged; NLL continuity after split; Adam state remapping; checkpoint resume across a phase boundary. Smoke run of a few epochs per phase. Review artifacts, fixed-sample capture and base_mdn hashes. No full training without explicit approval.

## 10. Risks and limits
- Selected K depends on epsilon, schedule and seed; it is not a proof of a global optimum.
- Warm-started nested models see more total epochs on shared parameters than a fresh K model; compare against baseline by total compute and state this.
- Single-seed results are insufficient to claim K differences; use multiple seeds before concluding.

## Revision 1 (2026-10-04) — split offset relative to sigma, softer LR restart, continuity guard

**Why.** The first full run (`growing_k12_seed2024`, stopped by the user) showed that the split in section 4 is not continuous on a trained model: validation NLL went from -0.0672 to 133.68 right after the K=1 -> 2 split and from -0.4255 to 2.52 after K=2 -> 3. The trained K=1 head has sigma of about 1e-3 m at the first forecast steps (about 1.4-1.6 m at step 48), so an absolute offset of 0.05 m is 40-60 sigma there. The unit test and smoke run did not catch it because they use untrained models with sigma of about 1. The invariant "validation NLL after a split differs from before by less than 1e-3" therefore must be checked on a trained model.

**Changes (these supersede the conflicting statements above).**
1. Split offset (section 4): the mu_x / mu_y bias offsets are `+/- delta_rel * sigma_med[t, axis]` instead of `+/- 0.05`. `sigma_med[t, axis]` is the median over all training samples of the decoded sigma of the source component at forecast step t. `delta_rel = 0.05` (config `growth.split_delta_relative`; the old `split_delta` key is removed). pi logits (`- ln 2`), sigma and rho handling are unchanged.
2. LR restart (section 6): phase 0 keeps lr 1e-3; every later phase restarts LinearLR from `lr_default * growth.restart_lr_factor` (0.1, i.e. 1e-4) with the same start/end factors. Rationale: restarting at 1e-3 on a converged model caused a training shock (train NLL 9.09 in the first epoch after the first split in the v1 run). This remains a deviation from the baseline protocol and is reported as such.
3. Continuity guard: after every split the trainer compares full-validation NLL before and after; if it is non-finite or `abs(after - before) > growth.max_split_nll_change` (0.01 for the full config) the run writes the growth record with `aborted: true`, sets the manifest status `aborted_split_discontinuity` and raises. The smoke config uses a looser 0.05 because its untrained model is not a meaningful test of continuity.
4. Verification on real converged weights: a read-only script applies old (absolute) and new (relative) splits to the saved `phase_k01.pt` / `phase_k02.pt` of the v1 run and reports validation NLL before/after. The relative split must give `abs(delta NLL) < 0.01`; if not, report and stop instead of tuning silently.
5. A regression unit test builds a model whose sigma is about 1e-3 at early steps and asserts: absolute 0.05 offset changes NLL by more than 10 nat, relative offset by less than 0.01.

The v1 run's artifacts are kept as evidence; its manifest is marked `stopped_by_user`. The new run uses id `growing_k12_v2_seed2024`.
