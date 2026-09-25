# Pipeline instrumentation verification

Date: 2026-09-24

## Scope

This report records implementation fixes and smoke verification performed before
any full IMPTC baseline run. The LSTM architecture, MDN likelihood, optimizer,
learning rate, number of Gaussian components, preprocessing, and official
baseline data reductions were not intentionally changed.

## Correctness fixes

### Shared MDN/GMM decoding

Training and evaluation now call the same distribution builder. For every future
timestep it decodes three bivariate Gaussian components as:

- `pi = softmax(alpha, dim=-1)`;
- `mu = (mu_x, mu_y)`;
- `sigma = exp(raw_sigma)`;
- `rho = tanh(raw_rho)`;
- `Sigma = [[sigma_x^2, rho*sigma_x*sigma_y],
            [rho*sigma_x*sigma_y, sigma_y^2]]`.

The former evaluation path normalized mixture logits on dimension 1. For an
output shaped `[batch, horizon, component]`, that dimension is the forecast
horizon rather than the component axis. It is now normalized on `dim=-1`, as in
the training likelihood.

### Sharpness grid area

The grid spans `[-range_x, +range_x] x [-range_y, +range_y]`. Its physical area
is therefore `(2*range_x)*(2*range_y)`. The former multiplier used only
`range_x*range_y`, under-reporting area-derived sharpness by a factor of four.
See `SHARPNESS_VERIFICATION.md` for the numerical check. New artifacts carry
`sharpness_formula_version: corrected` and must not be silently compared with
uncorrected upstream values.

### Evaluation timing and testing startup

GPU inference timing now synchronizes CUDA before and after the measured forward
pass and uses a monotonic high-resolution clock. The testing script's invalid
dummy forward pass, whose tensor shape did not match the model input, was
removed.

## Added experiment artifacts

Each instrumented run writes:

- resolved configuration and environment metadata;
- epoch-level train NLL, validation NLL, learning rate, duration, and sample counts;
- atomic best, periodic, last, and final checkpoints;
- optimizer, scheduler, RNG, and history state needed for resume;
- a stable eight-sample validation manifest and immutable input/ground truth file;
- checkpoint-specific raw MDN outputs and decoded `pi`, `mu`, `sigma`, `rho`, and covariance;
- official evaluation metric snapshots in JSON and CSV.

## Verification evidence

### Automated tests

Command:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Result: 6/6 tests passed. The tests cover output/parameter shapes, component-axis
normalization, shared train/evaluation distributions, positive-definite decoded
covariance, RNG restoration, and the corrected mesh-area factor.

### Checkpoint/artifact smoke run

Run ID: `smoke_protocol_v1`; config: `smoke_peds_imptc.json`; epochs: 3.

- history contains epochs 1, 2, and 3;
- best epoch is 3 for this smoke run;
- best, epoch 1/2/3, last, and final checkpoints exist;
- restored last checkpoint reports epoch 3 and scheduler `last_epoch = 3`;
- fixed prediction shapes include `pi: [8, 48, 3]`;
- mixture weights sum to one and decoded covariance matrices are positive definite.

### Full metric-path smoke run

Run ID: `smoke_metrics_v1`; config: `metric_smoke_peds_imptc.json`; epochs: 1.
It intentionally uses only about 18 training samples and 3 validation samples,
20 Monte Carlo confidence samples, two horizons, and a coarse 5-by-5 grid.

Verified metric keys:

- `ravg_percent`, `rmin_percent`, `reliability_by_horizon_percent`;
- `s95_m2_per_s`, `s68_m2_per_s`;
- `asaee_m_per_s`;
- `minade20_m`, `minfde20_m`;
- `inference_ms_per_batch`.

The run completed and wrote one consistent row to both epoch history and metric
history, plus all expected checkpoints and fixed-sample predictions.

## Interpretation warning

Smoke-run loss and metric values have no experimental meaning because the runs
use tiny random subsets, very few epochs, reduced Monte Carlo sampling, and (for
the metric smoke run) a coarse restricted spatial grid. They only demonstrate
that the pipeline executes and preserves the intended artifact schema.

## Status before full training

The technical path for training loss, validation loss, official metrics,
checkpoints, resume state, and fixed-sample MDN parameters has been exercised.
A full baseline remains intentionally unstarted pending explicit review and
authorization.

## Resume correction and end-to-end verification

The initial resume audit found that optimizer moment tensors were restored on
CPU before the model was moved to CUDA. The first resumed `optimizer.step()`
therefore failed with a mixed CPU/CUDA tensor error. Resume initialization now:

- selects CUDA before the first CUDA API call;
- creates the model on the target device before constructing/loading Adam;
- loads the checkpoint with the target device mapping;
- normalizes serialized CPU/CUDA RNG byte tensors to the devices required by
  PyTorch's generator APIs;
- preserves the mutable in-place training-data order in every checkpoint;
- restores that order before the next epoch;
- appends to `training.log` rather than truncating it;
- treats `MDN_RESUME_CHECKPOINT` as an explicit resume request, allowing the
  original config filename and artifact directory to remain unchanged.

Regression suite result after the correction: 8/8 tests passed, including an
Adam optimizer step on CUDA and exact next-epoch sample/order restoration.

An end-to-end GPU smoke run was trained through epoch 3, stopped, and resumed
from `last.pt` through epoch 4. Verification result:

- history epochs: `[1, 2, 3, 4]` with no duplicate epoch;
- resumed checkpoint epoch: 4;
- scheduler `last_epoch`: 4;
- resume event source epoch: 3;
- `data_loader_state_restored`: `true`;
- the resumed CUDA optimizer step and epoch completed successfully.
