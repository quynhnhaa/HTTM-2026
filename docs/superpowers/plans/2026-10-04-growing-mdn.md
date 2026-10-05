# Growing MDN (split-and-grow K) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `growing_mdn/`, an isolated variant of `base_mdn` whose number of Gaussian components K grows from 1 to 12 by splitting one component per phase, then select K with a pre-declared penalized validation rule.

**Architecture:** `growing_mdn/` imports the unchanged `base_mdn` modules (ConfigLoader, DataLoader, ExperimentTracker, MDN_Forecaster, `build_mdn_distribution`) through `sys.path`, exactly like `gated_mdn/`. It adds only: a model helper that rebuilds the `fc` head with K+1 components, an Adam-state remapper, a component-scoring/selection module, a tracker subclass, a phase-loop trainer, a K-selection CLI and a test evaluator.

**Tech Stack:** Python (repo `.venv`), PyTorch 2.11, numpy, `unittest` (pytest is not installed), IMPTC data under `data/trajdata/ego/imptc`.

**Spec:** `docs/superpowers/specs/2026-10-04-growing-mdn-design.md`

## Global Constraints

- `base_mdn/` must not be modified. Its SHA256 snapshot is recorded in `growing_mdn/reports/BASE_SOURCE_HASHES.json` and verified at the start and end of every run.
- Decoder stays legacy: sigma = exp, rho = tanh, pi = softmax, `Sigma = [[sx^2, rho*sx*sy],[rho*sx*sy, sy^2]]`.
- LSTM hidden 8, 1 layer, input 4, forecast horizon 48, Adam lr 1e-3, LinearLR start 1.0 -> end 1e-4 restarted in every phase, batch 4096 (train), train_data_reduction 0.5, full validation (`eval_data_reduction` 1.0), seed 2024.
- `fc` row index is `t*6K + block*K + k`; blocks are `mu_x, mu_y, log sigma_x, log sigma_y, rho_pre, pi_logit`. Each added component adds 2592 parameters.
- Split: pi logits of both copies = `logit_j - ln 2`; `mu_x`/`mu_y` biases `+delta` / `-delta`, `delta = 0.05`; sigma and rho unchanged.
- Schedule: K=1 for 400 epochs, then one split plus 200 epochs per phase, K_max = 12 (400 + 11*200 = 2600 epochs). All phases always run to K_max; K is selected afterwards.
- Per-phase best checkpoint is chosen by full validation NLL; the next split and the phase's official metrics use that best state.
- K selection uses validation only; tolerances are fixed in the config before any test evaluation. Test data is evaluated once, for the selected K.
- 8 fixed validation samples from `base_mdn/configs/imptc/fixed_samples_seed2024.json`.
- No full training without explicit user approval. Smoke runs only (<= 6 epochs) until then.
- Commit steps below run only if the user has authorized commits; end commit messages with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## Review Focus

- A component with ~0 responsibility mass (dead) has score `-inf`; if all scores are `-inf` the trainer must raise a clear error, not split an arbitrary row. (Task 3)
- Row remapping when K changes: wrong `fc` index math silently scrambles components; verified by decoded-parameter equality, not shapes. (Task 1, Task 2)
- Resume at a phase boundary (`phase_k##.pt`) and mid-phase (`last.pt`) must continue the exact same trajectory; a `last.pt` with all epochs done must still run the phase-end work. (Task 5, Task 6)
- K-selection edge cases: the best-NLL K is the smallest, ties, a non-finite metric, tolerances so tight the rule always picks the best-NLL K. (Task 3)
- Config misuse: `train_epochs` != phase total, `k_max <= initial_k`, `split_delta <= 0`, non-legacy covariance, `eval_data_reduction != 1`; and `evaluate.py` loading a checkpoint whose K differs from the requested K. (Task 4, Task 6)

---

### Task 1: Model helper — rebuild the head with K+1 components

**Files:**
- Create: `growing_mdn/__init__.py` (empty), `growing_mdn/tests/__init__.py` (empty)
- Create: `growing_mdn/model.py`
- Test: `growing_mdn/tests/test_model.py`

**Interfaces:**
- Produces:
  - `BLOCKS = 6`
  - `build_model(model_params: dict, k: int, device='cpu') -> LSTM_Trajectory_Forecast` (attaches `model.growth_params`, a deep copy of the params with `num_gaussians=k`)
  - `num_components(model) -> int`
  - `fc_row_map(k_old: int, source: int, horizon: int) -> LongTensor` (for each new `fc` row, the old row it copies)
  - `new_component_rows(k_old: int, horizon: int) -> LongTensor` (new-model rows belonging to the appended component, index `k_old`)
  - `grow_model(model, source: int, delta: float) -> (new_model, rows)`

- [ ] **Step 1: Write the failing tests**

Create `growing_mdn/tests/test_model.py`:

```python
"""Head-growth correctness: decoded parameters, not just shapes."""
import math
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from growing_mdn.model import (BLOCKS, build_model, fc_row_map, grow_model,
                               new_component_rows, num_components)
from utils.mdn_distribution import build_mdn_distribution, decode_mdn_output

PARAMS = {'lstm_input_shape': 4, 'lstm_hidden_size': 8, 'lstm_num_layers': 1,
          'num_gaussians': 3, 'output_factor': 6, 'forecast_horizon': 48,
          'mdn_parameterization': {'mode': 'legacy'}}


class GrowTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2024)

    def test_row_map_layout(self):
        rows = fc_row_map(3, 1, 48).reshape(48, BLOCKS, 4)
        for t in (0, 7, 47):
            for b in range(BLOCKS):
                for k in range(3):
                    self.assertEqual(int(rows[t, b, k]), t * 18 + b * 3 + k)
                self.assertEqual(int(rows[t, b, 3]), t * 18 + b * 3 + 1)

    def test_new_component_rows(self):
        rows = new_component_rows(3, 48)
        self.assertEqual(len(rows), 48 * BLOCKS)
        self.assertEqual(int(rows[0]), 3)
        self.assertEqual(int(rows[1]), 1 * 4 + 3)
        self.assertEqual(int(rows[BLOCKS]), 1 * 6 * 4 + 3)

    def test_grown_parameters(self):
        old = build_model(PARAMS, 3).eval()
        with torch.no_grad():
            old.fc.bias.normal_(0, 0.3)
        new, _ = grow_model(old, source=1, delta=0.05)
        new.eval()
        self.assertEqual(num_components(new), 4)
        x = torch.randn(4, 32, 4)
        po, pn = decode_mdn_output(old(x), 3), decode_mdn_output(new(x), 4)
        for k in (0, 2):
            for key in ('pi', 'mu', 'sigma', 'rho'):
                torch.testing.assert_close(pn[key][:, :, k], po[key][:, :, k])
        for key in ('sigma', 'rho'):
            torch.testing.assert_close(pn[key][:, :, 1], po[key][:, :, 1])
            torch.testing.assert_close(pn[key][:, :, 3], po[key][:, :, 1])
        torch.testing.assert_close(pn['pi'][:, :, 1], pn['pi'][:, :, 3])
        torch.testing.assert_close(pn['pi'][:, :, 1] + pn['pi'][:, :, 3], po['pi'][:, :, 1])
        torch.testing.assert_close(pn['mu'][:, :, 1], po['mu'][:, :, 1] + 0.05)
        torch.testing.assert_close(pn['mu'][:, :, 3], po['mu'][:, :, 1] - 0.05)
        torch.testing.assert_close(pn['pi'].sum(-1), torch.ones(4, 48))

    def test_lstm_copied(self):
        old = build_model(PARAMS, 3)
        new, _ = grow_model(old, 0, 0.05)
        for (n1, p1), (n2, p2) in zip(old.lstm.named_parameters(), new.lstm.named_parameters()):
            self.assertEqual(n1, n2)
            self.assertTrue(torch.equal(p1, p2))

    def test_nll_continuity_after_split(self):
        old = build_model(PARAMS, 3).eval()
        x = torch.randn(2048, 32, 4)
        with torch.no_grad():
            dist_old = build_mdn_distribution(old(x), 3)
            y = dist_old.sample()
            nll_old = -dist_old.log_prob(y).mean()
            new, _ = grow_model(old, 2, 0.05)
            nll_new = -build_mdn_distribution(new.eval()(x), 4).log_prob(y).mean()
        self.assertLess(abs(float(nll_new - nll_old)), 1e-3)

    def test_invalid_arguments(self):
        old = build_model(PARAMS, 3)
        with self.assertRaises(ValueError):
            grow_model(old, 3, 0.05)
        with self.assertRaises(ValueError):
            grow_model(old, -1, 0.05)
        with self.assertRaises(ValueError):
            grow_model(old, 0, 0.0)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/nhantq/Documents/IotLink/Sharing-macbook/GiuaKyHTMM/mdn_trajectory_forecasting && .venv/bin/python -m unittest growing_mdn.tests.test_model -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'growing_mdn.model'`.

- [ ] **Step 3: Implement `growing_mdn/model.py`**

Create empty `growing_mdn/__init__.py` and `growing_mdn/tests/__init__.py`, then:

```python
"""LSTM-MDN whose output head can grow by splitting one component (K -> K+1).

The architecture and covariance parameterization are exactly those of
``base_mdn/base_lstm.py``; only the construction of a (K+1)-component head from
a K-component one is added here.
"""
import copy
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / 'base_mdn')):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from base_lstm import LSTM_Trajectory_Forecast  # noqa: E402

# mu_x, mu_y, log sigma_x, log sigma_y, rho_pre, pi_logit
BLOCKS = 6
MU_BLOCKS = (0, 1)
PI_BLOCK = 5


def build_model(model_params, k, device='cpu'):
    params = copy.deepcopy(model_params)
    params['num_gaussians'] = int(k)
    if params['output_factor'] != BLOCKS:
        raise ValueError('growing_mdn assumes the 6-block legacy MDN layout')
    model = LSTM_Trajectory_Forecast(params).to(device)
    model.growth_params = params
    return model


def num_components(model):
    return model.output_size // BLOCKS


def fc_row_map(k_old, source, horizon):
    """For every row of the (K+1)-head, the row of the K-head it copies.

    Flat row index = t * 6K + block * K + k. The appended component (index
    ``k_old``) copies component ``source``.
    """
    k_new = k_old + 1
    t = torch.arange(horizon).view(-1, 1, 1)
    b = torch.arange(BLOCKS).view(1, -1, 1)
    k = torch.arange(k_new).view(1, 1, -1)
    old_k = torch.where(k == k_old, torch.full_like(k, source), k)
    return (t * BLOCKS * k_old + b * k_old + old_k).reshape(-1)


def new_component_rows(k_old, horizon):
    """Rows of the (K+1)-head that belong to the appended component."""
    k_new = k_old + 1
    t = torch.arange(horizon).view(-1, 1, 1)
    b = torch.arange(BLOCKS).view(1, -1, 1)
    return (t * BLOCKS * k_new + b * k_new + k_old).reshape(-1)


@torch.no_grad()
def grow_model(model, source, delta):
    """Split component ``source`` into two; return ``(new_model, rows)``."""
    k_old = num_components(model)
    if not 0 <= source < k_old:
        raise ValueError(f'source {source} outside [0, {k_old})')
    if not delta > 0:
        raise ValueError('delta must be positive')
    horizon = model.forecast_horizon
    device = model.fc.weight.device
    new = build_model(model.growth_params, k_old + 1, device=device)
    new.lstm.load_state_dict(model.lstm.state_dict())
    rows = fc_row_map(k_old, source, horizon).to(device)
    new.fc.weight.copy_(model.fc.weight[rows])
    new.fc.bias.copy_(model.fc.bias[rows])
    bias = new.fc.bias.view(horizon, BLOCKS, k_old + 1)
    bias[:, PI_BLOCK, source] -= math.log(2.0)
    bias[:, PI_BLOCK, k_old] -= math.log(2.0)
    for block in MU_BLOCKS:
        bias[:, block, source] += delta
        bias[:, block, k_old] -= delta
    return new, rows
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m unittest growing_mdn.tests.test_model -v`
Expected: 6 tests PASS. If `test_nll_continuity_after_split` fails by a small margin, report the measured difference and stop; do not silently loosen the 1e-3 bound or change delta (this is a spec invariant).

- [ ] **Step 5: Commit (only if authorized)**

```bash
git add growing_mdn/__init__.py growing_mdn/tests/__init__.py growing_mdn/model.py growing_mdn/tests/test_model.py
git commit -m "feat(growing_mdn): add head growth by component split" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Adam state remapping

**Files:**
- Create: `growing_mdn/optim_state.py`
- Test: `growing_mdn/tests/test_optim_state.py`

**Interfaces:**
- Consumes: `build_model`, `grow_model`, `new_component_rows` from `growing_mdn.model`.
- Produces: `remap_adam_state(old_opt, new_opt, old_model, new_model, rows, zero_new_rows=False) -> None` — copies the Adam state of every parameter from `old_opt` into `new_opt`; for `fc.weight`/`fc.bias` the `exp_avg`/`exp_avg_sq` are gathered with `rows`; if `zero_new_rows` the rows of the appended component are zeroed.

- [ ] **Step 1: Write the failing tests**

Create `growing_mdn/tests/test_optim_state.py`:

```python
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from growing_mdn.model import build_model, grow_model, new_component_rows
from growing_mdn.optim_state import remap_adam_state
from utils.mdn_distribution import build_mdn_distribution

PARAMS = {'lstm_input_shape': 4, 'lstm_hidden_size': 8, 'lstm_num_layers': 1,
          'num_gaussians': 3, 'output_factor': 6, 'forecast_horizon': 48,
          'mdn_parameterization': {'mode': 'legacy'}}


def one_step(model, optimizer, k):
    x, y = torch.randn(16, 32, 4), torch.randn(16, 48, 2)
    optimizer.zero_grad()
    loss = -build_mdn_distribution(model(x), k).log_prob(y).mean()
    loss.backward()
    optimizer.step()
    return float(loss)


class RemapTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2024)
        self.old = build_model(PARAMS, 3)
        self.old_opt = torch.optim.Adam(self.old.parameters(), lr=1e-3)
        one_step(self.old, self.old_opt, 3)
        one_step(self.old, self.old_opt, 3)
        self.new, self.rows = grow_model(self.old, 1, 0.05)
        self.new_opt = torch.optim.Adam(self.new.parameters(), lr=1e-3)

    def test_fc_moments_gathered_and_lstm_copied(self):
        remap_adam_state(self.old_opt, self.new_opt, self.old, self.new, self.rows)
        for name in ('weight', 'bias'):
            old_state = self.old_opt.state[getattr(self.old.fc, name)]
            new_state = self.new_opt.state[getattr(self.new.fc, name)]
            for key in ('exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(new_state[key], old_state[key][self.rows])
            self.assertEqual(float(new_state['step']), float(old_state['step']))
        for (_, po), (_, pn) in zip(self.old.lstm.named_parameters(), self.new.lstm.named_parameters()):
            for key in ('exp_avg', 'exp_avg_sq'):
                self.assertTrue(torch.equal(self.old_opt.state[po][key], self.new_opt.state[pn][key]))

    def test_zero_new_rows(self):
        remap_adam_state(self.old_opt, self.new_opt, self.old, self.new, self.rows, zero_new_rows=True)
        rows = new_component_rows(3, 48)
        state = self.new_opt.state[self.new.fc.weight]
        self.assertEqual(float(state['exp_avg'][rows].abs().sum()), 0.0)
        self.assertEqual(float(state['exp_avg_sq'][rows].abs().sum()), 0.0)
        kept = torch.ones(state['exp_avg'].shape[0], dtype=torch.bool)
        kept[rows] = False
        self.assertGreater(float(state['exp_avg'][kept].abs().sum()), 0.0)

    def test_training_continues_finite(self):
        remap_adam_state(self.old_opt, self.new_opt, self.old, self.new, self.rows)
        for _ in range(3):
            self.assertTrue(torch.isfinite(torch.tensor(one_step(self.new, self.new_opt, 4))))


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m unittest growing_mdn.tests.test_optim_state -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'growing_mdn.optim_state'`.

- [ ] **Step 3: Implement `growing_mdn/optim_state.py`**

```python
"""Carry Adam moments across a head growth so training continues smoothly."""
import torch

from growing_mdn.model import new_component_rows

FC_NAMES = ('fc.weight', 'fc.bias')
ROW_KEYS = ('exp_avg', 'exp_avg_sq')


@torch.no_grad()
def remap_adam_state(old_opt, new_opt, old_model, new_model, rows, zero_new_rows=False):
    old_params = dict(old_model.named_parameters())
    zero_rows = None
    if zero_new_rows:
        k_old = old_model.output_size // 6
        zero_rows = new_component_rows(k_old, old_model.forecast_horizon).to(rows.device)
    for name, new_param in new_model.named_parameters():
        state = old_opt.state.get(old_params[name])
        if not state:
            continue
        mapped = {}
        for key, value in state.items():
            if not torch.is_tensor(value):
                mapped[key] = value
            elif name in FC_NAMES and key in ROW_KEYS:
                gathered = value[rows].clone()
                if zero_rows is not None:
                    gathered[zero_rows] = 0
                mapped[key] = gathered
            else:
                mapped[key] = value.clone()
        new_opt.state[new_param] = mapped
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m unittest growing_mdn.tests.test_optim_state growing_mdn.tests.test_model -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit (only if authorized)**

```bash
git add growing_mdn/optim_state.py growing_mdn/tests/test_optim_state.py
git commit -m "feat(growing_mdn): remap Adam state across head growth" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Component scoring and K selection

**Files:**
- Create: `growing_mdn/selection.py`
- Test: `growing_mdn/tests/test_selection.py`

**Interfaces:**
- Consumes: `num_components`, `build_model` from `growing_mdn.model`; `decode_mdn_output` from `base_mdn/utils/mdn_distribution.py`.
- Produces:
  - `component_scores(model, X, y, device, batch_size) -> numpy.ndarray` shape `[K]`: responsibility-weighted mean of `-log N_k(y)` over samples and timesteps; `-inf` where the total responsibility is below 1e-8.
  - `pick_split(scores) -> int` (index of the largest score; first index on ties; `ValueError` if no finite score).
  - `METRIC_KEYS` tuple of the official metric names used by the rule.
  - `select_k(records, epsilon_nll, ravg_tolerance_pp, sharpness_tolerance_ratio) -> dict` with keys `selected_k`, `best_nll_k`, `candidate_ks`, `rule`.

- [ ] **Step 1: Write the failing tests**

Create `growing_mdn/tests/test_selection.py`:

```python
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from growing_mdn.model import build_model
from growing_mdn.selection import component_scores, pick_split, select_k

PARAMS = {'lstm_input_shape': 4, 'lstm_hidden_size': 8, 'lstm_num_layers': 1,
          'num_gaussians': 2, 'output_factor': 6, 'forecast_horizon': 48,
          'mdn_parameterization': {'mode': 'legacy'}}


def constant_model(sigma1_log=2.0, mu1=0.0):
    """K=2 model whose output ignores the input: comp0 N(0,1), comp1 broader."""
    model = build_model(PARAMS, 2)
    with torch.no_grad():
        model.fc.weight.zero_()
        model.fc.bias.zero_()
        bias = model.fc.bias.view(48, 6, 2)
        bias[:, 2, 1] = sigma1_log
        bias[:, 3, 1] = sigma1_log
        bias[:, 0, 1] = mu1
        bias[:, 1, 1] = mu1
    return model


class ScoreTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2024)

    def test_worse_fitting_component_scores_higher(self):
        model = constant_model()
        X = np.zeros((512, 32, 4), dtype=np.float32)
        y = torch.randn(512, 48, 2).numpy()
        scores = component_scores(model, X, y, 'cpu', 128)
        self.assertEqual(scores.shape, (2,))
        self.assertGreater(scores[1], scores[0])
        self.assertEqual(pick_split(scores), 1)

    def test_dead_component_is_minus_inf(self):
        model = constant_model(sigma1_log=0.0, mu1=100.0)
        X = np.zeros((256, 32, 4), dtype=np.float32)
        y = torch.randn(256, 48, 2).numpy()
        scores = component_scores(model, X, y, 'cpu', 128)
        self.assertTrue(np.isneginf(scores[1]))
        self.assertTrue(np.isfinite(scores[0]))
        self.assertEqual(pick_split(scores), 0)

    def test_pick_split_requires_finite_score(self):
        with self.assertRaises(ValueError):
            pick_split(np.array([-np.inf, -np.inf]))

    def test_pick_split_tie_takes_first(self):
        self.assertEqual(pick_split(np.array([1.0, 3.0, 3.0])), 1)


def record(k, nll, ravg, rmin, s68, s95):
    return {'k': k, 'validation_nll': nll, 'ravg_percent': ravg, 'rmin_percent': rmin,
            's68_m2_per_s': s68, 's95_m2_per_s': s95}


# Values from docs/BASE_MDN_K_COMPARISON.md (test table), used only as shape-of-data fixtures.
RECORDS = [record(1, -0.3446, 86.4988, 76.5309, 3.8193, 8.1405),
           record(2, -0.9265, 95.7992, 90.2453, 5.9493, 11.8354),
           record(3, -1.0921, 97.4709, 94.0646, 1.3575, 6.0610),
           record(5, -1.1317, 97.9062, 95.9466, 2.2009, 7.6275),
           record(8, -1.2066, 98.4943, 96.5872, 0.9022, 5.3482)]


class SelectTest(unittest.TestCase):
    def test_sharpness_guard_keeps_best_nll_k(self):
        out = select_k(RECORDS, 0.1, 1.0, 0.10)
        self.assertEqual(out['best_nll_k'], 8)
        self.assertEqual(out['selected_k'], 8)

    def test_loose_sharpness_selects_smaller_k(self):
        out = select_k(RECORDS, 0.1, 1.0, 10.0)
        self.assertEqual(out['selected_k'], 5)
        self.assertEqual(out['candidate_ks'], [5, 8])

    def test_zero_tolerances_return_best_nll_k(self):
        self.assertEqual(select_k(RECORDS, 0.0, 0.0, 0.0)['selected_k'], 8)

    def test_best_is_smallest_k(self):
        records = [record(1, -2.0, 99, 98, 1.0, 2.0), record(2, -1.0, 90, 80, 2.0, 4.0)]
        self.assertEqual(select_k(records, 0.5, 1.0, 0.1)['selected_k'], 1)

    def test_non_finite_metric_rejected(self):
        bad = RECORDS + [record(10, -1.3, float('nan'), 97, 1.0, 5.0)]
        with self.assertRaises(ValueError):
            select_k(bad, 0.1, 1.0, 0.1)

    def test_negative_tolerance_rejected(self):
        with self.assertRaises(ValueError):
            select_k(RECORDS, -0.1, 1.0, 0.1)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m unittest growing_mdn.tests.test_selection -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'growing_mdn.selection'`.

- [ ] **Step 3: Implement `growing_mdn/selection.py`**

```python
"""Which component to split, and which K to keep. Validation data only."""
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributions as dist

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / 'base_mdn')):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from growing_mdn.model import num_components  # noqa: E402
from utils.mdn_distribution import decode_mdn_output  # noqa: E402

METRIC_KEYS = ('ravg_percent', 'rmin_percent', 's68_m2_per_s', 's95_m2_per_s')
MIN_MASS = 1e-8


@torch.no_grad()
def component_scores(model, X, y, device, batch_size):
    """Responsibility-weighted mean of -log N_k(y) per component.

    A high score means the component explains its assigned points poorly. The
    component index is a head index, not a physically consistent mode.
    """
    k = num_components(model)
    model.eval()
    mass = torch.zeros(k, dtype=torch.float64)
    weighted = torch.zeros(k, dtype=torch.float64)
    for start in range(0, len(X), batch_size):
        x = torch.as_tensor(X[start:start + batch_size], dtype=torch.float32, device=device)
        target = torch.as_tensor(y[start:start + batch_size], dtype=torch.float32, device=device)
        p = decode_mdn_output(model(x), k)
        log_n = dist.MultivariateNormal(p['mu'], p['covariance']).log_prob(target.unsqueeze(2))
        resp = torch.softmax(p['pi'].clamp_min(torch.finfo(p['pi'].dtype).tiny).log() + log_n, dim=-1)
        mass += resp.sum((0, 1)).double().cpu()
        weighted += (resp * -log_n).sum((0, 1)).double().cpu()
    scores = torch.full((k,), -math.inf, dtype=torch.float64)
    alive = mass > MIN_MASS
    scores[alive] = weighted[alive] / mass[alive]
    return scores.numpy()


def pick_split(scores):
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).any():
        raise ValueError('No component has a finite score; cannot choose a split')
    return int(np.argmax(scores))


def select_k(records, epsilon_nll, ravg_tolerance_pp, sharpness_tolerance_ratio):
    """Smallest K within tolerance of the best-NLL K on validation.

    ``records``: dicts with ``k``, ``validation_nll`` and ``METRIC_KEYS``.
    """
    if min(epsilon_nll, ravg_tolerance_pp, sharpness_tolerance_ratio) < 0:
        raise ValueError('Tolerances must be non-negative')
    for r in records:
        values = [r['validation_nll']] + [r[key] for key in METRIC_KEYS]
        if not all(np.isfinite(values)):
            raise ValueError(f'Non-finite validation values for K={r["k"]}')
    best = min(records, key=lambda r: r['validation_nll'])

    def acceptable(r):
        return (r['validation_nll'] <= best['validation_nll'] + epsilon_nll
                and r['ravg_percent'] >= best['ravg_percent'] - ravg_tolerance_pp
                and r['rmin_percent'] >= best['rmin_percent'] - ravg_tolerance_pp
                and r['s68_m2_per_s'] <= best['s68_m2_per_s'] + abs(best['s68_m2_per_s']) * sharpness_tolerance_ratio
                and r['s95_m2_per_s'] <= best['s95_m2_per_s'] + abs(best['s95_m2_per_s']) * sharpness_tolerance_ratio)

    candidates = sorted(r['k'] for r in records if acceptable(r))
    return {'selected_k': candidates[0], 'best_nll_k': best['k'], 'candidate_ks': candidates,
            'rule': {'epsilon_nll': epsilon_nll, 'ravg_tolerance_pp': ravg_tolerance_pp,
                     'sharpness_tolerance_ratio': sharpness_tolerance_ratio}}
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m unittest discover -s growing_mdn/tests -t . -v`
Expected: all tests in the three test files PASS. (If `test_dead_component_is_minus_inf` fails because comp1 still has mass > 1e-8, raise `mu1` to 200; the test is checking the dead-component rule, not the number 100.)

- [ ] **Step 5: Commit (only if authorized)**

```bash
git add growing_mdn/selection.py growing_mdn/tests/test_selection.py
git commit -m "feat(growing_mdn): add component scoring and K selection rule" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Base-hash guard, tracker subclass and configs

**Files:**
- Create: `growing_mdn/base_hashes.py`
- Create: `growing_mdn/artifacts.py`
- Create: `growing_mdn/configs/imptc/growing_peds_imptc.json`, `growing_mdn/configs/imptc/smoke_growing_peds_imptc.json`
- Create (generated): `growing_mdn/reports/BASE_SOURCE_HASHES.json`
- Test: `growing_mdn/tests/test_config.py`

**Interfaces:**
- Produces:
  - `base_hashes.compute() -> dict[str, str]`, `base_hashes.verify() -> None` (raises `RuntimeError` listing changed files), `python -m growing_mdn.base_hashes --write|--check`.
  - `artifacts.ARCH = 'growing_mdn_v1'`; `GrowingTracker(ExperimentTracker)` with: attributes `phase, current_k, epoch_in_phase, phase_best, phase_best_epoch, phase_complete`; methods `start_phase(phase, k, keep_best=False)`, `update_best(...)` (per-phase `best_k{K:02d}.pt`), `restore(saved)`, `log_growth(record)`, `log_phase(record)`; payload adds `architecture, phase, num_gaussians, epoch_in_phase, phase_best_nll, phase_best_epoch, phase_complete`; fixed-prediction npz adds `phase`, `num_gaussians`.
  - Config key `experiment_params.growth = {initial_k, k_max, phase0_epochs, phase_epochs, split_delta, zero_new_moments}` and `experiment_params.selection = {epsilon_nll, ravg_tolerance_pp, sharpness_tolerance_ratio}`.
  - `growing_mdn.train.check_config(cfg) -> growth dict` is introduced in Task 5; its tests live in `test_config.py` and are written here as failing tests first (they fail on import until Task 5, so this task's test file only covers JSON validity and the hash guard; the `check_config` tests are added at the start of Task 5).

- [ ] **Step 1: Write the failing test for the hash guard and configs**

Create `growing_mdn/tests/test_config.py`:

```python
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from growing_mdn import base_hashes

CONFIG_DIR = ROOT / 'growing_mdn/configs/imptc'


class ConfigFilesTest(unittest.TestCase):
    def load(self, name):
        return json.loads((CONFIG_DIR / name).read_text())

    def test_full_config_matches_spec(self):
        cfg = self.load('growing_peds_imptc.json')
        g = cfg['experiment_params']['growth']
        self.assertEqual((g['initial_k'], g['k_max'], g['phase0_epochs'], g['phase_epochs']), (1, 12, 400, 200))
        self.assertEqual(g['split_delta'], 0.05)
        self.assertEqual(cfg['train_params']['train_epochs'], 400 + 11 * 200)
        self.assertEqual(cfg['model_params']['num_gaussians'], 1)
        self.assertEqual(cfg['model_params']['mdn_parameterization'], {'mode': 'legacy'})
        self.assertEqual(cfg['train_params']['eval_data_reduction'], 1.0)
        self.assertEqual(cfg['train_params']['train_data_reduction'], 0.5)
        self.assertEqual(cfg['train_params']['batch_size'], 4096)
        self.assertEqual(cfg['train_params']['lr_default'], 1e-3)
        self.assertEqual(cfg['experiment_params']['seed'], 2024)
        self.assertEqual(set(cfg['experiment_params']['selection']),
                         {'epsilon_nll', 'ravg_tolerance_pp', 'sharpness_tolerance_ratio'})

    def test_smoke_config_is_small(self):
        cfg = self.load('smoke_growing_peds_imptc.json')
        g = cfg['experiment_params']['growth']
        total = g['phase0_epochs'] + (g['k_max'] - g['initial_k']) * g['phase_epochs']
        self.assertEqual(cfg['train_params']['train_epochs'], total)
        self.assertLessEqual(total, 6)

    def test_all_other_baseline_fields_unchanged(self):
        base = json.loads((ROOT / 'base_mdn/configs/imptc/default_peds_imptc.json').read_text())
        cfg = self.load('growing_peds_imptc.json')
        for key in ('paths', 'eval_metrics', 'test_params'):
            self.assertEqual(cfg[key], base[key])
        for key in ('delta_t', 'max_input_horizon', 'forecast_horizon', 'lstm_input_shape',
                    'lstm_hidden_size', 'lstm_num_layers', 'output_factor'):
            self.assertEqual(cfg['model_params'][key], base['model_params'][key])
        for key in ('lr_default', 'lr_start_factor', 'lr_end_factor', 'batch_size',
                    'randomize_train_data', 'dynamic_input_horizon'):
            self.assertEqual(cfg['train_params'][key], base['train_params'][key])


class HashGuardTest(unittest.TestCase):
    def test_current_base_matches_snapshot(self):
        base_hashes.verify()

    def test_detects_change(self):
        snapshot = base_hashes.compute()
        key = next(iter(snapshot))
        snapshot[key] = '0' * 64
        with self.assertRaises(RuntimeError):
            base_hashes.verify(expected=snapshot)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m unittest growing_mdn.tests.test_config -v`
Expected: FAIL (`ModuleNotFoundError: growing_mdn.base_hashes`).

- [ ] **Step 3: Implement `growing_mdn/base_hashes.py`, write the snapshot**

```python
"""Prove base_mdn is untouched: SHA256 snapshot recorded once, verified per run."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / 'growing_mdn/reports/BASE_SOURCE_HASHES.json'


def compute():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / 'base_mdn').rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts}


def verify(expected=None):
    expected = json.loads(SNAPSHOT.read_text()) if expected is None else expected
    actual = compute()
    changed = sorted(k for k in set(expected) | set(actual) if expected.get(k) != actual.get(k))
    if changed:
        raise RuntimeError(f'base_mdn differs from snapshot: {changed}')


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--write', action='store_true')
    mode.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.write:
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(json.dumps(compute(), indent=2, sort_keys=True) + '\n')
        print(f'wrote {SNAPSHOT}')
    else:
        verify()
        print('base_mdn unchanged')


if __name__ == '__main__':
    main()
```

Run: `.venv/bin/python -m growing_mdn.base_hashes --write`
Expected: prints `wrote .../growing_mdn/reports/BASE_SOURCE_HASHES.json`. (The snapshot records the *current* state of `base_mdn`, which the user already has in their working tree.)

- [ ] **Step 4: Generate the two configs from the baseline config**

Run:

```bash
.venv/bin/python - <<'EOF'
import copy, json
from pathlib import Path
root = Path('.')
base = json.loads((root / 'base_mdn/configs/imptc/default_peds_imptc.json').read_text())
out = root / 'growing_mdn/configs/imptc'
out.mkdir(parents=True, exist_ok=True)

def make(smoke):
    cfg = copy.deepcopy(base)
    cfg['model_params']['num_gaussians'] = 1
    cfg['model_params']['mdn_parameterization'] = {'mode': 'legacy'}
    growth = {'initial_k': 1, 'k_max': 3 if smoke else 12,
              'phase0_epochs': 2 if smoke else 400, 'phase_epochs': 1 if smoke else 200,
              'split_delta': 0.05, 'zero_new_moments': False}
    total = growth['phase0_epochs'] + (growth['k_max'] - growth['initial_k']) * growth['phase_epochs']
    cfg['train_params'].update(train_epochs=total, eval_epoch_step=total, eval_data_reduction=1.0)
    ep = cfg['experiment_params']
    ep.update(experiment_name='imptc_growing_mdn', best_selection='per_phase_full_validation_nll',
              growth=growth,
              selection={'epsilon_nll': 0.02, 'ravg_tolerance_pp': 0.5, 'sharpness_tolerance_ratio': 0.10},
              metric_every=0)
    if smoke:
        cfg['train_params'].update(batch_size=128, train_data_reduction=0.001)
        cfg['test_params'].update(num_samples=64, mesh_resolution=1.0)
        ep.update(checkpoint_epochs=[1, 2, 3, 4], checkpoint_every=0)
    return cfg

for name, smoke in (('growing_peds_imptc.json', False), ('smoke_growing_peds_imptc.json', True)):
    (out / name).write_text(json.dumps(make(smoke), indent=2) + '\n')
    print('wrote', out / name)
EOF
```

Expected: two files written. The selection tolerances (`epsilon_nll=0.02`, `ravg_tolerance_pp=0.5`, `sharpness_tolerance_ratio=0.10`) are proposals; they must be shown to the user and confirmed (or changed) **before any full run**, and must not change after test results are seen.

- [ ] **Step 5: Implement `growing_mdn/artifacts.py`**

```python
"""Tracker that records K, phase and growth events next to the base artifacts."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / 'base_mdn')):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from utils.experiment import ExperimentTracker  # noqa: E402

ARCH = 'growing_mdn_v1'


class GrowingTracker(ExperimentTracker):
    history_fields = ExperimentTracker.history_fields + ['phase', 'num_gaussians', 'epoch_in_phase']

    def __init__(self, *args, **kwargs):
        # Set before the base constructor, which may already touch tracker state.
        self.phase = 0
        self.current_k = None
        self.epoch_in_phase = 0
        self.phase_best = float('inf')
        self.phase_best_epoch = None
        self.phase_complete = False
        super().__init__(*args, **kwargs)

    def start_phase(self, phase, k, keep_best=False):
        self.phase, self.current_k = int(phase), int(k)
        if not keep_best:
            self.epoch_in_phase = 0
            self.phase_best = float('inf')
            self.phase_best_epoch = None
            self.phase_complete = False

    def update_best(self, epoch, validation_nll, model, optimizer, scheduler, history):
        """Per-phase best; the run-level best fields track the minimum over all phases."""
        if validation_nll >= self.phase_best:
            return False
        self.phase_best = float(validation_nll)
        self.phase_best_epoch = int(epoch)
        if validation_nll < self.best_validation_nll:
            self.best_validation_nll = float(validation_nll)
            self.best_epoch = int(epoch)
        self.save_checkpoint(f'best_k{self.current_k:02d}', epoch, model, optimizer, scheduler, history)
        self._write_manifest('running')
        return True

    def checkpoint_payload(self, epoch, model, optimizer, scheduler, history):
        payload = super().checkpoint_payload(epoch, model, optimizer, scheduler, history)
        payload.update(architecture=ARCH, phase=self.phase, num_gaussians=self.current_k,
                       epoch_in_phase=self.epoch_in_phase, phase_best_nll=self.phase_best,
                       phase_best_epoch=self.phase_best_epoch, phase_complete=self.phase_complete)
        return payload

    def restore(self, saved):
        self.phase, self.current_k = saved['phase'], saved['num_gaussians']
        self.epoch_in_phase = saved['epoch_in_phase']
        self.phase_best = saved['phase_best_nll']
        self.phase_best_epoch = saved['phase_best_epoch']
        self.phase_complete = saved['phase_complete']
        self.best_epoch = saved.get('best_epoch')
        self.best_validation_nll = saved.get('best_validation_nll', float('inf'))

    def capture_fixed_predictions(self, model, epoch, label=None):
        super().capture_fixed_predictions(model, epoch, label)
        path = self.prediction_dir / f'{label or f"epoch_{epoch:04d}"}.npz'
        with np.load(path) as data:
            payload = {key: data[key] for key in data.files}
        payload.update(phase=np.int64(self.phase), num_gaussians=np.int64(self.current_k))
        np.savez_compressed(path, **payload)

    def log_growth(self, record):
        with (self.run_dir / 'growth_history.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        self.event('component_split', record.get('epoch'), record)

    def log_phase(self, record):
        """phase_summary.json keyed by K; rewritten atomically so a rerun is idempotent."""
        path = self.run_dir / 'phase_summary.json'
        summary = json.loads(path.read_text()) if path.exists() else {'phases': {}}
        summary['phases'][str(record['k'])] = record
        self._write_json(path, summary)
        self.event('phase_completed', record.get('last_epoch'), {'k': record['k']})
```

- [ ] **Step 6: Run to verify pass**

Run: `.venv/bin/python -m unittest growing_mdn.tests.test_config -v`
Expected: all tests PASS. `.venv/bin/python -c "import growing_mdn.artifacts"` must import without error.

- [ ] **Step 7: Commit (only if authorized)**

```bash
git add growing_mdn/base_hashes.py growing_mdn/artifacts.py growing_mdn/configs growing_mdn/reports/BASE_SOURCE_HASHES.json growing_mdn/tests/test_config.py
git commit -m "feat(growing_mdn): add tracker, configs and base-hash guard" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Phase-loop trainer and smoke run

**Files:**
- Create: `growing_mdn/train.py`
- Test: extend `growing_mdn/tests/test_config.py` (config validation)
- Create (generated): `growing_mdn/reports/SMOKE.json`, `growing_mdn/reports/SMOKE.md` (Task 6 writes these)

**Interfaces:**
- Consumes: everything from Tasks 1-4.
- Produces:
  - `check_config(cfg) -> dict` (growth settings; raises `ValueError` on any config misuse listed in Review Focus)
  - `validation_nll(model, X, y, device, batch_size, k) -> float` (sample-weighted mean NLL)
  - `phase_epochs(growth, phase) -> int`
  - `run(config, run_id, resume=None, smoke=False, stop_after_phase=None, stop_after_epoch=None) -> Path` (run directory; the two `stop_after_*` options exist only so resume can be tested deterministically)
  - CLI: `python -m growing_mdn.train --run-id ID (--smoke | --full) [--config NAME] [--resume PATH] [--gpu 0] [--stop-after-phase N] [--stop-after-epoch N]`
  - Run artifacts: `history.csv` (global epoch, with `phase`, `num_gaussians`, `epoch_in_phase`), `growth_history.jsonl`, `phase_summary.json`, checkpoints `epoch_####`, `best_k##`, `phase_k##`, `last`, `final`, `fixed_samples/predictions/*.npz`, `metrics/epoch_####.json`.

- [ ] **Step 1: Write the failing config-validation tests**

Append to `growing_mdn/tests/test_config.py` (before `if __name__`):

```python
class CheckConfigTest(unittest.TestCase):
    def cfg(self, **overrides):
        import copy
        from types import SimpleNamespace
        raw = json.loads((CONFIG_DIR / 'growing_peds_imptc.json').read_text())
        for section, values in overrides.items():
            raw[section].update(values)
        return SimpleNamespace(model_params=raw['model_params'], train_params=raw['train_params'],
                               experiment_params=raw['experiment_params'])

    def test_valid_config(self):
        from growing_mdn.train import check_config
        self.assertEqual(check_config(self.cfg())['k_max'], 12)

    def test_rejects_misuse(self):
        from growing_mdn.train import check_config
        base = json.loads((CONFIG_DIR / 'growing_peds_imptc.json').read_text())['experiment_params']['growth']
        bad = [
            {'train_params': {'train_epochs': 2500}},
            {'train_params': {'eval_data_reduction': 0.5}},
            {'train_params': {'dynamic_input_horizon': True}},
            {'model_params': {'num_gaussians': 3}},
            {'model_params': {'mdn_parameterization': {'mode': 'paper'}}},
            {'experiment_params': {'growth': dict(base, k_max=1)}},
            {'experiment_params': {'growth': dict(base, split_delta=0.0)}},
        ]
        for overrides in bad:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    check_config(self.cfg(**overrides))
```

Run: `.venv/bin/python -m unittest growing_mdn.tests.test_config.CheckConfigTest -v`
Expected: FAIL (`ModuleNotFoundError`/`ImportError: check_config`). Note: the `paper` mode case relies on `resolve_parameterization` raising `ValueError`; `check_config` must call `resolve_parameterization` or compare to `{'mode': 'legacy'}` and raise `ValueError` itself.

- [ ] **Step 2: Implement `growing_mdn/train.py`**

```python
"""Split-and-grow MDN trainer. K grows from initial_k to k_max, one split per phase.

Smoke runs are limited to <= 6 epochs; full training requires --full and
explicit user approval (see AGENTS.md).
"""
import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / 'base_mdn')):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from utils.config_loader import ConfigLoader  # noqa: E402
from utils.data_loader import DataLoader  # noqa: E402
from utils.experiment import (capture_rng_state, restore_checkpoint, restore_rng_state,  # noqa: E402
                              set_global_seed, utc_now)
from utils.mdn_distribution import build_mdn_distribution  # noqa: E402
from eval import MDN_Forecaster  # noqa: E402
from growing_mdn import base_hashes  # noqa: E402
from growing_mdn.artifacts import ARCH, GrowingTracker  # noqa: E402
from growing_mdn.model import build_model, grow_model, num_components  # noqa: E402
from growing_mdn.optim_state import remap_adam_state  # noqa: E402
from growing_mdn.selection import component_scores, pick_split  # noqa: E402

GROWTH_KEYS = ('initial_k', 'k_max', 'phase0_epochs', 'phase_epochs', 'split_delta')


def check_config(cfg):
    if cfg.model_params.get('mdn_parameterization', {'mode': 'legacy'}) != {'mode': 'legacy'}:
        raise ValueError('Baseline covariance policy must stay legacy')
    if cfg.train_params['eval_data_reduction'] != 1 or cfg.train_params['dynamic_input_horizon']:
        raise ValueError('Use full deterministic validation and a fixed observation horizon')
    growth = cfg.experiment_params.get('growth', {})
    missing = [key for key in GROWTH_KEYS if key not in growth]
    if missing:
        raise ValueError(f'Missing growth settings: {missing}')
    if growth['k_max'] <= growth['initial_k'] or growth['initial_k'] < 1:
        raise ValueError('Need 1 <= initial_k < k_max')
    if not growth['split_delta'] > 0:
        raise ValueError('split_delta must be positive')
    if growth['initial_k'] != cfg.model_params['num_gaussians']:
        raise ValueError('model_params.num_gaussians must equal growth.initial_k')
    total = growth['phase0_epochs'] + (growth['k_max'] - growth['initial_k']) * growth['phase_epochs']
    if total != cfg.train_params['train_epochs']:
        raise ValueError(f'train_epochs {cfg.train_params["train_epochs"]} != phase total {total}')
    return growth


def phase_epochs(growth, phase):
    return growth['phase0_epochs'] if phase == 0 else growth['phase_epochs']


@torch.no_grad()
def validation_nll(model, X, y, device, batch_size, k):
    model.eval()
    total = 0.0
    for start in range(0, len(X), batch_size):
        stop = min(start + batch_size, len(X))
        raw = model(torch.as_tensor(X[start:stop], dtype=torch.float32, device=device))
        target = torch.as_tensor(y[start:stop], dtype=torch.float32, device=device)
        total += float(-build_mdn_distribution(raw, k).log_prob(target).mean()) * (stop - start)
    return total / len(X)


def make_optimizer(cfg, model, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train_params['lr_default'])
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=cfg.train_params['lr_start_factor'],
        end_factor=cfg.train_params['lr_end_factor'], total_iters=epochs)
    return optimizer, scheduler


def train_epoch(model, loader, optimizer, k, batch, device, tracker, epoch):
    model.train()
    X, y = loader.get_train_data()
    total = 0.0
    for start in range(0, len(X), batch):
        stop = min(start + batch, len(X))
        optimizer.zero_grad(set_to_none=True)
        inputs = torch.as_tensor(X[start:stop], dtype=torch.float32, device=device)
        target = torch.as_tensor(y[start:stop], dtype=torch.float32, device=device)
        try:
            raw = model(inputs)
            nll = -build_mdn_distribution(raw, k).log_prob(target).mean()
            if not torch.isfinite(nll):
                raise FloatingPointError('Non-finite loss')
            nll.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('Non-finite gradient')
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            torch.save({'epoch': epoch, 'start': start, 'X': inputs.cpu(), 'y': target.cpu(),
                        'error': str(exc), 'model_state_dict': model.state_dict(),
                        'rng_state': capture_rng_state()}, tracker.run_dir / 'failure.pt')
            tracker.diverged(epoch, epoch - 1)
            raise
        optimizer.step()
        total += float(nll.detach()) * (stop - start)
    return total / len(X), len(X)


def official_metrics(cfg, tracker, loader, model, device, epoch, smoke):
    """Legacy repository metrics; metric RNG must not disturb training RNG."""
    rng = capture_rng_state()
    try:
        set_global_seed(cfg.experiment_params['seed'])
        metric_loader = loader
        if smoke:
            metric_loader = copy.copy(loader)
            fixed = [a[tracker.fixed_indices] for a in loader.eval_data]
            metric_loader.get_eval_data = lambda: fixed
        evaluator = MDN_Forecaster(cfg, model, metric_loader, 'eval', logging.getLogger('growing'), device)
        return evaluator.evaluate(epoch=epoch)
    finally:
        restore_rng_state(rng)


def finish_phase(cfg, tracker, loader, model, optimizer, scheduler, history, device, epoch, smoke):
    """Restore the phase's best state, record official metrics, save the phase checkpoint."""
    k = num_components(model)
    best = torch.load(tracker.checkpoint_dir / f'best_k{k:02d}.pt', map_location=device, weights_only=False)
    model.load_state_dict(best['model_state_dict'])
    optimizer.load_state_dict(best['optimizer_state_dict'])
    scheduler.load_state_dict(best['scheduler_state_dict'])
    val = validation_nll(model, loader.eval_data[0], loader.eval_data[1], device,
                         cfg.test_params['batch_size'], k)
    metrics = official_metrics(cfg, tracker, loader, model, device, epoch, smoke)
    tracker.save_metrics(epoch, metrics, split='fixed_validation_smoke' if smoke else 'eval')
    tracker.phase_complete = True
    tracker.save_checkpoint(f'phase_k{k:02d}', epoch, model, optimizer, scheduler, history)
    tracker.log_phase({'k': k, 'phase': tracker.phase, 'best_epoch': best['epoch'], 'last_epoch': epoch,
                       'validation_nll': val, 'metrics': metrics, 'smoke_metric_subset': bool(smoke),
                       'parameter_count': sum(p.numel() for p in model.parameters()),
                       'timestamp_utc': utc_now()})


def split_step(cfg, growth, tracker, loader, model, optimizer, device, epoch):
    """Split the worst-fitting component; return the (K+1) model, optimizer, scheduler."""
    k = num_components(model)
    batch = cfg.test_params['batch_size']
    scores = component_scores(model, loader.train_data[0], loader.train_data[1], device, batch)
    source = pick_split(scores)
    nll_before = validation_nll(model, loader.eval_data[0], loader.eval_data[1], device, batch, k)
    new_model, rows = grow_model(model, source, growth['split_delta'])
    nll_after = validation_nll(new_model, loader.eval_data[0], loader.eval_data[1], device, batch, k + 1)
    new_opt, new_sched = make_optimizer(cfg, new_model, growth['phase_epochs'])
    remap_adam_state(optimizer, new_opt, model, new_model, rows,
                     zero_new_rows=bool(growth.get('zero_new_moments', False)))
    cfg.model_params['num_gaussians'] = k + 1
    tracker.log_growth({'epoch': epoch, 'k_before': k, 'k_after': k + 1, 'split_component': source,
                        'component_scores': [None if not np.isfinite(s) else float(s) for s in scores],
                        'validation_nll_before': nll_before, 'validation_nll_after': nll_after,
                        'split_delta': growth['split_delta'], 'timestamp_utc': utc_now()})
    return new_model, new_opt, new_sched


def run(config, run_id, resume=None, smoke=False, stop_after_phase=None, stop_after_epoch=None):
    cfg = ConfigLoader(str(config), 'imptc', False, False, Path(config).stem, 'growing_mdn', 'training')
    growth = check_config(cfg)
    if smoke and cfg.train_params['train_epochs'] > 6:
        raise ValueError('Smoke is limited to <= 6 epochs')
    base_hashes.verify()
    run_dir = Path(cfg.result_path) / 'runs' / run_id
    if run_dir.exists() and not resume:
        raise FileExistsError(run_dir)
    set_global_seed(cfg.experiment_params['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = DataLoader(cfg)
    loader.load_train_data()
    loader.load_eval_data()
    n_phases = growth['k_max'] - growth['initial_k'] + 1
    phase, epoch, epoch_in_phase, k, saved = 0, 0, 0, growth['initial_k'], None
    if resume:
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        if saved.get('architecture') != ARCH or saved['run_id'] != run_id:
            raise ValueError('Incompatible resume checkpoint')
        for key in ('train_params', 'experiment_params'):
            if saved['resolved_config'][key] != getattr(cfg, key):
                raise ValueError(f'Resume config differs: {key}')
        phase, epoch, epoch_in_phase, k = (saved['phase'], saved['epoch'],
                                           saved['epoch_in_phase'], saved['num_gaussians'])
    cfg.model_params['num_gaussians'] = k
    model = build_model(cfg.model_params, k, device)
    optimizer, scheduler = make_optimizer(cfg, model, phase_epochs(growth, phase))
    tracker = GrowingTracker(cfg, loader, device, run_id)
    history = []
    if resume:
        saved = restore_checkpoint(resume, model, optimizer, scheduler, map_location=device)
        if not loader.load_state_dict(saved['data_loader_state']):
            raise ValueError('Checkpoint has no data-loader state')
        history = saved['train_history']
        tracker.restore(saved)
        tracker.event('run_resumed', epoch, {'checkpoint': str(resume)})
    batch = cfg.train_params['batch_size']
    keep_best = bool(resume) and not saved['phase_complete']
    if resume and saved['phase_complete']:
        model, optimizer, scheduler = split_step(cfg, growth, tracker, loader, model, optimizer, device, epoch)
        phase, epoch_in_phase = phase + 1, 0
    while True:
        k = num_components(model)
        tracker.start_phase(phase, k, keep_best=keep_best)
        keep_best = False
        n_epochs = phase_epochs(growth, phase)
        for ep in range(epoch_in_phase + 1, n_epochs + 1):
            started = time.time()
            epoch += 1
            lr = optimizer.param_groups[0]['lr']
            train_nll, n_train = train_epoch(model, loader, optimizer, k, batch, device, tracker, epoch)
            scheduler.step()
            val = validation_nll(model, loader.eval_data[0], loader.eval_data[1], device,
                                 cfg.test_params['batch_size'], k)
            if not np.isfinite(val):
                tracker.diverged(epoch, epoch - 1)
                raise FloatingPointError('Non-finite validation NLL')
            tracker.epoch_in_phase = ep
            row = {'run_id': run_id, 'epoch': epoch, 'train_nll': train_nll, 'validation_nll': val,
                   'learning_rate': lr, 'duration_seconds': time.time() - started,
                   'train_sample_count': n_train, 'validation_sample_count': len(loader.eval_data[0]),
                   'gpu_peak_memory_bytes': int(torch.cuda.max_memory_allocated(device)) if device.type == 'cuda' else 0,
                   'finite': True, 'timestamp_utc': utc_now(),
                   'phase': phase, 'num_gaussians': k, 'epoch_in_phase': ep}
            history.append(row)
            best = tracker.update_best(epoch, val, model, optimizer, scheduler, history)
            periodic = tracker.should_checkpoint(epoch)
            if periodic:
                tracker.save_checkpoint(f'epoch_{epoch:04d}', epoch, model, optimizer, scheduler, history)
            row.update(is_best=best, checkpoint_saved=bool(periodic or best), full_metrics_evaluated=False)
            tracker.log_epoch(row)
            tracker.save_checkpoint('last', epoch, model, optimizer, scheduler, history, capture_predictions=False)
            print(f'Epoch {epoch} (K={k}, {ep}/{n_epochs}): train NLL={train_nll:.5f}, val NLL={val:.5f}', flush=True)
            if stop_after_epoch is not None and epoch == stop_after_epoch:
                tracker._write_manifest('stopped_for_test')
                return run_dir
        finish_phase(cfg, tracker, loader, model, optimizer, scheduler, history, device, epoch, smoke)
        if stop_after_phase is not None and phase == stop_after_phase:
            tracker._write_manifest('stopped_for_test')
            return run_dir
        if phase == n_phases - 1:
            break
        model, optimizer, scheduler = split_step(cfg, growth, tracker, loader, model, optimizer, device, epoch)
        phase, epoch_in_phase = phase + 1, 0
    tracker.save_checkpoint('final', epoch, model, optimizer, scheduler, history)
    tracker.complete(epoch)
    base_hashes.verify()
    return run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--resume')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--stop-after-phase', type=int)
    parser.add_argument('--stop-after-epoch', type=int)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--full', action='store_true')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    name = args.config or ('smoke_growing_peds_imptc.json' if args.smoke else 'growing_peds_imptc.json')
    config = ROOT / 'growing_mdn/configs/imptc' / name
    if args.smoke and json.loads(config.read_text())['train_params']['train_epochs'] > 6:
        raise ValueError('Smoke is limited to <= 6 epochs')
    print(run(config, args.run_id, args.resume, args.smoke, args.stop_after_phase, args.stop_after_epoch))


if __name__ == '__main__':
    main()
```

Note for the implementer: `_write_manifest('stopped_for_test')` (used by `--stop-after-phase` and `--stop-after-epoch`) writes a non-standard status only for the resume tests in Task 6; a later resume overwrites it with `running`.

- [ ] **Step 3: Run the config-validation tests**

Run: `.venv/bin/python -m unittest discover -s growing_mdn/tests -t . -v`
Expected: all tests PASS (including `CheckConfigTest`).

- [ ] **Step 4: Smoke run (K 1 -> 3, 4 epochs, tiny train subset, CPU-safe)**

Run: `.venv/bin/python -m growing_mdn.train --smoke --run-id growing_smoke_seed2024`
Expected: prints 4 `Epoch` lines with `K=1,1,2,3`, finishes and prints the run directory under `results/trained_models/growing_mdn/imptc/smoke_growing_peds_imptc/runs/growing_smoke_seed2024`. If it fails, debug with superpowers:systematic-debugging; do not patch around the failure.

- [ ] **Step 5: Quick artifact check**

Run:

```bash
RUN=results/trained_models/growing_mdn/imptc/smoke_growing_peds_imptc/runs/growing_smoke_seed2024
cat $RUN/history.csv | cut -d, -f1-5,15-17 | head
cat $RUN/growth_history.jsonl
ls $RUN/checkpoints $RUN/fixed_samples/predictions
```

Expected: 4 history rows with `num_gaussians` 1,1,2,3; 2 growth records (`k_before` 1 then 2); checkpoints include `best_k01..03`, `phase_k01..03`, `last`, `final`; fixed predictions include `phase_k01..03.npz`.

- [ ] **Step 6: Commit (only if authorized)**

```bash
git add growing_mdn/train.py growing_mdn/tests/test_config.py
git commit -m "feat(growing_mdn): add phase-loop trainer" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Resume check, smoke review, K-selection CLI and test evaluator

**Files:**
- Create: `growing_mdn/review_smoke.py`
- Create: `growing_mdn/select_k.py`
- Create: `growing_mdn/evaluate.py`
- Test: `growing_mdn/tests/test_select_cli.py`

**Interfaces:**
- Consumes: `run`, `validation_nll`, `phase_epochs` from `growing_mdn.train`; `select_k`, `METRIC_KEYS` from `growing_mdn.selection`; `build_model` from `growing_mdn.model`.
- Produces:
  - `select_k.build_records(summary: dict) -> list[dict]` and CLI `python -m growing_mdn.select_k --run-id ID [--config NAME]` writing `<run>/selection.json` (validation only).
  - `evaluate.py` CLI: `python -m growing_mdn.evaluate --run-id ID (--limit N | --official) [--k K] [--config NAME] [--gpu 0]` writing `<run>/testing/evaluation_k##[_limited].json`.
  - `review_smoke.py` writing `growing_mdn/reports/SMOKE.json` and `SMOKE.md`.

- [ ] **Step 1: Write the failing test for `build_records`**

Create `growing_mdn/tests/test_select_cli.py`:

```python
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from growing_mdn.select_k import build_records
from growing_mdn.selection import select_k

METRICS = {'ravg_percent': 97.0, 'rmin_percent': 94.0, 's68_m2_per_s': 1.0, 's95_m2_per_s': 5.0}


class BuildRecordsTest(unittest.TestCase):
    def test_records_sorted_and_complete(self):
        summary = {'phases': {
            '2': {'k': 2, 'validation_nll': -0.9, 'metrics': dict(METRICS)},
            '1': {'k': 1, 'validation_nll': -0.3, 'metrics': dict(METRICS)}}}
        records = build_records(summary)
        self.assertEqual([r['k'] for r in records], [1, 2])
        self.assertEqual(select_k(records, 0.0, 0.0, 0.0)['selected_k'], 2)

    def test_missing_metric_rejected(self):
        summary = {'phases': {'1': {'k': 1, 'validation_nll': -0.3, 'metrics': {'ravg_percent': 97.0}}}}
        with self.assertRaises(KeyError):
            build_records(summary)


if __name__ == '__main__':
    unittest.main()
```

Run: `.venv/bin/python -m unittest growing_mdn.tests.test_select_cli -v`
Expected: FAIL (`ModuleNotFoundError: growing_mdn.select_k`).

- [ ] **Step 2: Implement `growing_mdn/select_k.py`**

```python
"""Choose K from per-phase validation results. Never reads test data."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / 'base_mdn')):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from utils.config_loader import ConfigLoader  # noqa: E402
from growing_mdn.selection import METRIC_KEYS, select_k  # noqa: E402


def build_records(summary):
    records = []
    for phase in sorted(summary['phases'].values(), key=lambda p: p['k']):
        records.append({'k': phase['k'], 'validation_nll': phase['validation_nll'],
                        **{key: phase['metrics'][key] for key in METRIC_KEYS}})
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--config', default='growing_peds_imptc.json')
    args = parser.parse_args()
    path = ROOT / 'growing_mdn/configs/imptc' / args.config
    cfg = ConfigLoader(str(path), 'imptc', False, False, path.stem, 'growing_mdn', 'training')
    run = Path(cfg.result_path) / 'runs' / args.run_id
    summary = json.loads((run / 'phase_summary.json').read_text())
    if any(p.get('smoke_metric_subset') for p in summary['phases'].values()):
        raise ValueError('Smoke metrics come from 8 fixed samples; do not select K from them')
    rule = cfg.experiment_params['selection']
    result = select_k(build_records(summary), rule['epsilon_nll'], rule['ravg_tolerance_pp'],
                      rule['sharpness_tolerance_ratio'])
    result.update(run_id=args.run_id, split='validation')
    (run / 'selection.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
```

- [ ] **Step 3: Run the unit test**

Run: `.venv/bin/python -m unittest discover -s growing_mdn/tests -t . -v`
Expected: all PASS.

- [ ] **Step 4: Implement `growing_mdn/evaluate.py`**

```python
"""Evaluate the selected K on the test split with the legacy repository metrics.

Run only on explicit user request, once, after the selection rule is frozen.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / 'base_mdn')):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from utils.config_loader import ConfigLoader  # noqa: E402
from utils.data_loader import DataLoader  # noqa: E402
from utils.experiment import set_global_seed  # noqa: E402
from eval import MDN_Forecaster  # noqa: E402
from growing_mdn.artifacts import ARCH  # noqa: E402
from growing_mdn.model import build_model  # noqa: E402
from growing_mdn.train import validation_nll  # noqa: E402


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='growing_peds_imptc.json')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--k', type=int)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--official', action='store_true')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    if args.official == (args.limit is not None):
        raise ValueError('Pass exactly one of --official (full test split) or --limit N')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_global_seed(2024)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = ROOT / 'growing_mdn/configs/imptc' / args.config
    cfg = ConfigLoader(str(path), 'imptc', False, False, path.stem, 'growing_mdn', 'testing')
    run = Path(cfg.result_path) / 'runs' / args.run_id
    k = args.k or json.loads((run / 'selection.json').read_text())['selected_k']
    checkpoint = torch.load(run / f'checkpoints/phase_k{k:02d}.pt', map_location=device, weights_only=False)
    if checkpoint.get('architecture') != ARCH or checkpoint['num_gaussians'] != k:
        raise ValueError(f'Checkpoint is not a K={k} growing_mdn phase checkpoint')
    cfg.model_params['num_gaussians'] = k
    model = build_model(cfg.model_params, k, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    loader = DataLoader(cfg)
    loader.load_test_data()
    X, y = loader.get_test_data()[:2]
    n = len(X) if args.limit is None else min(args.limit, len(X))
    result = {'run_id': args.run_id, 'k': k, 'epoch': checkpoint['epoch'],
              'split': 'test' if args.official else 'test_limited', 'sample_count': n, 'seed': 2024,
              'mdn_parameterization': {'mode': 'legacy'}, 'evaluator_version': 'base_mdn_legacy_bins',
              'test_params': cfg.test_params, 'parameter_count': sum(p.numel() for p in model.parameters()),
              'test_nll': validation_nll(model, X[:n], y[:n], device, cfg.test_params['batch_size'], k)}
    if args.official:
        cfg.testing_path = str(run / f'testing/official_k{k:02d}')
        Path(cfg.testing_path).mkdir(parents=True, exist_ok=True)
        set_global_seed(2024)
        forecaster = MDN_Forecaster(cfg, model, loader, 'testing', logging.getLogger('growing'), device)
        result['official_metrics'] = forecaster.evaluate(epoch=checkpoint['epoch'])
    destination = run / 'testing'
    destination.mkdir(exist_ok=True)
    out = destination / (f'evaluation_k{k:02d}.json' if args.official else f'evaluation_k{k:02d}_limited.json')
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(out)


if __name__ == '__main__':
    main()
```

Smoke-check without touching the full test split:
Run: `.venv/bin/python -m growing_mdn.evaluate --config smoke_growing_peds_imptc.json --run-id growing_smoke_seed2024 --k 3 --limit 16`
Expected: writes `.../testing/evaluation_k03_limited.json` with finite `test_nll`. (This only checks the code path on 16 samples; it is not a model-quality result.)

- [ ] **Step 5: Resume across a phase boundary and mid-phase (exact continuation check)**

Run on CPU for determinism:

```bash
.venv/bin/python -m growing_mdn.train --smoke --gpu -1 --run-id resume_ref
.venv/bin/python -m growing_mdn.train --smoke --gpu -1 --run-id resume_cut --stop-after-phase 0
BASE=results/trained_models/growing_mdn/imptc/smoke_growing_peds_imptc/runs
.venv/bin/python -m growing_mdn.train --smoke --gpu -1 --run-id resume_cut --resume $BASE/resume_cut/checkpoints/phase_k01.pt
.venv/bin/python - <<'EOF'
import csv
base = 'results/trained_models/growing_mdn/imptc/smoke_growing_peds_imptc/runs/'
def rows(run):
    return [(int(r['epoch']), int(r['num_gaussians']), float(r['validation_nll']))
            for r in csv.DictReader(open(base + run + '/history.csv'))]
ref, cut = rows('resume_ref'), rows('resume_cut')
assert [r[:2] for r in ref] == [r[:2] for r in cut], (ref, cut)
for a, b in zip(ref, cut):
    assert abs(a[2] - b[2]) < 1e-6, (a, b)
print('phase-boundary resume reproduces the uninterrupted run:', len(cut), 'epochs')
EOF
```

Expected: last line prints `... 4 epochs`.

Mid-phase resume (from `last.pt`). The smoke schedule is epoch 1-2 = phase 0 (K=1), epoch 3 = K=2, epoch 4 = K=3. Check two cut points, each compared with `resume_ref`:

```bash
BASE=results/trained_models/growing_mdn/imptc/smoke_growing_peds_imptc/runs
for CUT in 1 2; do
  .venv/bin/python -m growing_mdn.train --smoke --gpu -1 --run-id resume_e$CUT --stop-after-epoch $CUT
  .venv/bin/python -m growing_mdn.train --smoke --gpu -1 --run-id resume_e$CUT --resume $BASE/resume_e$CUT/checkpoints/last.pt
done
.venv/bin/python - <<'EOF'
import csv
base = 'results/trained_models/growing_mdn/imptc/smoke_growing_peds_imptc/runs/'
def rows(run):
    return [(int(r['epoch']), int(r['num_gaussians']), float(r['validation_nll']))
            for r in csv.DictReader(open(base + run + '/history.csv'))]
ref = rows('resume_ref')
for run in ('resume_e1', 'resume_e2'):
    cut = rows(run)
    assert [r[:2] for r in ref] == [r[:2] for r in cut], (run, ref, cut)
    for a, b in zip(ref, cut):
        assert abs(a[2] - b[2]) < 1e-6, (run, a, b)
    print(run, 'reproduces the uninterrupted run:', len(cut), 'epochs')
EOF
```

Expected: both lines print `reproduces the uninterrupted run: 4 epochs`. Cut 1 is a true mid-phase resume (one epoch of phase 0 remaining). Cut 2 is the case where all epochs of the phase are done but the phase-end work (restore best state, official metrics, `phase_k01.pt`, split) has not run; resume must execute it and continue. Also check that `phase_summary.json` of `resume_e2` has exactly one entry per K (no duplicates) and `growth_history.jsonl` has exactly 2 lines. If either cut fails, debug the cause with superpowers:systematic-debugging; do not weaken the comparison.

- [ ] **Step 6: Implement `growing_mdn/review_smoke.py`**

```python
"""Review smoke artifacts: pipeline integrity only, not model quality."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'base_mdn'))
from growing_mdn import base_hashes  # noqa: E402
from growing_mdn.artifacts import ARCH  # noqa: E402
from growing_mdn.selection import METRIC_KEYS  # noqa: E402

CONFIG = ROOT / 'growing_mdn/configs/imptc/smoke_growing_peds_imptc.json'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='growing_smoke_seed2024')
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text())
    g = cfg['experiment_params']['growth']
    ks = list(range(g['initial_k'], g['k_max'] + 1))
    expected_k_per_epoch = [g['initial_k']] * g['phase0_epochs'] + [k for k in ks[1:] for _ in range(g['phase_epochs'])]
    run = ROOT / 'results/trained_models/growing_mdn/imptc/smoke_growing_peds_imptc/runs' / args.run_id

    base_hashes.verify()
    rows = list(csv.DictReader((run / 'history.csv').open()))
    assert [int(r['epoch']) for r in rows] == list(range(1, len(expected_k_per_epoch) + 1))
    assert [int(r['num_gaussians']) for r in rows] == expected_k_per_epoch
    assert all(np.isfinite(float(r['train_nll'])) and np.isfinite(float(r['validation_nll'])) for r in rows)

    growth = [json.loads(line) for line in (run / 'growth_history.jsonl').read_text().splitlines()]
    assert [(r['k_before'], r['k_after']) for r in growth] == [(k, k + 1) for k in ks[:-1]]
    assert all(abs(r['validation_nll_after'] - r['validation_nll_before']) < 5e-3 for r in growth)

    summary = json.loads((run / 'phase_summary.json').read_text())['phases']
    assert sorted(int(k) for k in summary) == ks
    for k in ks:
        item = summary[str(k)]
        assert np.isfinite(item['validation_nll']) and item['smoke_metric_subset'] is True
        for key in METRIC_KEYS:
            assert np.isfinite(item['metrics'][key]), (k, key)

    fixed = np.load(run / 'fixed_samples/inputs.npz')
    assert len(fixed['sample_ids']) == 8
    for k in ks:
        for name in (f'best_k{k:02d}', f'phase_k{k:02d}'):
            cp = torch.load(run / 'checkpoints' / f'{name}.pt', map_location='cpu', weights_only=False)
            assert cp['architecture'] == ARCH and cp['num_gaussians'] == k
            assert 'rng_state' in cp and 'data_loader_state' in cp and cp['optimizer_state_dict']['state']
        pred = np.load(run / 'fixed_samples/predictions' / f'phase_k{k:02d}.npz')
        np.testing.assert_array_equal(pred['sample_ids'], fixed['sample_ids'])
        assert pred['mu'].shape == (8, 48, k, 2) and pred['pi'].shape == (8, 48, k)
        assert int(pred['num_gaussians']) == k
        np.testing.assert_allclose(pred['pi'].sum(-1), 1, atol=1e-6)
        for key in ('raw_output', 'sigma', 'rho', 'covariance'):
            assert np.isfinite(pred[key]).all()
    for name in ('last', 'final'):
        assert (run / 'checkpoints' / f'{name}.pt').exists()

    report = {'status': 'passed', 'run_id': args.run_id, 'k_sequence': ks, 'epochs': len(rows),
              'fixed_samples': 8, 'base_source_hashes_unchanged': True,
              'metric_scope': '8 fixed validation samples, MC64, grid 1 m; not full validation metrics',
              'full_training_started': False,
              'quality_conclusion': 'None; a few-epoch smoke run says nothing about the best K'}
    out = ROOT / 'growing_mdn/reports'
    out.mkdir(exist_ok=True)
    (out / 'SMOKE.json').write_text(json.dumps(report, indent=2) + '\n')
    (out / 'SMOKE.md').write_text(
        '# Smoke review: growing MDN\n\n'
        f'Passed. K sequence {ks}, {len(rows)} epochs, 8 fixed validation samples, base_mdn hashes unchanged. '
        'Metrics come from the 8 fixed samples only and are not full validation results. '
        'No conclusion about the best K and no full training was started.\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
```

Run: `.venv/bin/python growing_mdn/review_smoke.py`
Expected: prints the report with `"status": "passed"`. Fix any assertion failure at its cause; do not weaken the assertion.

- [ ] **Step 7: Commit (only if authorized)**

```bash
git add growing_mdn/review_smoke.py growing_mdn/select_k.py growing_mdn/evaluate.py growing_mdn/tests/test_select_cli.py growing_mdn/reports/SMOKE.json growing_mdn/reports/SMOKE.md
git commit -m "feat(growing_mdn): add K selection, test evaluator and smoke review" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Documentation and final verification

**Files:**
- Create: `growing_mdn/README.md`
- Modify: `MIDTERM_NOTES.md` (append one section at the end)

- [ ] **Step 1: Write `growing_mdn/README.md`**

Content (write it in Vietnamese to match the repo's notes; keep these facts): purpose and the claim boundaries below; layout of files; how to run unit tests (`.venv/bin/python -m unittest discover -s growing_mdn/tests -t .`), smoke (`python -m growing_mdn.train --smoke --run-id ...`), review (`python growing_mdn/review_smoke.py`), full run command (`python -m growing_mdn.train --full --run-id ...`, only after explicit approval), selection (`python -m growing_mdn.select_k --run-id ...`), test evaluation (`python -m growing_mdn.evaluate --run-id ... --official`, once, after the rule is frozen).

Claim boundaries to state explicitly:
- K is selected on validation with a penalized rule; it is not a proof of a global optimum and depends on epsilon, the schedule and the seed.
- "Component k" is a head index, not a physically consistent mode across samples or timesteps.
- Deviations from baseline: LinearLR restarts in each phase; selection uses full validation NLL (baseline used reduction 0.5); nested warm-started models see more epochs on shared weights, so compare against K=3/K=8 by total compute.
- Several seeds are required before claiming K differences.

- [ ] **Step 2: Append a section to `MIDTERM_NOTES.md`**

Add at the end a section "Variant học K: growing_mdn — 2026-10-04" summarizing: design spec and plan paths, what was implemented, what was verified (unit tests, smoke, hash guard), and that no full training has been run. Do not state any result about K.

- [ ] **Step 3: Final verification**

Run, and read the output before claiming anything:

```bash
.venv/bin/python -m unittest discover -s growing_mdn/tests -t . -v
.venv/bin/python -m growing_mdn.base_hashes --check
git status --short base_mdn
```

Expected: all unit tests PASS; `base_mdn unchanged`; `git status --short base_mdn` shows only the changes that were already present before this work (the pre-existing deletions of `compare_m1_m3.py`, `create_demo2_horizon_uncertainty.py`, `visualize_experiment.py`), nothing new.

- [ ] **Step 4: Commit (only if authorized)**

```bash
git add growing_mdn/README.md MIDTERM_NOTES.md
git commit -m "docs(growing_mdn): add README and midterm notes" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 5: Stop and report**

Report to the user: tests/smoke results, the selection tolerances awaiting confirmation (`epsilon_nll=0.02`, `ravg_tolerance_pp=0.5`, `sharpness_tolerance_ratio=0.10`), and ask for explicit approval before any full run (2600 epochs, K 1 -> 12).
