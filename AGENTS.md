# Project Working Instructions

## Project purpose

This repository supports a three-student midterm project about multivariate time-series modelling using a Gaussian Mixture Model combined with a sequence model.

The selected application is probabilistic human trajectory prediction based on the paper **Reliable Probabilistic Human Trajectory Prediction for Autonomous Applications** (ECCV Workshop 2024) and its official implementation.

The primary experimental flow is:

```text
IMPTC past trajectory
-> LSTM
-> Mixture Density Network
-> pi, mu, Sigma
-> 2D Gaussian mixture distributions
-> probabilistic future trajectory
```

## Current priorities

1. Understand the paper and actual implementation.
2. Reproduce the official IMPTC baseline as closely as practical.
3. Preserve sufficient training artifacts for analysis and presentation.
4. Produce theoretically correct explanations and visualizations.
5. Consider supplementary experiments only after establishing the baseline.

## Baseline preservation rules

- Read the relevant source code before proposing or implementing a change.
- Do not rewrite the repository unnecessarily.
- Do not change the architecture, loss, optimizer, learning rate, number of Gaussian components, preprocessing, official metrics, or important hyperparameters unless explicitly requested.
- Do not invent metrics or silently substitute similar metrics.
- Do not assume the MDN output format; derive it from the implementation.
- Do not assume uncertainty increases with forecast horizon; visualize actual outputs.
- Keep additions for logging, checkpointing, and visualization modular and minimally invasive.
- Explain the reason and expected effect of every material change.
- Do not start a full training run until logging, checkpointing, and fixed-sample capture have been reviewed.

## Experimental data

- Primary dataset: IMPTC.
- The input is a multivariate temporal sequence representing an observed pedestrian trajectory.
- The target is the future 2D pedestrian trajectory.
- Preserve the repository's preprocessing and human-ego-centric coordinate conventions for the baseline.

## Required training artifacts

Before full training, ensure the pipeline can preserve:

- epoch;
- training NLL;
- validation NLL;
- learning rate;
- official repository/paper metrics;
- periodic, best, and final checkpoints;
- fixed validation/test sample identifiers;
- observed trajectories;
- ground-truth future trajectories;
- predicted trajectories;
- raw or decoded MDN parameters.

The implementation parameterizes every bivariate Gaussian using:

- mixture weight `alpha` / `pi`;
- mean `mu_x`, `mu_y`;
- standard deviations `sigma_x`, `sigma_y`;
- correlation `rho`.

When a covariance matrix is needed, construct it as:

```text
Sigma = [[sigma_x^2,              rho * sigma_x * sigma_y],
         [rho * sigma_x * sigma_y,              sigma_y^2]]
```

Verify this against the current source before using it in an experiment or explanation.

## Fixed-sample policy

- Select approximately 5-10 validation/test samples before full training.
- Persist their stable identifiers or indices.
- Use exactly the same observed trajectory and ground truth at every selected checkpoint.
- Only the model prediction and its probability distribution may change between epochs.

Candidate checkpoints include epoch 1, 5, 10, later periodic checkpoints, best, and final. Final checkpoint scheduling must be chosen based on the actual baseline duration and configuration.

## Required visualizations

1. Training and validation NLL versus epoch.
2. Official evaluation metrics versus epoch.
3. Observed trajectory, future ground truth, and prediction in x-y coordinates.
4. GMM components at a selected future timestep, showing weights, means, covariance ellipses or contours, and ground truth.
5. Predicted uncertainty at multiple future timesteps using actual model outputs.
6. The same fixed sample across multiple checkpoints.

After training, include representative good, difficult/uncertain, and failure cases. Do not present only favorable samples.

## Interpretation constraints

- Clearly distinguish the LSTM sequence encoder, MDN parameter head, and GMM output distribution.
- Treat the output according to the code: a bivariate mixture distribution at each forecast timestep.
- Do not claim that the implementation defines a single joint mixture over the complete future trajectory unless the code proves that relationship.
- Use reliability and sharpness to discuss uncertainty quality, alongside the official displacement metrics.
- Separate observations supported by results from hypotheses or qualitative interpretations.

## Repository documentation

- Preserve the upstream `README.md`.
- Use `README_LOCAL.md` for local environment and execution instructions.
- Use `MIDTERM_NOTES.md` for academic scope, experiment planning, and presentation notes.

## Change and execution policy

- A request to inspect, explain, or plan does not authorize code changes or training.
- Do not train a model unless explicitly requested.
- Before a long run, perform a small smoke test and verify that all required artifacts are being written correctly.
- Report missing datasets, checkpoints, or ambiguous implementation details instead of guessing.

