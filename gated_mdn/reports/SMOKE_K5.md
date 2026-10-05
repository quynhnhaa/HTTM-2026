# Smoke review: gated MDN

Passed 4 unit tests: true-zero gates and open-probability formula, unchanged legacy covariance/normalized pi, compact distribution equivalence, finite gate gradients and exact stochastic optimizer/RNG continuation.

Real IMPTC integration: 3 epochs, 189 train samples/epoch, full 19,148 validation NLL, same 8 fixed validation IDs. All seven displacement/reliability/sharpness/ASAEE metric branches ran on those 8 samples using MC64 and grid1m; these are **not full validation metrics**. Evaluator limited test16 and compact checkpoint export passed. Periodic/best/last/final retain gate state, optimizer/scheduler/RNG/loader ordering. NLL and penalized objective are separate. Base source hashes unchanged.

Gate parameters updated, but deterministic active K remains5 after smoke. Expected K≈4.799 is an expectation, not a learned integer K. No quality improvement claim and no full training started. Detailed values in SMOKE.json.
