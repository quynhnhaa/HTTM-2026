# Smoke review: sparsemax MDN

Passed. 3 CPU epochs, 5 fixed-sample prediction files (8 samples, pi shape (8, 48, 8)), checkpoints epoch_0001, epoch_0002, epoch_0003, best, last, final with architecture `sparsemax_mdn_v1`, official metrics finite for 7 keys, base_mdn hashes unchanged before and after.

Finding: NO exactly-zero pi in any smoke prediction (model still dense after a few epochs); zero handling is covered by unit tests, not by this smoke run.

Metrics come from reduced validation data and say nothing about quality. No full training was started.
