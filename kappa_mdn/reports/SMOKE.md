# Smoke review: kappa MDN

Passed. 3 CPU epochs, 5 fixed-sample prediction files (8 samples), checkpoints epoch_0001, epoch_0002, epoch_0003, best, last, final with architecture `kappa_mdn_v2`, tau followed the declared schedule, the penalty/NLL split is consistent, kappa moved by up to 0.1137 from its initial value, optimizer lr ratio 10.0 for kappa, exact-zero weights above round(kappa), official metrics finite for 7 keys, base_mdn hashes unchanged before and after.

Metrics come from reduced validation data and say nothing about quality. No full training was started.
