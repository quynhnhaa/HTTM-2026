# Smoke review: growing MDN

Passed. K sequence [1, 2, 3], 4 epochs, 8 fixed validation samples, base_mdn hashes unchanged. Metrics come from the 8 fixed samples only and are not full validation results. No conclusion about the best K and no full training was started.

## Split continuity

Validation NLL change across each split: 1->2: -0.0089, 2->3: -0.0056. The review uses only a loose sanity guard (finite and |delta| <= 0.05, the smoke config max_split_nll_change). The strict continuity check on trained weights is in SPLIT_CHECK.json and the regression unit test; real-data deltas of a barely trained model are data-dependent.

## Resume checks (CPU, compared with uninterrupted `resume_ref`, tolerance 1e-6)

resume_cut: stop after phase 0, resume from phase_k01.pt. resume_e1: mid-phase resume from last.pt after epoch 1. resume_e2: resume from last.pt after epoch 2 (phase-end work not yet run). resume_p2: stop after final phase, resume from phase_k03.pt.

- `resume_cut`: verified (4 epochs, max |validation NLL diff| vs resume_ref = 0.0)
- `resume_e1`: verified (4 epochs, max |validation NLL diff| vs resume_ref = 0.0)
- `resume_e2`: verified (4 epochs, max |validation NLL diff| vs resume_ref = 0.0)
- `resume_p2`: verified (4 epochs, max |validation NLL diff| vs resume_ref = 0.0)
