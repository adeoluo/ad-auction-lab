# ADR-0002: Hashed CTR models in a Rust core with a Python reference

- **Status:** Accepted
- **Date:** 2026-11-05

## Context

The CTR models train on 9.17 million Criteo rows with 39 categorical fields each. Criteo's
categorical columns have tens of millions of distinct values, so a vocabulary would need a
counting pass and a large dictionary, and every new value at serving time would need a rule.
The learners themselves (FTRL-Proximal and an AdaGrad factorization machine) are per-row
sequential updates over a few dozen active coordinates, which is the worst case for numpy: the
work per row is tiny and the loop is in Python. A per-row Python loop runs at about 4,000 rows
per second, so one pass over the training split would take about 40 minutes.

## Decision

* Hash every `(field, value)` token to one of `2^bits` buckets with the SplitMix64 finalizer of
  `((field + 1) << 32) | value` (`hashing.py`, `rust/src/hashing.rs`, identical bit for bit).
  Integer features are first bucketed with the log-squared rule of the Kaggle winners into a
  `uint8` block; categorical features stay as their 32-bit ids in a `uint32` block. No vocabulary,
  no rare-value filtering: collisions are the price, measured by the `bits` ablation.
* Implement both learners and prediction in Rust (`rust/src/ftrl.rs`, `rust/src/fm.rs`), compiled
  into the package with maturin as `ad_auction_lab.ad_auction_lab_rs` and used by default.
  Training is sequential in the caller's row order (the update order defines the model);
  prediction is parallel over 4,096-row chunks with rayon.
* Keep a readable per-row Python implementation (`ctr_reference.py`) that performs the same
  operations in the same order, and require parity in `tests/test_rust_parity.py` and
  `tests/test_ctr.py`. `backend="python"` selects it explicitly.

## Alternatives considered

- **scikit-learn `SGDClassifier` or PyTorch for the models.** Both are run as references in
  `benchmarks/ctr_criteo.py reference`, but neither is FTRL-Proximal with per-coordinate L1, and
  the point is to own the learner.
- **Numba.** Simpler build, but the portfolio already ships Rust cores with PyO3, and rayon makes
  the parallel prediction path trivial.
- **A vocabulary with a minimum count (as in the BARS Criteo protocols).** Fewer collisions, but a
  second pass over the data and state that has to be shipped with the model.

## Consequences

- One pass over 9.17 million rows takes about 9 seconds for FTRL and about 30 for the FM on the
  M1 (`benchmarks/results/ctr_final.json`), against minutes per million rows in Python
  (`ctr_speed.json`).
- Hash collisions cost accuracy at small `bits`; the ablation quantifies it.
- Two implementations must be kept in step; the parity tests fail if they drift.
