# ADR-0003: FTRL-Proximal and an AdaGrad factorization machine, trained in log order

- **Status:** Accepted
- **Date:** 2026-11-05

## Context

The spec asks for logistic regression with hashing and FTRL-Proximal plus a factorization
machine. Production CTR systems train online on a time-ordered stream, evaluate with log loss,
AUC and normalized entropy (He et al., 2014), and care about calibration because the predicted
CTR multiplies the bid in the auction.

## Decision

* Logistic regression is trained with FTRL-Proximal (McMahan et al., 2013): per-coordinate
  learning rates `alpha / (beta + sqrt(n_i))` and an L1 term that keeps most weights exactly zero.
* The factorization machine (Rendle, 2010) uses the `O(k * fields)` form of the pairwise term and
  AdaGrad per parameter, with L2 on the factors.
* Benchmarks train in file order. Criteo's `train.txt` is time ordered, so this is the order an
  online learner would see; it also turns memory-map reads into sequential reads. A shuffled pass
  was measured at about 100,000 rows per second against about 1,000,000 in order on this machine
  (page faults on the 1.35 GB map dominate), so shuffling is left as an option of `fit` and not
  used for the headline runs.
* Normalized entropy uses the training click rate as the background rate. Calibration is reported
  raw and after Platt scaling or isotonic regression fitted on the validation split
  (`calibration.py`, checked against scikit-learn).
* Hyperparameters are selected on the validation split only (`ctr_criteo.py sweep`).

## Alternatives considered

- **Field-aware FM or deep models (DeepFM, DCN).** Stronger on Criteo, but outside the spec; their
  published numbers are quoted in the README for context and marked as not rerun.
- **Plain SGD logistic regression.** Run as the scikit-learn reference instead.

## Consequences

- Results depend on row order; the variance across random 80/10/10 splits is reported instead of
  shuffling seeds.
- The models are small enough (2^22 buckets) to serve from the HTTP service.
