# ADR-0004: Off-policy estimators as per-round terms that match obp exactly

- **Status:** Accepted
- **Date:** 2026-11-05

## Context

The Open Bandit Dataset logs have up to 12.4 million rounds (BTS, all items). The Open Bandit
Pipeline (obp) is the reference implementation, but its estimators take `(n, n_actions,
n_positions)` arrays for the evaluation policy and the reward model: for the BTS log that is
3 billion float64 values per array (about 24 GB), which does not fit on this machine. Confidence
intervals by resampling and re-running an estimator multiply the cost by the number of replicates.

## Decision

* Implement IPS, SNIPS, DM, DR, SNDR and Switch-DR with exactly the definitions of obp 0.5.7,
  including its normalization of the evaluation policy over actions at each position in the DM
  term, and test agreement to 1e-12 (`tests/test_validation.py`).
* Accept context-free policies and reward models as `(n_actions, n_positions)` arrays and
  compute `q(x, pi, k)` once per position instead of materializing the 3-D arrays.
* Write every estimator as `mean(a) + sum(b) / sum(c)` of per-round terms. A Poisson bootstrap
  replicate is then a weighted sum of four columns (`bootstrap`), so 200 replicates over 12.4
  million rounds need no estimator re-runs.
* Ground truth for evaluation is the on-policy click rate of the target policy in its own log,
  as in Saito et al. (2021). The BTS policy is summarized as its mean logged propensity per
  (item, position), which is exact for a context-free policy that is fixed in time and an
  approximation for BTS, whose propensities drift as its posterior updates.

## Alternatives considered

- **Call obp directly.** Used for validation on 200,000-round prefixes, where its arrays fit.
- **Multinomial bootstrap.** Exact resampling, but needs index arrays per replicate; Poisson(1)
  weights are the standard large-sample approximation and stream in chunks.

## Consequences

- Full-size estimates with confidence intervals run in seconds and in a bounded amount of memory.
- Any change to an estimator must keep the obp parity tests green.
