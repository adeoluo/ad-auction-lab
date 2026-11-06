# Model Card - Ad Auction Lab CTR models

## Overview

- **What it is:** two click-through-rate models on hashed categorical features: logistic
  regression trained with FTRL-Proximal, and a second-order factorization machine trained with
  AdaGrad. Both are implemented in Rust (`rust/src/ftrl.rs`, `rust/src/fm.rs`) with Python
  references, trained from scratch on Criteo.
- **Version:** 0.1.0.
- **Hyperparameters:** chosen on the Criteo validation split; see
  `benchmarks/results/ctr_sweep.json` and the README.

## Intended use

- Research and teaching: CTR estimation for auction simulations (quality scores and eCPM), reward
  models for off-policy evaluation, and a baseline to compare other CTR learners against.

## Out of scope

- Production ad serving without monitoring, bot filtering and recalibration on fresh data.
- Any decision about individual people; the features are anonymized ad and context signals.

## Training data

Criteo Display Advertising Challenge, every 4th row (11,460,155 rows), random 80/10/10 split; see
[DATA_CARD.md](DATA_CARD.md).

## Evaluation

Log loss, ROC AUC, normalized entropy (log loss over the entropy of the training click rate) and
calibration (mean prediction over mean label, expected calibration error over 20 equal-count
bins), on the held-out test split over three random splits; DeLong and paired log-loss tests
between models. Numbers are in the README and [EVALUATION_REPORT.md](EVALUATION_REPORT.md).

## Limitations

- Hash collisions at small table sizes; no feature crosses beyond the FM's pairwise term.
- Trained on a fixed 7-day window; CTR drifts, so calibration must be refreshed in deployment.
- Criteo's features are anonymized, so fairness across user groups cannot be assessed.
