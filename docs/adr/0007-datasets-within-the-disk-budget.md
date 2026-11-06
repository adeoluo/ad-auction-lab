# ADR-0007: Datasets within the disk budget

- **Status:** Accepted
- **Date:** 2026-11-05

## Context

The machine (Apple M1, 8 GB of memory) has a few GB of disk budget for this project and no GPU. The spec asks
for the largest feasible Criteo split (at least several million rows), the full Open Bandit
Dataset if feasible, and market prices for bidding.

## Decision

* **Criteo Display Advertising Challenge** (Kaggle 2014, 45,840,617 labelled rows, 4.58 GB
  archive): stream the archive without storing it and keep every 4th row as compact arrays
  (`uint8` integer bins, `uint32` categorical ids, `uint8` labels): 11,460,155 rows in 1.35 GB,
  memory-mapped for training. Split at random 80/10/10.
* **Open Bandit Dataset** (Saito et al., 2021), full size: random and Bernoulli TS logs for the
  all, men and women campaigns (26.9 million rounds over the six logs; the all campaign contains the other two), converted to `.npz` (481 MB); user
  context is kept for the random logs.
* **Criteo Attribution Modeling for Bidding** (Diemert et al., 2017): 16,468,027 impressions with
  the price paid, time ordered, six columns (283 MB). Used as the market price distribution for
  bid shading and pacing.

## Alternatives considered

- **Criteo 1TB click logs.** Far beyond the disk and time budget.
- **All 45.8 million Kaggle rows.** 5.4 GB in the same format; over the budget, and the 1-in-4
  sample still gives 9.17 million training rows.

## Consequences

- Absolute Criteo numbers are not directly comparable with papers that train on all 45.8 million
  rows and filter rare values; the README says so next to every published number.
- The attribution prices are prices of impressions Criteo won, a censored sample of the market.
