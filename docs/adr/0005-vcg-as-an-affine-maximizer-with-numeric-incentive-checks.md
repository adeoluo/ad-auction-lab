# ADR-0005: Closed-form position auctions with numeric incentive checks

- **Status:** Accepted
- **Date:** 2026-11-05

## Context

The spec asks for GSP, VCG and first-price auctions with reserve prices and quality scores, plus
incentive checks. Ads systems rank by `q^lambda * b` (with a squashing exponent `lambda`) and add
a per-click reserve, and the textbook VCG formula for position auctions assumes neither.

## Decision

* Treat VCG with quality weights and reserves as an affine maximizer: maximize
  `sum_j alpha_j q_i^lambda (b_i - r)` and charge `r` plus the weighted externality, which in a
  separable position auction has the closed form `E_j = sum_{k > j} (alpha_{k-1} - alpha_k) s'_k`.
  Affine maximizers are dominant-strategy truthful for every `lambda` and `r`.
* Vectorize every mechanism over a batch of `T` auctions (`(T, n)` arrays).
* Verify incentives numerically: `deviation_gain` lets one bidder scale its bid by 13 factors from
  0 to 4 while the others bid truthfully and reports the best utility gain. The checks run on
  values and quality scores drawn from real Criteo data (`benchmarks/markets.py incentives`).
* Check VCG against exhaustive assignment with Clarke pivot payments (`vcg_bruteforce`) and GSP
  against the lowest envy-free equilibrium of Edelman, Ostrovsky and Schwarz (2007).

## Alternatives considered

- **Brute-force VCG everywhere.** Exact but factorial in the number of slots; kept as the test
  oracle only.
- **Analytic incentive proofs only.** The numeric check also catches implementation errors in
  tie-breaking and reserves, which proofs do not.

## Consequences

- The deviation grid can only show that a mechanism is not truthful (a profitable deviation
  exists); a zero result means no profitable deviation on the grid, not a proof.
