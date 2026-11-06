# ADR-0006: Budget pacing in a Rust simulator on real prices and CTRs

- **Status:** Accepted
- **Date:** 2026-11-05

## Context

Budget pacing has to be evaluated over long horizons (a million auctions or more) with several
advertisers competing, and the result depends on how prices move over the day. A synthetic i.i.d.
market hides exactly the non-stationarity that makes PID controllers and dual descent behave
differently.

## Decision

* Simulate repeated single-slot second-price auctions on eCPM in Rust (`rust/src/pacing.rs`)
  with a line-for-line Python reference (`simulate_reference`) for parity tests.
* Build each market from real data (`build_market`): the external competing eCPM is a contiguous,
  time-ordered window of prices from the Criteo Attribution dataset, and each advertiser's CTR in
  each auction is drawn from the FM's predictions on Criteo test rows. Budgets are a fixed share of
  what each advertiser would spend unpaced, so every budget binds.
* Implement two pacers: a PID controller on the normalized spend error (multiplicative, updated
  every `interval` auctions) and dual descent on the budget constraint (Balseiro and Gur, 2019).
* For a single advertiser against the external market, report the competitive ratio against the
  hindsight LP optimum (fractional knapsack), which upper-bounds every online policy.

## Alternatives considered

- **i.i.d. synthetic prices.** Easier to analyse; used in unit tests only.
- **Replaying a single real campaign.** The attribution data has no counterfactual bids, so a
  simulator is needed either way.

## Consequences

- The simulator's market is exogenous apart from the simulated advertisers; feedback from other
  bidders adapting is not modelled.
- Ten million auctions with ten advertisers simulate in seconds, so all pacing results use full
  horizons and five seeds.
