# Evaluation Report - Ad Auction Lab

> Reproduce everything here with `make data validation evaluation`; `make report` prints every
> table from the raw JSON in `benchmarks/results/`. Each section names its command.

## Setup

- **Hardware:** Apple M1, 8 cores, 8 GB, macOS, shared with other work during the runs (the load
  is recorded in every results file under `machine`). Python 3.12, numpy 2, Rust core built with
  `--release`.
- **Data** (details in [DATA_CARD.md](DATA_CARD.md)):
  - Criteo Display Advertising Challenge, every 4th row in time order: 11,460,155 rows, base rate
    25.6%; random 80/10/10 splits (9,168,124 / 1,146,015 / 1,146,016 for seed 0).
  - Open Bandit Dataset, full size: 1,374,327 random and 12,357,200 BTS rounds in the all
    campaign (men and women are subsets).
  - Criteo Attribution Modeling for Bidding: 16,468,027 won impressions over 31 days with the
    price paid.
- **Reference libraries:** scikit-learn (SGD logistic regression), PyTorch (a factorization
  machine with sparse embeddings), obp 0.5.7 (off-policy estimators). Published Criteo numbers are
  quoted from BARS (Zhu et al. 2021) and were not rerun.
- **Seeds:** CTR final runs use 3 random splits (seeds 0 to 2, which also seed the FM's
  initialization); OPE intervals use 200 Poisson bootstrap replicates; the OPE learning curve 30
  random subsamples per size; pacing 5 market seeds.

## Metrics

- **Log loss** and **ROC AUC** on held-out rows. **Normalized entropy** (He et al. 2014) is log
  loss divided by the entropy of the training click rate, so 1.0 is the constant predictor.
- **Calibration** is mean prediction over mean label; **ECE** is the mean absolute gap between
  prediction and label over 20 equal-count bins.
- **DeLong** test for the AUC difference of two models on the same rows and a paired test on
  per-row log loss differences.
- **OPE relative error** is |estimate - truth| / truth, with the truth the click rate the target
  policy achieved in its own log (Saito et al. 2021); interval coverage is whether the 95%
  bootstrap interval contains the truth.
- **Auctions:** share of (bidder, auction) pairs where some bid factor in
  {0, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 1.05, 1.1, 1.25, 1.5, 2, 4} raises utility over truthful
  bidding; revenue and welfare per auction.
- **Shading:** expected surplus on the test days as a share of the surplus of the bid that is
  optimal for the test days themselves.
- **Pacing:** value won relative to no pacing, budget utilization, the time the budget runs out
  (as a fraction of the horizon), and value relative to the fractional knapsack optimum in
  hindsight.

## CTR models (`benchmarks/ctr_criteo.py`)

**Tuning** (`sweep`, `sweep-fm`) on the validation split: FTRL over alpha {0.02, 0.05, 0.1, 0.2} x
L1 {0, 1, 4}, 2 passes; FM first over k {8, 16} x AdaGrad rate {0.02, 0.05, 0.1}, 3 passes, then
over k {8, 16, 32} x rate {0.005, 0.01, 0.02} x factor L2 {1e-5, 1e-4}, 2 passes. Chosen: FTRL alpha 0.05, L1 1, 2 passes (within 0.0002 logloss of
L1 0 with half the weights); FM k 32, rate 0.01, 1 pass (every FM overfits on its second pass,
for example 0.4528 to 0.4630 validation logloss at rate 0.02).

**Final** (`final`), test split, mean ± std over 3 splits:

| Model | Logloss | AUC | NE | Calibration | ECE |
|---|---:|---:|---:|---:|---:|
| FTRL | 0.4586 ± 0.0003 | 0.7911 ± 0.0001 | 0.8059 ± 0.0005 | 1.004 ± 0.002 | 0.0028 ± 0.0003 |
| FTRL + Platt | 0.4586 ± 0.0003 | 0.7911 ± 0.0001 | 0.8059 ± 0.0005 | 1.001 ± 0.000 | 0.0028 ± 0.0004 |
| FTRL + isotonic | 0.4586 ± 0.0002 | 0.7911 ± 0.0002 | 0.8059 ± 0.0005 | 1.001 ± 0.000 | 0.0011 ± 0.0000 |
| FM | 0.4518 ± 0.0002 | 0.7991 ± 0.0001 | 0.7939 ± 0.0004 | 0.998 ± 0.002 | 0.0062 ± 0.0002 |
| FM + Platt | 0.4518 ± 0.0002 | 0.7991 ± 0.0001 | 0.7939 ± 0.0004 | 1.000 ± 0.000 | 0.0061 ± 0.0005 |
| FM + isotonic | 0.4516 ± 0.0002 | 0.7991 ± 0.0001 | 0.7935 ± 0.0004 | 1.001 ± 0.000 | 0.0012 ± 0.0002 |

FM minus FTRL on the seed 0 split: AUC +0.00826 (DeLong 95% CI [+0.00797, +0.00854]), log loss
-0.00704 (paired CI [-0.00728, -0.00680]); both p < 1e-12. The FM's reliability curve
over-predicts slightly in the lowest bins (0.023 predicted against 0.018 observed) and
under-predicts in the 0.25 to 0.45 range by about 0.01; isotonic calibration fixes both.

**References** (`reference`): REFERENCES_TEXT

**Published baselines** (quoted from the BARS Criteo_x4_001 leaderboard, not rerun): LR 0.4568 /
0.7934, FM 0.4431 / 0.8086, FFM 0.4407 / 0.8113, DeepFM 0.4376 / 0.8143 (logloss / AUC), trained on
36.7M rows with a count-filtered vocabulary and embedding size 16. This repo's FTRL is 0.0018
logloss behind the BARS LR on a quarter of the data, its FM 0.0087 behind the BARS FM.

**Ablations** (`ablate`), seed 0 test split, chosen passes:

- *Hash size:* from 2^16 to 2^22 buckets FTRL improves 0.46441 to 0.45869 logloss and the FM
  0.45521 to 0.45165; 2^24 adds 0.00014 for FTRL. The FM at 2^24 was not run (4 GB of state).
- *Feature groups:* the 13 integer fields alone give AUC 0.715 (FTRL) and 0.720 (FM); the 26
  categorical fields alone 0.762 and 0.770; together 0.791 and 0.799.
- *Training size:* 100k, 300k, 1M, 3M and 9.17M rows give FM AUC 0.7535, 0.7664, 0.7797, 0.7907,
  0.7992. FTRL beats the FM at 100k rows on both metrics (AUC 0.7551 against 0.7535) and at 300k
  on logloss; from 1M rows on the FM is ahead.
- *FM rank:* k = 1, 2, 4, 8, 16, 32 give logloss 0.46003, 0.45925, 0.45735, 0.45572, 0.45315,
  0.45165; k = 1 is already better than FTRL (0.45869), and doubling k still helps at 32.
- *L1:* 0, 1, 4, 16 give 3,722,562, 1,775,280, 214,897 and 48,888 nonzero weights at logloss
  0.45855, 0.45869, 0.45933, 0.46114.

**Speed** (`speed`): SPEED_TEXT

## Off-policy evaluation (`benchmarks/ope_obd.py`)

**Design.** For each campaign, the BTS policy is evaluated from the random log and the random
policy from the BTS log. The BTS target is represented by its marginal (item, position)
distribution over the week; the random target is uniform. Reward models: the (item, position)
click rate smoothed toward the position rate with prior 10, and on the random logs (which carry
user features) 2-fold cross-fitted hashed FTRL and FM models with user, item, affinity and
user-by-item features. Switch-DR switches to the model above the 99th percentile of the weights.

**Main results** (`main`): the README table gives the relative error of every estimator; the
95% interval covered the truth for:

| Setting | IPS | SNIPS | DR | SNDR | Switch-DR |
|---|---:|---:|---:|---:|---:|
| all: BTS from random | yes | yes | yes | yes | yes |
| all: random from BTS | yes | no | yes | yes | yes |
| men: BTS from random | no | no | no | no | no |
| men: random from BTS | yes | no | yes | yes | no |
| women: BTS from random | yes | yes | yes | yes | yes |
| women: random from BTS | no | no | no | no | yes |

DM intervals are narrow (the model is fixed and only the averaging is resampled) and never
cover; the bootstrap does not include reward-model uncertainty. The men BTS setting is missed by
every estimator by about 10%, including IPS which uses no model: the context-free marginal
representation of a policy that changed during the week is the likely cause, and it is listed as
a threat below.

**Reward models.** Contextual models did not help: on the all campaign DM error was 0.094 (FTRL)
and 0.064 (FM) against 0.006 for the item-position model, and DR changed by less than 0.004. A
smoothing prior above 100 hurts DM (0.049 at 1000 on all, BTS target) while DR stays at 0.006,
the expected robustness of DR to a biased model.

**Learning curve** (`curve`, 30 subsamples per size, all campaign): for BTS from random every
estimator's mean relative error falls from 0.18 to 0.20 at 10k rounds to 0.047 to 0.048 at 300k.
For random from BTS, at 10k rounds DR is off by 0.53, SNIPS 0.43, IPS 0.35, DM 0.20 and Switch-DR
0.19; at 1M rounds DM and Switch-DR are at 0.029 and the others at 0.057 to 0.084.

**Agreement with obp** (`obp`): on 200k-round samples of all six settings, every estimator matches
obp 0.5.7 within 6.1e-18 absolute. Point estimates are 56x to 97x faster than obp's; bootstrap
intervals (100 replicates) take 0.5x to 1.3x obp's time. The validation test suite repeats the
comparison on synthetic logs with context-dependent policies in CI.

## Auctions, shading and pacing (`benchmarks/markets.py`)

**Incentives** (`incentives`, 20,000 auctions per configuration, 3, 5 and 10 bidders, 1, 3 and 5
slots, squashing exponent 1 and 0.5, with and without the empirical reserve): VCG never offered a
profitable deviation; GSP did in 0% (one slot, where it is second price) to 21.6% of
bidder-auctions, with total gains of 2.4% to 5.1% of welfare; first price in 9.6% to 59.2%,
with gains of 61% to 69% of welfare. Reserves cut these shares to at most 0.6% (GSP) and 7.2% (first price)
because fewer bidders remain eligible. GSP at the lowest envy-free equilibrium matches VCG's
allocation and revenue (relative difference 4.3e-16 on 4,000 auctions); VCG's closed form matches
brute force with Clarke payments (4.7e-13 on 1,800 auctions).

**Reserves** (`reserves`): revenue with the empirical reserve exceeds the log-normal Myerson
reserve at every market size (for example 2.471 ± 0.013 against 2.417 ± 0.010 with 2 bidders) and
raises revenue over no reserve by 2.2x with 2 bidders and 7% with 10, keeping 70% and 91% of
welfare.

**Shading** (`shading`): the empirical landscape fit on days 1 to 21 keeps 99.8% to 100% of the
oracle surplus on days 22 to 31, the log-normal landscape 97.9% to 99.6%, and truthful bidding 0%.

**Pacing** (`pacing`, 1M auctions, 5 seeds): dual descent ends within 0.3% of the horizon with
the budget spent and wins 5.2x to 5.7x the value of no pacing at a 10% budget; PID with gains
kp 2, ki 0.05 runs out at 94% of the horizon and wins 3.2x to 3.5x. Against the hindsight optimum
for one advertiser dual descent reaches 0.996 to 0.999, PID 0.32 to 0.98.

**Speed** (`speed`): the Rust simulator is 58x to 162x faster than the Python reference, with
identical results, and simulates 10M auctions with 10 advertisers in 0.95 s.

## Threats to validity

- **Shared machine.** Other jobs ran during the benchmarks (load average up to 108 during the
  final CTR runs), so wall-clock numbers vary between runs (FM training 37k to 290k rows/s).
  Speed comparisons run both implementations back to back in one process; quality metrics do not
  depend on load.
- **Subsampled Criteo.** A quarter of the rows, so comparisons with published numbers mix data
  size with model; the training-size ablation shows the FM still improving at 9.17M rows.
- **Tuning.** Hyperparameters were chosen on validation for this repo's models; the scikit-learn
  and PyTorch references got their own small grids on the same validation split, but less tuning
  effort than the main models.
- **OPE ground truth.** On-policy truth has its own sampling error (reported as truth ± 1.96 SE
  in `make report`, up to ±4.1% relative for the random policy on men), and the BTS target is
  represented without context.
- **Simulated values.** Auction values and pacing values are derived from prices and CTRs, not
  observed advertiser values; incentives are checked on a bid grid, not proven.
- **Censored prices.** The Criteo Attribution prices are for impressions Criteo won, so the
  landscapes describe that bidder's market, not the whole exchange.

## Reproducibility

`make data` downloads and prepares the data (2.1 GB); `make validation` installs the references;
`make evaluation` runs every command above in order; raw results land in `benchmarks/results/`
with the machine, library versions, date and peak memory of each run; `make report` prints the
tables. The tests (`make coverage`, `make parity`, `make validation`) run on small fixtures in CI.
