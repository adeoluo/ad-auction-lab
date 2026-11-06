# Data Card - Ad Auction Lab

Nothing below is committed to the repository; `make data` (`scripts/download_data.sh`) downloads
and converts every dataset into `../.bench-data/ad-auction-lab/` (2.1 GB in total). The tests use
small synthetic fixtures (`tests/fixtures.py`) in the same on-disk formats.

## Criteo Display Advertising Challenge (Kaggle, 2014)

- **Source:** `train.txt` from the Criteo Labs release (figshare mirror, 4.58 GB archive, 45,840,617
  labelled rows over 7 days, 13 integer and 26 hashed categorical features, CTR about 25.6%).
- **Kept:** every 4th row in file order (stride 4), 11,460,155 rows, streamed without storing the
  archive (`scripts/prepare_criteo.py`). Integer features become log-squared bins in `uint8`
  (0 = missing), categorical features stay as 32-bit ids in `uint32`, labels in `uint8`; 1.35 GB,
  memory-mapped.
- **Splits:** random 80/10/10 partitions from `split_rows(seed)`: 9,168,124 train, 1,146,015
  validation and 1,146,016 test rows for seed 0. Seeds 1 and 2 give the other partitions used for
  variance.
- **Licence and use:** released by Criteo for research; the features are anonymized and contain no
  directly identifying information.
- **Caveats:** the 1-in-4 sample keeps the time order and the class balance but has a quarter of
  the rows of the full set, so rare categorical values are rarer still.

## Open Bandit Dataset (Saito et al., 2021)

- **Source:** ZOZOTOWN fashion e-commerce recommendation logs, full size (413 MB zip), released
  by ZOZO Research for research use.
- **Kept:** all six logs (`scripts/prepare_obd.py`), 481 MB as `.npz`:

| Log | Rounds | Clicks | CTR | Items |
|---|---:|---:|---:|---:|
| random, all | 1,374,327 | 4,768 | 0.347% | 80 |
| BTS, all | 12,357,200 | 61,208 | 0.495% | 80 |
| random, men | 452,949 | 2,321 | 0.512% | 34 |
| BTS, men | 4,077,727 | 27,495 | 0.674% | 34 |
| random, women | 864,585 | 4,163 | 0.482% | 46 |
| BTS, women | 7,765,497 | 49,539 | 0.638% | 46 |

- Each round: item, position (3 slots), click, propensity. The random logs also keep the four
  hashed user features and the user-item affinity columns used by the contextual reward models.
- **Caveats:** the all campaign contains the men and women rounds; BTS propensities were
  computed by the platform with Monte Carlo sampling and drift over time.

## Criteo Attribution Modeling for Bidding (Diemert et al., 2017)

- **Source:** Hugging Face mirror of the Criteo dataset (653 MB gzip), streamed.
- **Kept:** 16,468,027 impressions over 30.9 days from 675 campaigns, time ordered, with
  timestamp, campaign, click (5,947,563), conversion (806,196), cost and cost per order; 283 MB.
- **Use:** `cost` is the (transformed) price Criteo paid for each impression it won, used as the
  market price for bid shading and as the external market in the pacing simulator.
- **Caveats:** prices of won impressions only (censored), scaled by Criteo, so absolute values are
  not currency.
