# Threat Model - Ad Auction Lab

This is an offline research library with a small HTTP service. It reads datasets and model files
chosen by the operator and JSON sent to the service; it calls no external services at run time.
Auctions, bidding and pacing run in simulation on public data; nothing here spends real money.

## Assets

- **Auction integrity:** who wins and what they pay. A wrong price or allocation transfers money
  between advertisers and the seller.
- **Evaluation integrity:** CTR metrics and off-policy estimates decide which model or policy
  would ship; a biased estimator or a leaked test split makes a bad policy look good.
- **Budgets:** a pacing bug can exhaust an advertiser's budget early or leave it unspent.
- **Service availability** of `/auction`, `/shade`, `/predict` and `/ope`.

## Trust boundaries

1. JSON arriving at the HTTP service is untrusted.
2. Model files loaded by `serve --model` are trusted to come from the operator but are parsed
   defensively.
3. Datasets come from the public Criteo, ZOZO and Hugging Face URLs in `scripts/download_data.sh`.

## Threats and mitigations

| Threat | Vector | Impact | Mitigation |
|---|---|---|---|
| Resource exhaustion | Large bodies, many bids or long market arrays | Memory or CPU exhaustion | Bodies over 2 MB get 413 before they are read; at most 1,000 bidders per auction, 10,000 rows per prediction or shading request and 200,000 market prices; threaded server |
| Malformed input | Strings, NaN, negative bids, wrong shapes | Crashes or wrong prices | Every route validates types, shapes and ranges and answers 400 or 422; covered by `tests/test_service.py` |
| Code execution through model files | Pickled objects in an `.npz` | Arbitrary code on load | Models are loaded with `np.load(allow_pickle=False)`; parameters are parsed from a literal, not evaluated |
| Bid shading as a strategic attack | A bidder learns the landscape and shades in a GSP or VCG auction | Lower seller revenue | Shading only pays in first-price auctions; the incentive benchmark measures profitable deviations per mechanism |
| Reserve price gaming | Bidders collude below the reserve | Lower revenue | Out of scope; reserves are evaluated against i.i.d. values only |
| Biased off-policy estimates | Wrong propensities, policies without support on logged actions | A worse policy appears better | Estimators match obp to 1e-12; benchmarks report effective sample size, weight maxima, bootstrap intervals and coverage of the on-policy truth |
| Test-set leakage in CTR tuning | Tuning on the test split | Optimistic metrics | Hyperparameters are chosen on the validation split; `split_rows` partitions are disjoint and seeded |
| Click spam or bots | Fake clicks in training logs | Inflated CTRs and bids | Public research data only; production logs need bot filtering before training |
| Supply chain | Malicious dependency | Code execution | Dependencies pinned in `uv.lock`; the Rust crate depends only on `pyo3` and `rayon`; CodeQL on main |

## Residual risk

The service has no authentication or rate limiting and is meant for local use or behind a
gateway. Simulated markets do not model adversarial bidders who adapt to the pacing or shading
rules of others.
