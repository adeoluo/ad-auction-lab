# Changelog

All notable changes to this project are documented here. The file is generated
from Conventional Commits by [git-cliff](https://git-cliff.org) when a release is cut
with `make cut-release`.

## Unreleased

### Features

- Hashed CTR models with a Rust core (#1)
- Auction mechanisms, bid shading and budget pacing (#2)
- Off-policy estimators validated against obp (#3)
- Command line interface and HTTP service (#4)
- Real data benchmark harness for CTR, OPE and markets (#5)
- Criteo CTR results, ablations and reference libraries (#6)
- Auction incentive, reserve price and OPE learning curve results
- OPE on the full Open Bandit Dataset, estimator ablations and pacing results
- Agreement with obp 0.5.7 on 200k-round samples of every setting
- Rust pacing simulator speed against the Python reference
- First-price bid shading on a time split of Criteo market prices
- Scikit-learn logistic regression reference on the same hashed features
- Rust CTR core speed against the Python reference on 200k rows

### Performance

- Poisson bootstrap in bounded memory column blocks (#7)
- Share bootstrap weights across estimators (#8)
- Build the bid shading envelope once per landscape (#10)

### Documentation

- README and evaluation report with the measured results

### Maintenance

- Scaffold project
