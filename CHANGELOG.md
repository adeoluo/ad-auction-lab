# Changelog

All notable changes to this project are documented here. The file is generated
from Conventional Commits by [git-cliff](https://git-cliff.org) when a release is cut
with `make cut-release`.

## [0.1.0] 2026-11-07

### Features

- Hashed CTR models with a Rust core (#1)
- Auction mechanisms, bid shading and budget pacing (#2)
- Off-policy estimators validated against obp (#3)
- Command line interface and HTTP service (#4)
- Real data benchmark harness for CTR, OPE and markets (#5)
- Criteo CTR results, ablations and reference libraries (#6)
- Off-policy evaluation, market and reference library results (#9)

### Performance

- Poisson bootstrap in bounded memory column blocks (#7)
- Share bootstrap weights across estimators (#8)
- Build the bid shading envelope once per landscape (#10)

### Maintenance

- Scaffold project
- Release tooling, changelog and agent notes (#11)
