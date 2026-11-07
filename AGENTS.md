# AGENTS.md

Working notes for coding agents and contributors in **Ad Auction Lab**. The [README](README.md)
explains what the project does and why; this file explains how to change it safely.

## Project

Position auctions, bid shading, budget pacing, hashed CTR models and off-policy evaluation, measured on Criteo and the Open Bandit Dataset.

## Layout

- `benchmarks/`: benchmark scripts and recorded results
- `docker-compose.yml`: local container setup
- `docs/`: documentation, architecture diagram source and render, design records
- `rust/`: Rust core compiled into a Python extension with PyO3
- `scripts/`: maintenance scripts (release, assets, data)
- `src/`: library and service source
- `tests/`: test suite

Key modules:

- `rust/src/fm.rs`: Second-order factorization machine (Rendle, 2010) on hashed binary features, trained with AdaGrad on the log loss.
- `rust/src/ftrl.rs`: Logistic regression on hashed features trained with FTRL-Proximal (McMahan et al., 2013).
- `rust/src/hashing.rs`: Feature hashing shared by every model.
- `rust/src/lib.rs`: Rust core of ad-auction-lab.
- `rust/src/metrics.rs`: Small numeric helpers shared by the learners.
- `rust/src/pacing.rs`: Multi-advertiser budget pacing simulator: repeated single-slot second-price auctions on expected cost per impression (eCPM = bid per click x predicted CTR).
- `rust/src/python.rs`: PyO3 bindings, built by maturin into the `ad_auction_lab_rs` extension module.
- `src/ad_auction_lab/__init__.py`: Ad auctions, bidding, budget pacing, hashed CTR models and off-policy evaluation.
- `src/ad_auction_lab/auctions.py`: Position auctions for sponsored search and display slots: GSP, VCG and first price.
- `src/ad_auction_lab/backend.py`: The Rust backend, compiled into the package on install and used by default.
- `src/ad_auction_lab/bidding.py`: Bid shading for first-price auctions.
- `src/ad_auction_lab/calibration.py`: Post-hoc calibration of click probabilities: Platt scaling and isotonic regression.
- `src/ad_auction_lab/cli.py`: Command line interface: ``python -m ad_auction_lab <command>``.
- `src/ad_auction_lab/ctr.py`: Hashed click-through-rate models: FTRL-Proximal logistic regression and a factorization machine.
- `src/ad_auction_lab/ctr_reference.py`: Readable numpy reference implementations of the hashed CTR learners.
- `src/ad_auction_lab/data.py`: Tabular click data as two categorical blocks, and the Criteo loaders.
- `src/ad_auction_lab/hashing.py`: Feature hashing shared with the Rust core.
- `src/ad_auction_lab/metrics.py`: Click model metrics: log loss, ROC AUC, normalized entropy, calibration and significance tests.
- `src/ad_auction_lab/obd.py`: The Open Bandit Dataset (Saito et al., 2021) as bandit logs, and its contextual reward model.
- `src/ad_auction_lab/ope.py`: Off-policy estimators for (item, position) bandit logs: IPS, SNIPS, DM, DR, SNDR, Switch-DR.
- `src/ad_auction_lab/pacing.py`: Budget pacing in a repeated multi-advertiser second-price auction market.
- `src/ad_auction_lab/service.py`: A standard-library HTTP service for auctions, bid shading, CTR prediction and OPE.

## Commands

| Command | Purpose |
|---|---|
| `make help` | Show this help |
| `make setup` | Install dependencies (builds the Rust core) and pre-commit hooks |
| `make format` | Format automatically |
| `make lint` | Lint (no fixes) |
| `make type` | Type-check |
| `make test` | Run tests |
| `make coverage` | Tests with coverage gate |
| `make rust-check` | Rust core: fmt, clippy (with and without Python), unit tests |
| `make rust-develop` | Rebuild the Rust core into .venv after editing rust/ (uv sync also builds it) |
| `make parity` | Rust vs numpy reference parity tests |
| `make validation` | Tests against obp and scikit-learn (obp is installed without its pinned dependencies) |
| `make data` | Stream Criteo (1 row in 4), the Open Bandit Dataset and Criteo Attribution into $(DATA) |
| `make evaluation` | Every benchmark on the full data (about 2 hours on an M1; needs `make data validation`) |
| `make report` | Print the README tables from benchmarks/results/ |
| `make demos` | Record the demo GIFs (needs vhs and the prepared data) |
| `make diagram` | Render architecture diagram to SVG |
| `make docker-build` | Build the image |
| `make docker-up` | Run via docker compose |
| `make clean` | Remove caches and artifacts |
| `make all` | Full local gate (mirror of continuous integration) |
| `make changelog` | Regenerate CHANGELOG.md from Conventional Commits (git-cliff) |
| `make cut-release` | Cut a release locally: version bump, changelog, tag, GitHub release (RELEASE_VERSION=vX.Y.Z optional) |

Run the full local gate (`make all`) before committing; CI runs the same checks.

## Conventions

- Commits follow Conventional Commits (`feat:`, `fix:`, `perf:`, `refactor:`, `docs:`, `test:`, `ci:`, `chore:`).
  The changelog and the next version number are derived from them.
- Every number quoted in the README is produced by a command in this repository. When behavior
  changes, rerun that command and update the number in the same commit.
- When a component is added, removed or renamed, update `docs/architecture/architecture.mmd`
  and re-render `architecture.svg` so the diagram matches the code.
- README sections keep their fixed order. Prose avoids em and en dashes and promotional wording.
- Tests accompany behavior changes; a bug fix comes with a test that fails without it.
- Releases are cut with `make cut-release`.

## Python notes

- Python is managed with `uv`; run commands through `uv run` so the locked environment is used.
- Lint and format with `ruff`; type-check with `mypy`; tests use `pytest`.
- Public functions carry type hints and a docstring that states the contract, not the steps.

## Rust notes

- `cargo fmt --check` and `cargo clippy --all-targets -- -D warnings` must pass.
- No `unwrap()` in library code paths that can fail at runtime; return a typed error.
- Unsafe code is limited to the places that need a syscall and carries a `SAFETY:` comment.
