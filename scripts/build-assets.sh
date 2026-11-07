#!/usr/bin/env bash
# Build the release assets into dist/release: a wheel with the compiled Rust core for this
# platform (abi3, CPython 3.11+) and the source distribution. Called by scripts/release.sh.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
rm -rf dist/release
uv build --wheel --sdist --out-dir dist/release
ls -l dist/release
