#!/usr/bin/env bash
# Download and prepare the benchmark data under $1 (default ../.bench-data/ad-auction-lab).
#
# * Criteo Display Advertising Challenge: the 4.58 GB archive is streamed (never stored) and every
#   4th row of train.txt is kept as compact arrays (11,460,155 rows, 1.35 GB).
# * Open Bandit Dataset (full size, 413 MB zip): the random and Bernoulli TS logs of the three
#   campaigns are converted to compact .npz files; the zip is deleted afterwards (KEEP_ZIP=1 keeps it).
set -euo pipefail
OUT="${1:-../.bench-data/ad-auction-lab}"
mkdir -p "$OUT"
if [[ ! -f "$OUT/criteo/meta.json" ]]; then
  curl -sL https://ndownloader.figshare.com/files/10082655 \
    | uv run --no-sync python scripts/prepare_criteo.py --out "$OUT/criteo" --stride "${CRITEO_STRIDE:-4}"
fi
if [[ ! -f "$OUT/obd/meta.json" ]]; then
  [[ -f "$OUT/obd.zip" ]] || curl -sL -o "$OUT/obd.zip" https://research.zozo.com/data_release/open_bandit_dataset.zip
  uv run --no-sync python scripts/prepare_obd.py --zip "$OUT/obd.zip" --out "$OUT/obd"
  [[ "${KEEP_ZIP:-0}" == "1" ]] || rm -f "$OUT/obd.zip"
fi
du -sh "$OUT"/criteo "$OUT"/obd
