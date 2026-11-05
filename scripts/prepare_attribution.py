"""Stream the Criteo Attribution Modeling for Bidding dataset into compact arrays.

The gzip file (653 MB, 16.5 million impressions from 30 days of Criteo live traffic) is piped in
on stdin and never stored. Each impression is a display Criteo won in a real-time auction; the
``cost`` column is the (transformed) price Criteo paid for it, so it is a sample of the market
price an advertiser has to beat. Written to ``--out`` as one ``.npy`` per column:

* ``timestamp`` (uint32, seconds from the start), ``campaign`` (uint32), ``click`` (uint8),
  ``conversion`` (uint8), ``cost`` (float32), ``cpo`` (float32, cost per order)

plus ``meta.json``. Usage::

    curl -sL "$URL" | python scripts/prepare_attribution.py \
        --out ../.bench-data/ad-auction-lab/attribution
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

URL = (
    "https://huggingface.co/datasets/criteo/criteo-attribution-dataset/resolve/main/"
    "criteo_attribution_dataset.tsv.gz"
)
COLUMNS = {
    "timestamp": np.uint32,
    "campaign": np.uint32,
    "click": np.uint8,
    "conversion": np.uint8,
    "cost": np.float32,
    "cpo": np.float32,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    parts: dict[str, list[np.ndarray]] = {k: [] for k in COLUMNS}
    t0 = time.time()
    with gzip.open(sys.stdin.buffer, "rt") as fh:
        for chunk in pd.read_csv(fh, sep="\t", usecols=list(COLUMNS), chunksize=1_000_000):
            for k, dt in COLUMNS.items():
                parts[k].append(chunk[k].to_numpy().astype(dt))
            rows = sum(len(x) for x in parts["click"])
            print(f"{rows:,} rows ({time.time() - t0:.0f} s)", file=sys.stderr, flush=True)
    arrays = {k: np.concatenate(v) for k, v in parts.items()}
    order = np.argsort(arrays["timestamp"], kind="stable")
    for k, v in arrays.items():
        np.save(args.out / f"{k}.npy", v[order])
    meta = {
        "source": "Criteo Attribution Modeling for Bidding dataset (Diemert et al., 2017)",
        "url": URL,
        "rows": len(order),
        "campaigns": len(np.unique(arrays["campaign"])),
        "clicks": int(arrays["click"].sum()),
        "conversions": int(arrays["conversion"].sum()),
        "days": float(arrays["timestamp"].max() / 86400),
        "seconds": round(time.time() - t0, 1),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
