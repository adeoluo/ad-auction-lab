"""Stream the Criteo Display Advertising Challenge training set into compact arrays.

The archive (dac.tar.gz, 4.58 GB) is piped in on stdin and never stored: tar members are
decompressed on the fly, every ``--stride``-th labelled row of ``train.txt`` is kept, and rows
are written as three flat binary files in ``--out``:

* ``label.u8``  one byte per row (0 or 1)
* ``num.u8``    13 bytes per row, integer features mapped by ``ad_auction_lab.data.bin_integer``
* ``cat.u32``   26 little-endian uint32 per row, the original 32-bit hex categories (0 = missing)

plus ``meta.json`` with the row counts. Usage::

    curl -sL https://ndownloader.figshare.com/files/10082655 | \
        python scripts/prepare_criteo.py --out ../.bench-data/ad-auction-lab/criteo --stride 4
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

from ad_auction_lab.data import bin_integer

URL = "https://ndownloader.figshare.com/files/10082655"
N_INT = 13
N_CAT = 26
CHUNK = 200_000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--max-rows", type=int, default=0, help="stop after this many kept rows")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (
        open(args.out / "label.u8", "wb") as f_lab,
        open(args.out / "num.u8", "wb") as f_num,
        open(args.out / "cat.u32", "wb") as f_cat,
    ):
        lab = np.zeros(CHUNK, np.uint8)
        num = np.zeros((CHUNK, N_INT), np.uint8)
        cat = np.zeros((CHUNK, N_CAT), np.uint32)
        k = kept = seen = 0

        t0 = time.time()

        def flush() -> None:
            nonlocal k
            f_lab.write(lab[:k].tobytes())
            f_num.write(num[:k].tobytes())
            f_cat.write(cat[:k].astype("<u4").tobytes())
            k = 0

        with tarfile.open(fileobj=sys.stdin.buffer, mode="r|gz") as tar:
            for member in tar:
                if not member.name.endswith("train.txt"):
                    continue
                fh = tar.extractfile(member)
                assert fh is not None
                for raw in fh:
                    seen += 1
                    if (seen - 1) % args.stride:
                        continue
                    parts = raw.rstrip(b"\n").split(b"\t")
                    lab[k] = parts[0] == b"1"
                    for j in range(N_INT):
                        s = parts[1 + j]
                        num[k, j] = bin_integer(int(s)) if s else 0
                    for j in range(N_CAT):
                        s = parts[14 + j]
                        cat[k, j] = int(s, 16) if s else 0
                    k += 1
                    kept += 1
                    if k == CHUNK:
                        flush()
                        print(
                            f"{kept:,} rows kept of {seen:,} ({time.time() - t0:.0f} s)", flush=True
                        )
                    if args.max_rows and kept >= args.max_rows:
                        break
                break
        flush()
    meta = {
        "source": "Criteo Display Advertising Challenge (Kaggle 2014), train.txt",
        "url": URL,
        "rows_seen": seen,
        "rows_kept": kept,
        "stride": args.stride,
        "n_int": N_INT,
        "n_cat": N_CAT,
        "seconds": round(time.time() - t0, 1),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
