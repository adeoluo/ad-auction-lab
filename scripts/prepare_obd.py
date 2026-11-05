"""Convert the full Open Bandit Dataset zip into compact ``.npz`` files.

For each behavior policy (``random``, ``bts``) and campaign (``all``, ``men``, ``women``) the CSV
inside the zip is read in chunks, without extracting it, and written to
``<out>/<policy>_<campaign>.npz`` with:

* ``item`` (uint8), ``position`` (uint8, 0-based), ``click`` (uint8), ``pscore`` (float64)
* for the random logs only (the contexts of the larger BTS logs are not needed here):
  ``user`` (uint32, 4 columns), the first 32 bits of each hashed user feature, and
  ``affinity`` (uint8, one column per item), user-item affinity counts capped at 255

plus ``item_<campaign>.npz`` with the item features and ``meta.json`` with row counts and click
rates. Usage::

    python scripts/prepare_obd.py --zip obd.zip --out ../.bench-data/ad-auction-lab/obd
"""

from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = "open_bandit_dataset"


def hex32(col: pd.Series) -> np.ndarray:
    return np.array([int(s[:8], 16) for s in col.astype(str)], dtype=np.uint32)


def convert(zf: zipfile.ZipFile, policy: str, campaign: str, out: Path) -> dict[str, float]:
    name = f"{ROOT}/{policy}/{campaign}/{campaign}.csv"
    contexts = policy == "random"
    keys = ["item", "position", "click", "pscore"] + (["user", "affinity"] if contexts else [])
    parts: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    usecols = None if contexts else ["item_id", "position", "click", "propensity_score"]
    t0 = time.time()
    with zf.open(name) as fh:
        for chunk in pd.read_csv(fh, chunksize=500_000, usecols=usecols):
            parts["item"].append(chunk["item_id"].to_numpy(np.uint8))
            parts["position"].append((chunk["position"].to_numpy() - 1).astype(np.uint8))
            parts["click"].append(chunk["click"].to_numpy(np.uint8))
            parts["pscore"].append(chunk["propensity_score"].to_numpy(np.float64))
            if not contexts:
                continue
            parts["user"].append(
                np.stack([hex32(chunk[f"user_feature_{j}"]) for j in range(4)], axis=1)
            )
            aff = chunk[
                [c for c in chunk.columns if c.startswith("user-item_affinity_")]
            ].to_numpy()
            parts["affinity"].append(np.clip(aff, 0, 255).astype(np.uint8))
        rows = sum(len(x) for x in parts["item"])
        print(f"  {policy}/{campaign}: {rows:,} rows ({time.time() - t0:.0f} s)", flush=True)
    arrays = {k: np.concatenate(v) for k, v in parts.items()}
    np.savez(out / f"{policy}_{campaign}.npz", **arrays)
    n = len(arrays["item"])
    return {
        "rows": n,
        "clicks": int(arrays["click"].sum()),
        "ctr": float(arrays["click"].mean()),
        "n_items": int(arrays["item"].max()) + 1,
    }


def items(zf: zipfile.ZipFile, campaign: str, out: Path) -> None:
    with zf.open(f"{ROOT}/random/{campaign}/item_context.csv") as fh:
        df = pd.read_csv(fh)
    np.savez(
        out / f"item_{campaign}.npz",
        item_id=df["item_id"].to_numpy(np.int64),
        feature_0=df["item_feature_0"].to_numpy(np.float64),
        features=np.stack([hex32(df[f"item_feature_{j}"]) for j in (1, 2, 3)], axis=1),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--policies", default="random,bts")
    ap.add_argument("--campaigns", default="all,men,women")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    meta: dict[str, object] = {
        "source": "Open Bandit Dataset (Saito et al., 2021), full size",
        "logs": {},
    }
    with zipfile.ZipFile(args.zip) as zf:
        for campaign in args.campaigns.split(","):
            items(zf, campaign, args.out)
            for policy in args.policies.split(","):
                meta["logs"][f"{policy}_{campaign}"] = convert(zf, policy, campaign, args.out)  # type: ignore[index]
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
