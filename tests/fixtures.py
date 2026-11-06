"""On-disk fixtures in the formats written by the preparation scripts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ad_auction_lab.data import synthetic_clicks

A, N = 6, 4000


def write_fixture(d: Path, seed: int = 0) -> None:
    """A small dataset in the format of ``scripts/prepare_obd.py``."""

    rng = np.random.default_rng(seed)
    user = rng.integers(0, 2**32, (N, 4), dtype=np.uint64).astype(np.uint32)
    user[:, 0] = rng.integers(0, 3, N)  # a feature with a real effect
    affinity = rng.integers(0, 5, (N, A)).astype(np.uint8)
    item = rng.integers(0, A, N).astype(np.uint8)
    logit = -2.0 + 0.6 * (user[:, 0] == item % 3) + 0.3 * affinity[np.arange(N), item]
    click = (rng.random(N) < 1 / (1 + np.exp(-logit))).astype(np.uint8)
    np.savez(
        d / "random_all.npz",
        item=item,
        position=rng.integers(0, 3, N).astype(np.uint8),
        click=click,
        pscore=np.full(N, 1 / A),
        user=user,
        affinity=affinity,
    )
    p = rng.dirichlet(np.ones(A), N).max(axis=1)
    np.savez(
        d / "bts_all.npz",
        item=rng.integers(0, A, N).astype(np.uint8),
        position=rng.integers(0, 3, N).astype(np.uint8),
        click=(rng.random(N) < 0.1).astype(np.uint8),
        pscore=p,
    )
    np.savez(
        d / "item_all.npz",
        item_id=np.arange(A),
        feature_0=rng.normal(size=A),
        features=rng.integers(0, 2**31, (A, 3)),
    )


def write_criteo(d: Path, n: int = 3000, seed: int = 0) -> None:
    """Arrays in the layout of ``scripts/prepare_criteo.py`` (13 binned ints, 26 categories)."""

    t = synthetic_clicks(n, n_small=13, n_wide=26, seed=seed)
    d.mkdir(parents=True, exist_ok=True)
    t.small.tofile(d / "num.u8")
    t.wide.astype("<u4").tofile(d / "cat.u32")
    t.labels.tofile(d / "label.u8")
    (d / "meta.json").write_text(json.dumps({"rows_kept": n}))


def write_attribution(d: Path, n: int = 50_000, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "cost.npy", rng.lognormal(-9.5, 1.2, n).astype(np.float32))


def write_all(root: Path) -> None:
    (root / "obd").mkdir(parents=True, exist_ok=True)
    write_fixture(root / "obd")
    write_criteo(root / "criteo")
    write_attribution(root / "attribution")
