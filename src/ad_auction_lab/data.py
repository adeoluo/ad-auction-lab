"""Tabular click data as two categorical blocks, and the Criteo loaders.

A :class:`Table` stores ``small`` (``uint8``, one column per binned integer feature), ``wide``
(``uint32``, one column per categorical feature) and ``labels`` (``uint8``). The Criteo arrays
written by ``scripts/prepare_criteo.py`` are memory-mapped, so 11.46 million rows (1.35 GB) can be
trained on without loading them into memory.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

CRITEO_INT = 13
CRITEO_CAT = 26


@dataclass(frozen=True)
class Table:
    """Row-major categorical features and binary labels."""

    small: NDArray[np.uint8]
    wide: NDArray[np.uint32]
    labels: NDArray[np.uint8]

    def __post_init__(self) -> None:
        n = len(self.labels)
        if self.small.ndim != 2 or self.wide.ndim != 2:
            raise ValueError("small and wide must be 2-D")
        if self.small.shape[0] != n or self.wide.shape[0] != n:
            raise ValueError("small, wide and labels disagree on the row count")
        if self.small.shape[1] + self.wide.shape[1] == 0:
            raise ValueError("a table needs at least one feature column")
        if self.small.dtype != np.uint8 or self.wide.dtype != np.uint32:
            raise ValueError("small must be uint8 and wide uint32")

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def n_fields(self) -> int:
        return int(self.small.shape[1] + self.wide.shape[1])

    def take(self, rows: NDArray[np.integer]) -> Table:
        """A copy restricted to ``rows``."""

        return Table(
            np.ascontiguousarray(self.small[rows]),
            np.ascontiguousarray(self.wide[rows]),
            np.ascontiguousarray(self.labels[rows]),
        )


def bin_integer(v: int) -> int:
    """Log-square bucketing of a Criteo integer feature, as used by the Kaggle winners.

    0 is reserved for missing values, negatives map to 1, 0 to 2 map to 2 to 4 and larger values
    to ``floor(ln(v)^2) + 5``, capped at 255.
    """

    if v < 0:
        return 1
    if v <= 2:
        return v + 2
    return min(int(math.log(v) ** 2) + 5, 255)


def parse_criteo_line(line: str, labelled: bool = True) -> tuple[int, list[int], list[int]]:
    """Parse one tab-separated Criteo row into (label, 13 integer bins, 26 categorical ids)."""

    parts = line.rstrip("\r\n").split("\t")
    expected = CRITEO_INT + CRITEO_CAT + (1 if labelled else 0)
    if len(parts) != expected:
        raise ValueError(f"expected {expected} tab-separated fields, got {len(parts)}")
    label = 0
    if labelled:
        if parts[0] not in ("0", "1"):
            raise ValueError("label must be 0 or 1")
        label = int(parts[0])
        parts = parts[1:]
    nums = [bin_integer(int(s)) if s else 0 for s in parts[:CRITEO_INT]]
    cats = []
    for s in parts[CRITEO_INT:]:
        if not s:
            cats.append(0)
            continue
        if len(s) > 8:
            raise ValueError("categorical values are 32-bit hexadecimal strings")
        cats.append(int(s, 16))
    return label, nums, cats


def parse_criteo(lines: Iterable[str], labelled: bool = True) -> Table:
    """Parse Criteo text rows (the Kaggle ``train.txt`` format) into a :class:`Table`."""

    labels, nums, cats = [], [], []
    for line in lines:
        if not line.strip():
            continue
        y, n, c = parse_criteo_line(line, labelled)
        labels.append(y)
        nums.append(n)
        cats.append(c)
    if not labels:
        raise ValueError("no rows")
    return Table(
        np.asarray(nums, dtype=np.uint8),
        np.asarray(cats, dtype=np.uint32),
        np.asarray(labels, dtype=np.uint8),
    )


def load_criteo(directory: str | Path) -> Table:
    """Memory-map the arrays written by ``scripts/prepare_criteo.py``."""

    d = Path(directory)
    meta = json.loads((d / "meta.json").read_text())
    n = int(meta["rows_kept"])
    small = np.memmap(d / "num.u8", dtype=np.uint8, mode="r", shape=(n, CRITEO_INT))
    wide = np.memmap(d / "cat.u32", dtype="<u4", mode="r", shape=(n, CRITEO_CAT))
    labels = np.memmap(d / "label.u8", dtype=np.uint8, mode="r", shape=(n,))
    return Table(small, wide, labels)


def split_rows(
    n: int, fractions: tuple[float, ...] = (0.8, 0.1, 0.1), seed: int = 0
) -> list[NDArray[np.uint64]]:
    """Random disjoint row sets with the given fractions (each sorted, ``uint64``)."""

    if n <= 0 or any(f <= 0 for f in fractions) or sum(fractions) > 1 + 1e-9:
        raise ValueError("n must be positive and the fractions positive with sum at most 1")
    perm = np.random.default_rng(seed).permutation(n).astype(np.uint64)
    out, start = [], 0
    for i, f in enumerate(fractions):
        stop = (
            n if i == len(fractions) - 1 and abs(sum(fractions) - 1) < 1e-9 else start + int(f * n)
        )
        out.append(np.sort(perm[start:stop]))
        start = stop
    return out


def synthetic_clicks(
    n: int, n_small: int = 4, n_wide: int = 6, seed: int = 0, base_logit: float = -1.5
) -> Table:
    """Synthetic click data with planted main effects and one pairwise interaction.

    Used by the tests and the CI fixtures; real-data runs use Criteo.
    """

    rng = np.random.default_rng(seed)
    small = rng.integers(0, 12, size=(n, n_small), dtype=np.uint8)
    card = np.array([50, 200, 1000, 20, 5000, 7][:n_wide], dtype=np.int64)
    card = np.resize(card, n_wide)
    wide = (rng.zipf(1.3, size=(n, n_wide)) % card).astype(np.uint32)
    effect_small = rng.normal(0, 0.4, size=(n_small, 256))
    effect_wide = rng.normal(0, 0.5, size=(n_wide, int(np.max(card))))
    logit = np.full(n, base_logit)
    for j in range(n_small):
        logit += effect_small[j, small[:, j]]
    for j in range(n_wide):
        logit += effect_wide[j, wide[:, j]]
    if n_small and n_wide:
        logit += 0.8 * ((small[:, 0] % 2) == (wide[:, 0] % 2)) - 0.4
    p = 1 / (1 + np.exp(-logit))
    labels = (rng.random(n) < p).astype(np.uint8)
    return Table(small, wide, labels)
