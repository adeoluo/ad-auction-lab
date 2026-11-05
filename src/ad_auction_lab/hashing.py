"""Feature hashing shared with the Rust core.

Token ``(field, value)`` maps to bucket ``mix64(((field + 1) << 32) | value) & (2^bits - 1)``,
where ``mix64`` is the SplitMix64 finalizer. Fields run over the ``small`` (uint8) block first and
then over the ``wide`` (uint32) block, as in ``rust/src/hashing.rs``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)


def mix64(x: NDArray[np.uint64]) -> NDArray[np.uint64]:
    """SplitMix64 finalizer, vectorized (uint64 arithmetic wraps modulo 2^64)."""

    x = np.asarray(x, dtype=np.uint64)
    with np.errstate(over="ignore"):
        x = x ^ (x >> np.uint64(30))
        x = x * _M1
        x = x ^ (x >> np.uint64(27))
        x = x * _M2
    return x ^ (x >> np.uint64(31))


def bucket(
    field: int | NDArray[np.integer], value: NDArray[np.integer], bits: int
) -> NDArray[np.int64]:
    """Hashed bucket of ``(field, value)`` in a table of ``2^bits`` weights."""

    f = (np.asarray(field, dtype=np.uint64) + np.uint64(1)) << np.uint64(32)
    v = np.asarray(value).astype(np.uint64)
    mask = np.uint64((1 << bits) - 1)
    return (mix64(f | v) & mask).astype(np.int64)


def hash_rows(small: NDArray[np.uint8], wide: NDArray[np.uint32], bits: int) -> NDArray[np.int64]:
    """Buckets of every token, shape ``(n_rows, n_small + n_wide)``."""

    n_small = small.shape[1]
    parts = []
    if n_small:
        parts.append(bucket(np.arange(n_small), small, bits))
    if wide.shape[1]:
        parts.append(bucket(np.arange(n_small, n_small + wide.shape[1]), wide, bits))
    return np.concatenate(parts, axis=1)


def hash_string(values: list[str] | NDArray[np.str_]) -> NDArray[np.uint32]:
    """Stable 32-bit ids for strings (FNV-1a), for categorical columns given as text."""

    out = np.empty(len(values), dtype=np.uint32)
    for i, s in enumerate(values):
        h = 0x811C9DC5
        for b in str(s).encode():
            h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
        out[i] = h
    return out
