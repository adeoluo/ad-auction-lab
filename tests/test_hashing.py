from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from ad_auction_lab.hashing import bucket, hash_rows, hash_string, mix64


def test_mix64_known_values() -> None:
    assert int(mix64(np.array([0], dtype=np.uint64))[0]) == 0
    # SplitMix64 finalizer of 1, computed independently with Python integers.
    x = 1
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) % 2**64
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) % 2**64
    x ^= x >> 31
    assert int(mix64(np.array([1], dtype=np.uint64))[0]) == x


@given(st.integers(0, 60), st.integers(0, 2**32 - 1), st.integers(1, 28))
@settings(max_examples=200, deadline=None)
def test_bucket_in_range_and_deterministic(field: int, value: int, bits: int) -> None:
    b = bucket(field, np.array([value]), bits)
    assert 0 <= int(b[0]) < 2**bits
    assert int(bucket(field, np.array([value]), bits)[0]) == int(b[0])


def test_hash_rows_shape_and_field_separation() -> None:
    small = np.array([[1, 1], [2, 3]], dtype=np.uint8)
    wide = np.array([[1], [5]], dtype=np.uint32)
    idx = hash_rows(small, wide, 20)
    assert idx.shape == (2, 3)
    # the same value in different fields lands in different buckets
    assert idx[0, 0] != idx[0, 1] != idx[0, 2]
    only_wide = hash_rows(np.zeros((2, 0), np.uint8), wide, 20)
    assert only_wide.shape == (2, 1)
    assert only_wide[0, 0] == bucket(0, np.array([1]), 20)[0]


def test_hash_rows_spreads_uniformly() -> None:
    rng = np.random.default_rng(0)
    wide = rng.integers(0, 2**32, size=(20000, 1), dtype=np.uint32)
    idx = hash_rows(np.zeros((20000, 0), np.uint8), wide, 4)
    counts = np.bincount(idx[:, 0], minlength=16)
    assert counts.min() > 1000 and counts.max() < 1500


def test_hash_string_is_fnv1a() -> None:
    out = hash_string(["", "a"])
    assert out[0] == 0x811C9DC5
    assert out[1] == 0xE40C292C
