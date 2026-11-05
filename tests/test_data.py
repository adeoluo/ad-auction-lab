from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from ad_auction_lab.data import (
    Table,
    bin_integer,
    load_criteo,
    parse_criteo,
    parse_criteo_line,
    split_rows,
    synthetic_clicks,
)

ROW = "1\t5\t\t3\t-1\t" + "\t".join(["7"] * 9) + "\t" + "\t".join(["68fd1e64"] * 25) + "\t"


def test_bin_integer_buckets() -> None:
    assert bin_integer(-3) == 1
    assert [bin_integer(v) for v in (0, 1, 2)] == [2, 3, 4]
    assert bin_integer(3) == int(np.log(3) ** 2) + 5
    assert bin_integer(10**12) == 255


@given(st.integers(-(10**9), 10**12))
def test_bin_integer_range_and_monotone(v: int) -> None:
    b = bin_integer(v)
    assert 1 <= b <= 255
    if v >= 0:
        assert bin_integer(v + 1) >= b


def test_parse_line() -> None:
    y, nums, cats = parse_criteo_line(ROW)
    assert y == 1
    assert nums[:4] == [bin_integer(5), 0, bin_integer(3), 1]
    assert cats[0] == 0x68FD1E64 and cats[-1] == 0
    t = parse_criteo([ROW, "", ROW])
    assert len(t) == 2 and t.n_fields == 39


def test_parse_unlabelled_and_errors() -> None:
    _, nums, _ = parse_criteo_line(ROW.split("\t", 1)[1], labelled=False)
    assert len(nums) == 13
    with pytest.raises(ValueError, match="fields"):
        parse_criteo_line("1\t2")
    with pytest.raises(ValueError, match="label"):
        parse_criteo_line("2" + ROW[1:])
    with pytest.raises(ValueError, match="hexadecimal"):
        parse_criteo_line(ROW.replace("68fd1e64", "68fd1e6400", 1))
    with pytest.raises(ValueError):
        parse_criteo_line(ROW.replace("68fd1e64", "zz", 1))
    with pytest.raises(ValueError, match="no rows"):
        parse_criteo([""])


def test_table_validation() -> None:
    with pytest.raises(ValueError, match="row count"):
        Table(np.zeros((2, 1), np.uint8), np.zeros((3, 1), np.uint32), np.zeros(2, np.uint8))
    with pytest.raises(ValueError, match="at least one"):
        Table(np.zeros((2, 0), np.uint8), np.zeros((2, 0), np.uint32), np.zeros(2, np.uint8))
    with pytest.raises(ValueError, match="uint8"):
        Table(np.zeros((2, 1), np.int64), np.zeros((2, 1), np.uint32), np.zeros(2, np.uint8))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="2-D"):
        Table(np.zeros(2, np.uint8), np.zeros((2, 1), np.uint32), np.zeros(2, np.uint8))


def test_split_rows_disjoint_and_complete() -> None:
    parts = split_rows(1000, (0.8, 0.1, 0.1), seed=3)
    allr = np.concatenate(parts)
    assert sorted(allr.tolist()) == list(range(1000))
    assert [len(p) for p in parts] == [800, 100, 100]
    assert len(split_rows(1000, (0.5,), seed=0)[0]) == 500
    with pytest.raises(ValueError):
        split_rows(10, (0.9, 0.2))
    with pytest.raises(ValueError):
        split_rows(0)


def test_load_criteo_roundtrip(tmp_path: Path) -> None:
    t = synthetic_clicks(50, n_small=13, n_wide=26, seed=2)
    (tmp_path / "num.u8").write_bytes(t.small.tobytes())
    (tmp_path / "cat.u32").write_bytes(t.wide.astype("<u4").tobytes())
    (tmp_path / "label.u8").write_bytes(t.labels.tobytes())
    (tmp_path / "meta.json").write_text(json.dumps({"rows_kept": 50}))
    back = load_criteo(tmp_path)
    assert np.array_equal(back.small, t.small) and np.array_equal(back.wide, t.wide)
    assert np.array_equal(back.labels, t.labels)
    sub = back.take(np.array([3, 1]))
    assert np.array_equal(sub.labels, t.labels[[3, 1]])


def test_synthetic_has_signal() -> None:
    t = synthetic_clicks(5000, seed=0)
    assert 0.05 < t.labels.mean() < 0.6
