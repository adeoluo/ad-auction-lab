from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ad_auction_lab.calibration import IsotonicCalibration, PlattScaling
from ad_auction_lab.metrics import (
    calibration,
    delong,
    entropy,
    log_loss,
    midrank,
    normalized_entropy,
    paired_loss_test,
    roc_auc,
    summary,
)


def brute_auc(y: np.ndarray, p: np.ndarray) -> float:
    pos, neg = p[y == 1], p[y == 0]
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


@given(
    st.lists(st.tuples(st.integers(0, 1), st.integers(0, 6)), min_size=2, max_size=60).filter(
        lambda r: 0 < sum(y for y, _ in r) < len(r)
    )
)
@settings(max_examples=150, deadline=None)
def test_auc_matches_brute_force_with_ties(rows: list[tuple[int, int]]) -> None:
    y = np.array([r[0] for r in rows])
    p = np.array([r[1] for r in rows]) / 6.0
    assert roc_auc(y, p) == pytest.approx(brute_auc(y, p))


def test_midrank() -> None:
    assert midrank(np.array([3.0, 1.0, 3.0, 2.0])).tolist() == [3.5, 1.0, 3.5, 2.0]


def test_log_loss_and_entropy() -> None:
    y = np.array([1, 0, 1, 0])
    assert log_loss(y, np.full(4, 0.5)) == pytest.approx(math.log(2))
    assert normalized_entropy(y, np.full(4, 0.5)) == pytest.approx(1.0)
    assert entropy(0.0) >= 0
    assert normalized_entropy(y, np.full(4, 0.5), base_rate=0.25) > 1.0
    assert np.isfinite(log_loss([1, 0], [0.0, 1.0]))


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        log_loss([1, 0], [0.5])
    with pytest.raises(ValueError):
        log_loss([1, 2], [0.5, 0.5])
    with pytest.raises(ValueError):
        log_loss([1, 0], [0.5, 1.5])
    with pytest.raises(ValueError):
        log_loss([1, 0], [0.5, float("nan")])
    with pytest.raises(ValueError, match="both classes"):
        roc_auc([1, 1], [0.2, 0.3])
    with pytest.raises(ValueError):
        calibration([1, 0], [0.5, 0.5], n_bins=0)
    with pytest.raises(ValueError):
        delong([1, 0, 0], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        paired_loss_test([1], [0.5], [0.5])


def test_calibration_of_a_calibrated_model() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.6, 200_000)
    y = (rng.random(len(p)) < p).astype(int)
    c = calibration(y, p, n_bins=10)
    assert c.ratio == pytest.approx(1.0, abs=0.01)
    assert c.ece < 0.01
    assert len(c.bins) == 10
    assert set(c.to_dict()) >= {"ratio", "ece", "bins"}


def test_delong_detects_a_better_model_and_not_a_copy() -> None:
    rng = np.random.default_rng(1)
    n = 4000
    y = rng.integers(0, 2, n)
    good = np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(0, 0.2, n), 0, 1)
    bad = np.clip(0.5 + 0.1 * (y - 0.5) + rng.normal(0, 0.2, n), 0, 1)
    c = delong(y, good, bad)
    assert c.difference > 0 and c.p_value < 1e-6 and c.ci_low > 0
    same = delong(y, good, good)
    assert same.difference == 0 and same.p_value == 1.0


def test_delong_variance_matches_bootstrap() -> None:
    rng = np.random.default_rng(2)
    n = 3000
    y = rng.integers(0, 2, n)
    a = np.clip(0.5 + 0.2 * (y - 0.5) + rng.normal(0, 0.25, n), 0, 1)
    b = np.clip(0.5 + 0.15 * (y - 0.5) + rng.normal(0, 0.25, n), 0, 1)
    c = delong(y, a, b)
    se = (c.ci_high - c.ci_low) / (2 * 1.96)
    diffs = []
    for _ in range(300):
        i = rng.integers(0, n, n)
        diffs.append(roc_auc(y[i], a[i]) - roc_auc(y[i], b[i]))
    assert se == pytest.approx(float(np.std(diffs)), rel=0.2)


def test_paired_loss_test() -> None:
    rng = np.random.default_rng(3)
    p = rng.uniform(0.05, 0.5, 20000)
    y = (rng.random(len(p)) < p).astype(int)
    worse = np.clip(p * 1.5, 0, 1)
    c = paired_loss_test(y, p, worse)
    assert c.difference < 0 and c.p_value < 1e-6
    assert paired_loss_test(y, p, p).p_value == 1.0


def test_summary_keys() -> None:
    s = summary([1, 0, 1, 0], [0.9, 0.2, 0.6, 0.4])
    assert s["auc"] == 1.0 and s["rows"] == 4


def test_platt_recovers_a_miscalibration() -> None:
    rng = np.random.default_rng(4)
    true = rng.uniform(0.02, 0.5, 100_000)
    y = (rng.random(len(true)) < true).astype(int)
    logit = np.log(true / (1 - true))
    skewed = 1 / (1 + np.exp(-(2.0 * logit - 0.5)))
    fixed = PlattScaling().fit(skewed, y).transform(skewed)
    assert log_loss(y, fixed) < log_loss(y, skewed)
    assert calibration(y, fixed).ece < 0.01
    with pytest.raises(ValueError):
        PlattScaling().fit([0.5], [1, 0])


def test_isotonic_is_monotone_and_improves() -> None:
    rng = np.random.default_rng(5)
    true = rng.uniform(0.02, 0.5, 50_000)
    y = (rng.random(len(true)) < true).astype(int)
    skewed = true**2
    iso = IsotonicCalibration().fit(skewed, y)
    out = iso.transform(np.sort(skewed))
    assert np.all(np.diff(out) >= -1e-12)
    assert log_loss(y, np.clip(iso.transform(skewed), 1e-6, 1)) < log_loss(y, skewed)
    with pytest.raises(RuntimeError):
        IsotonicCalibration().transform([0.1])
    with pytest.raises(ValueError):
        IsotonicCalibration().fit([], [])
