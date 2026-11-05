"""Bid shading: exact empirical solver, log-normal landscape and the uniform equilibrium."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ad_auction_lab.bidding import (
    EmpiricalLandscape,
    LogNormalLandscape,
    shading_report,
    surplus,
    symmetric_equilibrium_bid,
)


def _gain(land: EmpiricalLandscape, v: np.ndarray, b: np.ndarray) -> np.ndarray:
    out: np.ndarray = (v - b) * land.win_rate(b)
    return out


@given(
    st.lists(st.floats(0.0, 10.0), min_size=1, max_size=60),
    st.lists(st.floats(0.0, 12.0), min_size=1, max_size=30),
)
@settings(max_examples=200, deadline=None)
def test_empirical_shading_matches_bruteforce(prices: list[float], values: list[float]) -> None:
    land = EmpiricalLandscape.fit(prices)
    v = np.asarray(values)
    fast, slow = land.shade(v), land.shade_bruteforce(v)
    assert np.allclose(_gain(land, v, fast), _gain(land, v, slow), atol=1e-7)


def test_empirical_shading_never_overbids_and_beats_truthful() -> None:
    rng = np.random.default_rng(0)
    land = EmpiricalLandscape.fit(rng.lognormal(0, 1, 5000))
    v = rng.lognormal(0.5, 1, 2000)
    b = land.shade(v)
    assert np.all(b <= v + 1e-6)
    assert np.all(_gain(land, v, b) >= _gain(land, v, v) - 1e-12)
    assert np.all(_gain(land, v, b) >= _gain(land, v, 0.8 * v) - 1e-9)


def test_shade_keeps_shape_and_zero_values() -> None:
    land = EmpiricalLandscape.fit([1.0, 2.0, 3.0])
    out = land.shade(np.array([[0.0, 0.5], [4.0, 10.0]]))
    assert out.shape == (2, 2)
    assert out[0].tolist() == [0.0, 0.0]
    # Value 4: bidding just above 1 wins 1/3 (gain 1), above 2 wins 2/3 (gain 4/3) and above 3
    # wins always (gain 1).
    assert out[1, 0] == pytest.approx(2.0, abs=1e-6)
    assert out[1, 1] == pytest.approx(3.0, abs=1e-6)


def test_win_rate_is_strict() -> None:
    land = EmpiricalLandscape.fit([1.0, 1.0, 2.0, 4.0])
    assert land.win_rate([1.0, 1.0001, 3.0, 5.0]).tolist() == [0.0, 0.5, 0.75, 1.0]


@pytest.mark.parametrize("bad", [[], [float("nan")], [-1.0]])
def test_empirical_rejects_bad_prices(bad: list[float]) -> None:
    with pytest.raises(ValueError):
        EmpiricalLandscape.fit(bad)


@pytest.mark.parametrize("n", [2, 3, 5, 10])
def test_uniform_equilibrium_is_a_best_response(n: int) -> None:
    # Against n - 1 opponents bidding (n-1)/n v with v ~ U(0,1), the best response is (n-1)/n v.
    rng = np.random.default_rng(n)
    opp = symmetric_equilibrium_bid(rng.random((400_000, n - 1)), n).max(axis=1)
    land = EmpiricalLandscape.fit(opp)
    # Values high enough that the sample contains winning opportunities (P(win) >= 0.2 percent).
    v = np.linspace(0.55, 0.95, 9)
    eq = symmetric_equilibrium_bid(v, n)
    shaded = land.shade(v)
    # The surplus curve is flat near the optimum, so compare surplus tightly and bids loosely.
    assert np.all(_gain(land, v, shaded) - _gain(land, v, eq) <= 2e-3 * v)
    assert np.allclose(shaded, eq, atol=0.03)


def test_symmetric_equilibrium_validation() -> None:
    assert symmetric_equilibrium_bid([1.0], 1).tolist() == [0.0]
    with pytest.raises(ValueError):
        symmetric_equilibrium_bid([1.0], 0)


def test_lognormal_fit_recovers_parameters() -> None:
    rng = np.random.default_rng(1)
    land = LogNormalLandscape.fit(rng.lognormal(-2.0, 0.7, 200_000))
    assert land.mu == pytest.approx(-2.0, abs=0.01)
    assert land.sigma == pytest.approx(0.7, abs=0.01)
    assert land.win_rate([np.exp(land.mu)])[0] == pytest.approx(0.5, abs=1e-12)
    assert land.win_rate([0.0])[0] == 0.0


def test_lognormal_shading_matches_grid_optimum() -> None:
    land = LogNormalLandscape(0.0, 0.5)
    for v in (0.5, 1.0, 2.0, 5.0):
        grid = np.linspace(1e-4, v, 200_001)
        best = grid[np.argmax((v - grid) * land.win_rate(grid))]
        assert land.shade([v])[0] == pytest.approx(best, rel=1e-3)


def test_lognormal_shade_interpolates_large_batches() -> None:
    land = LogNormalLandscape(0.0, 0.8)
    v = np.random.default_rng(0).lognormal(0, 1, 20_000)
    approx = land.shade(v, grid_size=2048)
    exact = np.array([land._shade_one(float(x)) for x in v[:200]])
    assert np.allclose(approx[:200], exact, rtol=1e-4)
    assert land.shade([0.0, -1.0]).tolist() == [0.0, 0.0]
    assert land.shade(np.zeros((2, 2))).shape == (2, 2)


def test_lognormal_tiny_values_bid_zero() -> None:
    land = LogNormalLandscape(5.0, 0.1)
    assert land._shade_one(1e-9) >= 0


def test_lognormal_validation() -> None:
    with pytest.raises(ValueError):
        LogNormalLandscape.fit([1.0])
    with pytest.raises(ValueError):
        LogNormalLandscape.fit([2.0, 2.0, 2.0])


def test_surplus_and_report() -> None:
    v = np.array([1.0, 1.0, 1.0])
    m = np.array([0.5, 0.9, 0.2])
    b = np.array([0.6, 0.6, 0.6])
    assert surplus(v, b, m).tolist() == pytest.approx([0.4, 0.0, 0.4])
    rep = shading_report(v, m, {"shaded": b, "truthful": v})
    assert rep["shaded"]["surplus"] == pytest.approx(0.8)
    assert rep["shaded"]["win_rate"] == pytest.approx(2 / 3)
    assert rep["truthful"]["surplus"] == 0.0
    assert rep["truthful"]["spend"] == pytest.approx(3.0)
