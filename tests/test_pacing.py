"""Budget pacing: Rust vs Python parity, budget safety, pacing quality and the hindsight bound."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ad_auction_lab.backend import rust_available
from ad_auction_lab.pacing import (
    Pacer,
    build_market,
    hindsight_optimum,
    simulate,
    simulate_reference,
)

needs_rust = pytest.mark.skipif(not rust_available(), reason="Rust extension not built")


def market(T: int, n: int, seed: int = 0, budget_share: float = 0.3):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    values = rng.uniform(0.5, 2.0, n)
    ctr = rng.beta(2, 50, (T, n)).astype(np.float32)
    mk = (rng.lognormal(-4.0, 0.7, T)).astype(np.float32)
    unpaced = simulate_reference(values, np.full(n, 1e9), ctr, mk, Pacer("none"))
    budgets = np.maximum(unpaced.spend * budget_share, 1e-6)
    return values, budgets, ctr, mk


PACERS = [
    Pacer("none"),
    Pacer("dual", step=0.01),
    Pacer("pid", kp=3.0, ki=0.1, kd=0.5, interval=50, m_min=0.05),
]


@needs_rust
@given(
    st.integers(0, 10_000), st.integers(1, 5), st.sampled_from(PACERS), st.sampled_from([0, 0.01])
)
@settings(max_examples=40, deadline=None)
def test_rust_matches_reference(seed: int, n: int, pacer: Pacer, reserve: float) -> None:
    v, b, c, m = market(800, n, seed)
    a = simulate(v, b, c, m, pacer, reserve=reserve, checkpoints=7, backend="rust")
    r = simulate(v, b, c, m, pacer, reserve=reserve, checkpoints=7, backend="python")
    assert a.wins.tolist() == r.wins.tolist()
    assert a.exhausted_at.tolist() == r.exhausted_at.tolist()
    assert np.allclose(a.spend, r.spend, rtol=1e-12, atol=1e-15)
    assert np.allclose(a.value, r.value, rtol=1e-12, atol=1e-15)
    assert np.allclose(a.trajectory, r.trajectory, rtol=1e-12, atol=1e-15)
    assert a.market_wins == r.market_wins and a.unsold == r.unsold
    assert a.revenue == pytest.approx(r.revenue, rel=1e-12)


@pytest.mark.parametrize("pacer", PACERS, ids=lambda p: p.kind)
def test_budget_is_never_exceeded_and_accounts_add_up(pacer: Pacer) -> None:
    v, b, c, m = market(3000, 4, 1)
    r = simulate(v, b, c, m, pacer)
    assert np.all(r.spend <= b + 1e-12)
    assert r.wins.sum() + r.market_wins + r.unsold == 3000
    assert r.revenue == pytest.approx(r.spend.sum())
    assert np.allclose(r.trajectory[-1], r.spend)
    assert np.all(np.diff(r.trajectory, axis=0) >= -1e-15)


def test_pacing_spreads_spend_and_beats_no_pacing() -> None:
    v, b, c, m = market(20_000, 1, 2, budget_share=0.25)
    none = simulate(v, b, c, m, Pacer("none")).summary()
    dual = simulate(v, b, c, m, Pacer("dual", step=0.005)).summary()
    pid = simulate(v, b, c, m, Pacer("pid", kp=3.0, ki=0.05, interval=100)).summary()
    assert none["exhausted_share"] == 1.0
    assert none["mean_exhaustion_time"] < 0.5
    for paced in (dual, pid):
        assert paced["max_tracking_error"] < none["max_tracking_error"]
        assert paced["value"] > none["value"]
        assert paced["utilization"] > 0.8


def test_dual_descent_approaches_hindsight_optimum() -> None:
    rng = np.random.default_rng(5)
    T = 50_000
    ctr = rng.beta(2, 40, (T, 1)).astype(np.float32)
    mk = rng.lognormal(-4.0, 0.8, T).astype(np.float32)
    v = np.array([1.0])
    budget = 0.2 * float(np.minimum(ctr[:, 0], 1e9).astype(np.float64) @ np.ones(T)) * 0.05
    r = simulate(v, [budget], ctr, mk, Pacer("dual", step=0.002))
    opt = hindsight_optimum(ctr[:, 0].astype(np.float64) * v[0], mk.astype(np.float64), budget)
    assert r.value[0] <= opt["value"] + 1e-12
    assert r.value[0] / opt["value"] > 0.9


def test_hindsight_optimum_small_case() -> None:
    out = hindsight_optimum([3.0, 2.0, 1.0, 0.5], [1.0, 1.0, 1.0, 0.0], 1.5)
    # Free impression (0.5) + best ratio (3) + half of the next (1).
    assert out["value"] == pytest.approx(0.5 + 3.0 + 1.0)
    assert out["spend"] == pytest.approx(1.5)
    assert out["multiplier"] == pytest.approx(0.5)
    rich = hindsight_optimum([3.0, 2.0], [1.0, 1.0], 10.0)
    assert rich["value"] == 5.0 and rich["multiplier"] == 1.0


@given(
    st.lists(st.tuples(st.floats(0, 5), st.floats(0.01, 5)), min_size=1, max_size=30),
    st.floats(0, 20),
)
@settings(max_examples=100, deadline=None)
def test_hindsight_optimum_dominates_greedy_subsets(
    pairs: list[tuple[float, float]], b: float
) -> None:
    val = np.array([p[0] for p in pairs])
    pr = np.array([p[1] for p in pairs])
    out = hindsight_optimum(val, pr, b)
    assert out["spend"] <= b + 1e-9
    # Any integral prefix in arbitrary order is feasible and no better than the LP.
    perm = np.random.default_rng(0).permutation(len(val))
    k = np.searchsorted(np.cumsum(pr[perm]), b, side="right")
    assert val[perm[:k]].sum() <= out["value"] + 1e-9


def test_hindsight_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        hindsight_optimum([1.0], [1.0, 2.0], 1.0)
    with pytest.raises(ValueError):
        hindsight_optimum([1.0], [-1.0], 1.0)


def test_reserve_and_market_wins() -> None:
    v = np.array([1.0])
    c = np.array([[0.1], [0.1], [0.1]], dtype=np.float32)
    m = np.array([0.05, 0.2, 0.0], dtype=np.float32)
    r = simulate(v, [10.0], c, m, Pacer("none"), reserve=0.0)
    assert r.wins.tolist() == [2]
    assert r.market_wins == 1
    r = simulate(v, [10.0], c, m, Pacer("none"), reserve=0.5)
    assert r.wins.tolist() == [0] and r.unsold == 3 and r.market_wins == 0


def test_zero_budget_advertiser_never_wins() -> None:
    v, b, c, m = market(500, 2, 3)
    b = b.copy()
    b[1] = 0.0
    for p in PACERS:
        r = simulate(v, b, c, m, p)
        assert r.wins[1] == 0
        assert r.summary()["utilization"] <= 1.0


@pytest.mark.parametrize("backend", ["python", "auto"])
def test_validation_errors(backend: str) -> None:
    c = np.ones((3, 1), dtype=np.float32)
    m = np.ones(3, dtype=np.float32)
    with pytest.raises(ValueError):
        simulate([1.0], [1.0], np.ones((3, 2)), m, backend=backend)
    with pytest.raises(ValueError):
        simulate([-1.0], [1.0], c, m, backend=backend)
    with pytest.raises(ValueError):
        simulate([1.0], [1.0], c, m, reserve=-1, backend=backend)
    with pytest.raises(ValueError):
        simulate([], [], np.ones((3, 0)), m, backend=backend)


def test_pacer_validation() -> None:
    with pytest.raises(ValueError):
        Pacer("bang-bang")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Pacer("pid", interval=0)
    with pytest.raises(ValueError):
        Pacer("dual", step=-1)


def test_one_dimensional_ctr_and_summary_fields() -> None:
    r = simulate([1.0], [1.0], np.full(10, 0.1), np.full(10, 0.01), Pacer("none"), checkpoints=0)
    s = r.summary()
    assert r.trajectory.shape == (1, 1)
    assert set(s) >= {"value", "spend", "roi", "utilization", "max_tracking_error"}
    assert s["roi"] == pytest.approx(10.0)


def test_build_market_from_real_style_samples() -> None:
    rng = np.random.default_rng(3)
    prices = rng.lognormal(-5, 1, 5000)
    pool = rng.beta(2, 60, 3000)
    m = build_market(prices, pool, horizon=2000, n_advertisers=4, budget_share=0.4, seed=1)
    assert m.market.shape == (2000,) and m.ctr.shape == (2000, 4)
    assert np.allclose(m.budgets, 0.4 * m.unpaced_spend)
    # The window is contiguous in the price stream.
    start = int(np.flatnonzero(np.isclose(prices.astype(np.float32), m.market[0]))[0])
    assert np.allclose(prices[start : start + 2000].astype(np.float32), m.market)
    paced = simulate(m.values, m.budgets, m.ctr, m.market, Pacer("dual", step=0.01))
    assert np.all(paced.spend <= m.budgets * (1 + 1e-6) + 1e-9)
    again = build_market(prices, pool, horizon=2000, n_advertisers=4, budget_share=0.4, seed=1)
    assert np.array_equal(again.values, m.values)


@pytest.mark.parametrize(
    ("horizon", "n", "share", "pool"),
    [(0, 2, 0.3, 10), (10**6, 2, 0.3, 10), (10, 0, 0.3, 10), (10, 2, 0.0, 10), (10, 2, 0.3, 0)],
)
def test_build_market_rejects_bad_inputs(horizon: int, n: int, share: float, pool: int) -> None:
    with pytest.raises(ValueError):
        build_market(np.ones(100), np.full(pool, 0.01), horizon, n, budget_share=share)
