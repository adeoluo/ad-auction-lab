"""Auction mechanisms: hand-checked cases, VCG parity with brute force, incentive checks."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ad_auction_lab.auctions import (
    MECHANISMS,
    deviation_gain,
    empirical_reserve,
    gsp_envy_free_bids,
    myerson_reserve_lognormal,
    myerson_reserve_uniform,
    run_auctions,
    vcg_bruteforce,
)

ALPHA = np.array([0.3, 0.2, 0.1])


def test_single_slot_gsp_and_vcg_are_vickrey() -> None:
    bids, q = [5.0, 3.0, 1.0], [1.0, 1.0, 1.0]
    for mech in ("gsp", "vcg"):
        r = run_auctions(bids, q, [1.0], mech)
        assert r.winner.tolist() == [[0]]
        assert r.price[0, 0] == pytest.approx(3.0)
    fp = run_auctions(bids, q, [1.0], "first_price")
    assert fp.price[0, 0] == pytest.approx(5.0)


def test_gsp_textbook_example() -> None:
    # Three bidders, two slots, unit quality: GSP prices are the next bids.
    r = run_auctions([4.0, 2.0, 1.0], [1, 1, 1], [0.2, 0.1], "gsp")
    assert r.winner.tolist() == [[0, 1]]
    assert r.price[0].tolist() == pytest.approx([2.0, 1.0])
    assert r.revenue()[0] == pytest.approx(0.2 * 2 + 0.1 * 1)


def test_vcg_textbook_example() -> None:
    # VCG: slot 2 pays 1 per click; slot 1 pays (0.2 - 0.1) * 2 + 0.1 * 1 = 0.3 in total,
    # which is 1.5 per click.
    r = run_auctions([4.0, 2.0, 1.0], [1, 1, 1], [0.2, 0.1], "vcg")
    assert r.price[0].tolist() == pytest.approx([1.5, 1.0])
    assert r.revenue()[0] == pytest.approx(0.3 + 0.1)


def test_quality_scores_change_rank_and_price() -> None:
    # Bidder 1 bids less per click but has a higher CTR, so it ranks first.
    r = run_auctions([2.0, 1.5], [0.1, 0.2], [1.0], "gsp")
    assert r.winner[0, 0] == 1
    # It pays the smallest bid that keeps its score above 2 * 0.1 = 0.2: 0.2 / 0.2 = 1.0.
    assert r.price[0, 0] == pytest.approx(1.0)


def test_squash_zero_ranks_by_bid() -> None:
    r = run_auctions([2.0, 1.5], [0.1, 0.2], [1.0], "gsp", squash=0.0)
    assert r.winner[0, 0] == 0
    assert r.price[0, 0] == pytest.approx(1.5)


def test_reserve_excludes_and_floors() -> None:
    r = run_auctions([4.0, 2.0, 1.0], [1, 1, 1], [0.2, 0.1], "gsp", reserve=1.5)
    assert r.winner.tolist() == [[0, 1]]
    assert r.price[0].tolist() == pytest.approx([2.0, 1.5])
    r2 = run_auctions([4.0, 1.0], [1, 1], [0.2, 0.1], "gsp", reserve=1.5)
    assert r2.winner.tolist() == [[0, -1]]
    assert r2.price[0].tolist() == pytest.approx([1.5, 0.0])


def test_vcg_reserve_is_seller_cost() -> None:
    r = run_auctions([4.0, 1.0], [1, 1], [1.0], "vcg", reserve=1.5)
    assert r.price[0, 0] == pytest.approx(1.5)
    r = run_auctions([4.0, 3.0], [1, 1], [1.0], "vcg", reserve=1.5)
    assert r.price[0, 0] == pytest.approx(3.0)


def test_empty_auction_and_zero_bids() -> None:
    r = run_auctions([0.0, 0.0], [0.5, 0.5], [0.3, 0.1], "gsp")
    assert (r.winner == -1).all()
    assert r.revenue()[0] == 0
    assert r.welfare([1.0, 1.0])[0] == 0


def test_more_slots_than_bidders() -> None:
    for mech in MECHANISMS:
        r = run_auctions([3.0, 1.0], [1, 1], [0.3, 0.2, 0.1], mech)
        assert r.winner.tolist() == [[0, 1, -1]]
        assert r.price[0, 2] == 0


def test_ties_break_by_index() -> None:
    r = run_auctions([2.0, 2.0, 2.0], [1, 1, 1], [0.2, 0.1], "gsp")
    assert r.winner.tolist() == [[0, 1]]
    assert r.price[0].tolist() == pytest.approx([2.0, 2.0])


def test_batch_matches_single_auctions() -> None:
    rng = np.random.default_rng(0)
    b, q = rng.random((50, 6)), rng.random((50, 6))
    for mech in MECHANISMS:
        batch = run_auctions(b, q, ALPHA, mech, reserve=0.1, squash=0.7)
        for t in range(0, 50, 7):
            one = run_auctions(b[t], q[t], ALPHA, mech, reserve=0.1, squash=0.7)
            assert one.winner[0].tolist() == batch.winner[t].tolist()
            assert np.allclose(one.price[0], batch.price[t])


def test_accounting_identities() -> None:
    rng = np.random.default_rng(1)
    v, q = rng.random((200, 5)), rng.uniform(0.01, 0.2, (200, 5))
    for mech in MECHANISMS:
        r = run_auctions(v, q, ALPHA, mech, reserve=0.05)
        assert np.allclose(r.payments().sum(axis=1), r.revenue())
        assert np.allclose(r.utilities(v).sum(axis=1) + r.revenue(), r.welfare(v))


@pytest.mark.parametrize(
    ("kwargs", "msg"),
    [
        ({"mechanism": "dutch"}, "unknown mechanism"),
        ({"reserve": -1.0}, "reserve"),
        ({"squash": float("nan")}, "squash"),
    ],
)
def test_rejects_bad_parameters(kwargs: dict[str, object], msg: str) -> None:
    with pytest.raises(ValueError, match=msg):
        run_auctions([1.0], [1.0], [1.0], **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad",
    [
        {"bids": [-1.0], "quality": [1.0], "slot_ctr": [1.0]},
        {"bids": [float("inf")], "quality": [1.0], "slot_ctr": [1.0]},
        {"bids": [1.0], "quality": [-0.1], "slot_ctr": [1.0]},
        {"bids": [1.0], "quality": [1.0], "slot_ctr": []},
        {"bids": [1.0], "quality": [1.0], "slot_ctr": [0.1, 0.2]},
        {"bids": [[[1.0]]], "quality": [1.0], "slot_ctr": [1.0]},
    ],
)
def test_rejects_bad_inputs(bad: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        run_auctions(**bad)  # type: ignore[arg-type]


bids_st = st.lists(st.floats(0.0, 10.0), min_size=1, max_size=6)


@given(
    st.integers(0, 10_000),
    st.integers(1, 6),
    st.integers(1, 4),
    st.sampled_from([0.0, 0.3, 1.0]),
    st.sampled_from([0.0, 0.5, 1.0]),
)
@settings(max_examples=150, deadline=None)
def test_vcg_closed_form_matches_bruteforce(
    seed: int, n: int, k: int, reserve: float, squash: float
) -> None:
    rng = np.random.default_rng(seed)
    b = np.round(rng.uniform(0, 3, n), 3)
    q = np.round(rng.uniform(0.05, 1, n), 3)
    alpha = np.sort(rng.uniform(0.05, 1, k))[::-1]
    r = run_auctions(b, q, alpha, "vcg", reserve, squash)
    w_ref, p_ref = vcg_bruteforce(b, q, alpha, reserve, squash)
    w_ref_val = np.where(w_ref >= 0, (q**squash * (b - reserve))[np.maximum(w_ref, 0)], 0)
    w_val = np.where(r.winner[0] >= 0, (q**squash * (b - reserve))[np.maximum(r.winner[0], 0)], 0)
    # Same weighted welfare (allocations may differ only between exactly tied bidders).
    assert np.dot(alpha, w_val) == pytest.approx(np.dot(alpha, w_ref_val), abs=1e-9)
    if len(set(np.round(q**squash * (b - reserve), 12))) == n:
        assert r.winner[0].tolist() == w_ref.tolist()
        assert np.allclose(r.price[0], p_ref, atol=1e-9)


def test_bruteforce_limits() -> None:
    with pytest.raises(ValueError, match="8 bidders"):
        vcg_bruteforce(np.ones(9), np.ones(9), [1.0])


@given(st.integers(0, 10_000), st.integers(2, 6), st.sampled_from([0.0, 0.2]))
@settings(max_examples=60, deadline=None)
def test_vcg_is_truthful(seed: int, n: int, reserve: float) -> None:
    rng = np.random.default_rng(seed)
    v = rng.uniform(0, 2, (20, n))
    q = rng.uniform(0.02, 0.3, (20, n))
    for squash in (1.0, 0.5):
        for i in range(n):
            gain = deviation_gain(v, q, ALPHA, "vcg", i, reserve=reserve, squash=squash)
            assert gain.max() <= 1e-12


def test_gsp_and_first_price_are_not_truthful() -> None:
    rng = np.random.default_rng(3)
    v, q = rng.uniform(0, 2, (500, 5)), rng.uniform(0.02, 0.3, (500, 5))
    gsp = deviation_gain(v, q, ALPHA, "gsp", 0)
    fp = deviation_gain(v, q, ALPHA, "first_price", 0)
    assert (gsp > 1e-9).mean() > 0.01
    assert (fp > 1e-9).mean() > 0.3
    # With a single slot GSP is the Vickrey auction, so it is truthful.
    assert deviation_gain(v, q, [0.3], "gsp", 0).max() <= 1e-12


def test_deviation_gain_with_fixed_others() -> None:
    gain = deviation_gain([[1.0, 0.5]], [[1, 1]], [1.0], "first_price", 0, others=[[0.0, 0.55]])
    # Truthful bid wins and earns 0; bidding 0.7 wins at 0.7 and earns 0.3.
    assert gain[0] == pytest.approx(0.3, abs=0.01)


@given(st.integers(0, 10_000), st.integers(3, 7), st.integers(1, 4))
@settings(max_examples=100, deadline=None)
def test_eos_equilibrium_matches_vcg_revenue(seed: int, n: int, k: int) -> None:
    rng = np.random.default_rng(seed)
    v = rng.uniform(0.1, 5, n)
    q = rng.uniform(0.05, 0.5, n)
    alpha = np.sort(rng.uniform(0.05, 1, k))[::-1] * np.linspace(1, 0.5, k)
    bids = gsp_envy_free_bids(v, q, alpha)
    gsp = run_auctions(bids, q, alpha, "gsp")
    vcg = run_auctions(v, q, alpha, "vcg")
    assert gsp.winner.tolist() == vcg.winner.tolist()
    assert gsp.revenue()[0] == pytest.approx(vcg.revenue()[0], rel=1e-9, abs=1e-12)
    # Every bid is at most the value (no overbidding in the lowest envy-free equilibrium).
    assert np.all(bids <= v + 1e-12)


def test_eos_bids_are_a_nash_equilibrium() -> None:
    rng = np.random.default_rng(7)
    v, q = rng.uniform(0.5, 5, 6), rng.uniform(0.05, 0.5, 6)
    bids = gsp_envy_free_bids(v, q, ALPHA)
    for i in range(6):
        gain = deviation_gain(
            v[None, :],
            q[None, :],
            ALPHA,
            "gsp",
            i,
            others=bids[None, :],
            factors=np.linspace(0, 3, 301),
        )
        # Bidders compare against their own value as the baseline, so measure the utility at the
        # equilibrium bid directly.
        eq = run_auctions(bids, q, ALPHA, "gsp").utilities(v)[0, i]
        truthful = run_auctions(np.where(np.arange(6) == i, v, bids), q, ALPHA, "gsp").utilities(v)[
            0, i
        ]
        assert truthful + gain[0] <= eq + 1e-9


def test_myerson_uniform_revenue_five_twelfths() -> None:
    # Two Uniform(0,1) bidders, second price with reserve 1/2: revenue 5/12 (Myerson, 1981).
    rng = np.random.default_rng(0)
    v = rng.random((400_000, 2))
    r = myerson_reserve_uniform()
    rev = run_auctions(v, np.ones_like(v), [1.0], "gsp", reserve=r).revenue().mean()
    assert r == 0.5
    assert rev == pytest.approx(5 / 12, abs=0.002)
    no_reserve = run_auctions(v, np.ones_like(v), [1.0], "gsp").revenue().mean()
    assert no_reserve == pytest.approx(1 / 3, abs=0.002)


def test_myerson_lognormal_matches_grid_search() -> None:
    mu, sigma = -1.0, 0.6
    r = myerson_reserve_lognormal(mu, sigma)
    rng = np.random.default_rng(1)
    vals = rng.lognormal(mu, sigma, 2_000_000)
    grid = np.linspace(0.05, 1.5, 2000)
    best, _ = empirical_reserve(vals, grid)
    assert r == pytest.approx(best, rel=0.03)


def test_reserve_validation() -> None:
    with pytest.raises(ValueError):
        myerson_reserve_uniform(0)
    with pytest.raises(ValueError):
        myerson_reserve_lognormal(0, 0)
    with pytest.raises(ValueError):
        empirical_reserve([])


def test_empirical_reserve_exact() -> None:
    r, rev = empirical_reserve([1.0, 2.0, 3.0, 10.0])
    # Posting 10 sells to one of four: 2.5; posting 3 sells to two: 1.5.
    assert r == 10.0
    assert rev == pytest.approx(2.5)


@given(st.lists(st.floats(0.0, 100.0), min_size=1, max_size=50))
@settings(max_examples=100, deadline=None)
def test_empirical_reserve_is_optimal(vals: list[float]) -> None:
    r, rev = empirical_reserve(vals)
    v = np.asarray(vals)
    for c in v:
        assert c * np.mean(v >= c) <= rev + 1e-9
    assert rev == pytest.approx(r * np.mean(v >= r))


@given(bids_st)
@settings(max_examples=100, deadline=None)
def test_prices_bounded_by_bids_in_gsp_and_vcg(bids: list[float]) -> None:
    q = np.linspace(0.1, 0.9, len(bids))
    for mech in ("gsp", "vcg"):
        r = run_auctions(bids, q, ALPHA, mech, reserve=0.5)
        b = np.asarray(bids)
        for j, w in enumerate(r.winner[0]):
            if w >= 0:
                assert 0.5 - 1e-12 <= r.price[0, j] <= b[w] + 1e-9
    # Individual rationality: truthful bidders never get negative utility.
    for mech in ("gsp", "vcg", "first_price"):
        r = run_auctions(bids, q, ALPHA, mech, reserve=0.5)
        assert np.all(r.utilities(bids) >= -1e-9)


@given(st.integers(0, 1000))
@settings(max_examples=50, deadline=None)
def test_vcg_revenue_at_most_gsp_truthful_revenue(seed: int) -> None:
    # With truthful bids, GSP charges at least VCG in every auction (Edelman et al., 2007).
    rng = np.random.default_rng(seed)
    v, q = rng.uniform(0, 3, (30, 5)), rng.uniform(0.05, 0.4, (30, 5))
    g = run_auctions(v, q, ALPHA, "gsp").revenue()
    c = run_auctions(v, q, ALPHA, "vcg").revenue()
    assert np.all(c <= g + 1e-12)
