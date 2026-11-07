"""Off-policy estimators: closed forms, unbiasedness on a simulated bandit, bootstrap, helpers."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ad_auction_lab.ope import (
    ESTIMATORS,
    BanditLog,
    RoundTerms,
    bootstrap,
    empirical_action_dist,
    estimate,
    importance_weights,
    item_position_model,
    marginal_action_dist,
    model_terms,
    relative_error,
    round_terms,
    uniform_action_dist,
)


def simulated(n: int, A: int = 5, K: int = 2, seed: int = 0):  # type: ignore[no-untyped-def]
    """A uniform logging policy over A items at K positions with known click rates."""

    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.01, 0.3, (A, K))
    a = rng.integers(0, A, n)
    k = rng.integers(0, K, n)
    r = (rng.random(n) < theta[a, k]).astype(float)
    log = BanditLog.from_arrays(r, a, np.full(n, 1.0 / A), k, A, K)
    return log, theta


def target_policy(A: int, K: int, seed: int = 1) -> np.ndarray:
    pi = np.random.default_rng(seed).dirichlet(np.ones(A), K).T
    out: np.ndarray = pi
    return out


def true_value(theta: np.ndarray, pi: np.ndarray) -> float:
    # Positions are uniform in the log, so the value averages the per-position values.
    return float(np.mean((pi * theta).sum(axis=0)))


def test_hand_computed_estimates() -> None:
    log = BanditLog.from_arrays([1, 0, 1, 0], [0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5])
    pi = np.array([[0.8], [0.2]])
    q = np.array([[0.6], [0.3]])
    w = importance_weights(log, pi)
    assert w.tolist() == pytest.approx([1.6, 0.4, 0.4, 1.6])
    assert estimate("ips", log, pi) == pytest.approx((1.6 + 0.4) / 4)
    assert estimate("snips", log, pi) == pytest.approx(2.0 / 4.0)
    dm = 0.8 * 0.6 + 0.2 * 0.3
    assert estimate("dm", log, pi, q) == pytest.approx(dm)
    corr = np.array([1.6 * 0.4, 0.4 * -0.3, 0.4 * 0.7, 1.6 * -0.6])
    assert estimate("dr", log, pi, q) == pytest.approx(dm + corr.mean())
    assert estimate("sndr", log, pi, q) == pytest.approx(dm + corr.sum() / w.sum())
    # Switch-DR with tau = 1 drops the correction of the two rounds with weight 1.6.
    sw = dm + np.array([0, corr[1], corr[2], 0]).mean()
    assert estimate("switch-dr", log, pi, q, tau=1.0) == pytest.approx(sw)
    assert estimate("ips", log, pi, clip=1.0) == pytest.approx((1.0 + 0.4) / 4)


@pytest.mark.parametrize("name", ESTIMATORS)
def test_estimators_recover_the_true_value(name: str) -> None:
    log, theta = simulated(400_000, seed=3)
    pi = target_policy(5, 2)
    q = item_position_model(log)
    v = estimate(name, log, pi, q, tau=50.0)  # type: ignore[arg-type]
    assert v == pytest.approx(true_value(theta, pi), rel=0.02)


def test_dr_is_unbiased_with_a_wrong_model() -> None:
    log, theta = simulated(400_000, seed=4)
    pi = target_policy(5, 2, seed=2)
    wrong = np.full((5, 2), 0.5)
    truth = true_value(theta, pi)
    assert estimate("dm", log, pi, wrong) == pytest.approx(0.5)
    assert estimate("dr", log, pi, wrong) == pytest.approx(truth, rel=0.03)


def test_dm_is_exact_with_the_true_model() -> None:
    log, theta = simulated(1000, seed=5)
    pi = target_policy(5, 2)
    per_position = (pi * theta).sum(axis=0)
    assert estimate("dm", log, pi, theta) == pytest.approx(per_position[log.position].mean())


def test_switch_extremes() -> None:
    log, _ = simulated(5000, seed=6)
    pi = target_policy(5, 2)
    q = item_position_model(log)
    assert estimate("switch-dr", log, pi, q, tau=np.inf) == pytest.approx(
        estimate("dr", log, pi, q)
    )
    assert estimate("switch-dr", log, pi, q, tau=0.0) == pytest.approx(estimate("dm", log, pi, q))


def test_contextual_shapes_agree_with_context_free() -> None:
    log, _ = simulated(3000, seed=7)
    pi = target_policy(5, 2)
    q = item_position_model(log)
    pi3 = np.broadcast_to(pi, (3000, 5, 2))
    q3 = np.broadcast_to(q, (3000, 5, 2))
    for name in ESTIMATORS:
        a = estimate(name, log, pi, q)
        b = estimate(name, log, pi3, q3)
        assert a == pytest.approx(b, rel=1e-12)
    qa, fa = model_terms(log, pi, q)
    qb, fb = model_terms(log, pi3, q)
    assert np.allclose(qa, qb) and np.allclose(fa, fb)


def test_unnormalized_policy_is_normalized_in_dm() -> None:
    log, _ = simulated(100, seed=8)
    pi = target_policy(5, 2)
    q = item_position_model(log)
    assert estimate("dm", log, 3 * pi, q) == pytest.approx(estimate("dm", log, pi, q))


def test_bootstrap_interval_covers_and_shrinks() -> None:
    log, theta = simulated(100_000, seed=9)
    pi = target_policy(5, 2)
    w = importance_weights(log, pi)
    t = round_terms("ips", log.reward, w)
    ci, reps = bootstrap(t, n_boot=200, seed=0)
    assert ci.low <= ci.estimate <= ci.high
    assert ci.low <= true_value(theta, pi) <= ci.high
    assert reps.size == 200 and ci.std > 0
    small = round_terms("ips", log.reward[:10_000], w[:10_000])
    ci_small, _ = bootstrap(small, n_boot=200, seed=0)
    assert (ci_small.high - ci_small.low) > (ci.high - ci.low)
    assert set(ci.to_dict()) == {"estimate", "low", "high", "std"}


def test_bootstrap_matches_analytic_standard_error() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(1.0, 2.0, 50_000)
    t = RoundTerms(x, np.zeros_like(x), np.ones_like(x))
    ci, _ = bootstrap(t, n_boot=400, seed=1)
    assert ci.std == pytest.approx(2.0 / np.sqrt(50_000), rel=0.1)


def test_bootstrap_in_column_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    import ad_auction_lab.ope as ope

    rng = np.random.default_rng(3)
    x = rng.normal(1.0, 2.0, 20_000)
    t = RoundTerms(x, np.zeros_like(x), np.ones_like(x))
    whole, reps_whole = bootstrap(t, n_boot=300, seed=4)
    monkeypatch.setattr(ope, "MAX_COUNTS", 3_000)
    blocked, reps_blocked = bootstrap(t, n_boot=300, seed=4)
    assert blocked.estimate == whole.estimate
    assert not np.array_equal(reps_whole, reps_blocked)
    assert blocked.std == pytest.approx(2.0 / np.sqrt(20_000), rel=0.15)
    # Without column blocks the replicate chunk size does not change the draws (only BLAS rounding).
    small = RoundTerms(x[:500], np.zeros(500), np.ones(500))
    a = bootstrap(small, n_boot=20, seed=5, chunk=1)[1]
    b = bootstrap(small, n_boot=20, seed=5, chunk=6)[1]
    np.testing.assert_allclose(a, b, rtol=1e-12)


def test_bootstrap_of_ratio_estimators() -> None:
    log, _ = simulated(20_000, seed=10)
    pi = target_policy(5, 2)
    w = importance_weights(log, pi)
    t = round_terms("snips", log.reward, w)
    _, reps = bootstrap(t, n_boot=50, seed=2, chunk=3)
    assert reps.min() >= 0 and reps.max() <= 1
    assert t.value(np.ones(len(log))) == pytest.approx(t.value())


def test_bootstrap_validation() -> None:
    t = RoundTerms(np.ones(3), np.zeros(3), np.ones(3))
    with pytest.raises(ValueError):
        bootstrap(t, n_boot=1)
    with pytest.raises(ValueError):
        bootstrap(t, alpha=1.5)


def test_round_terms_validation() -> None:
    r = np.ones(3)
    with pytest.raises(ValueError, match="unknown"):
        round_terms("magic", r, r)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reward model"):
        round_terms("dr", r, r)
    with pytest.raises(ValueError):
        round_terms("ips", r, r, clip=0)
    zero = RoundTerms(np.zeros(2), np.zeros(2), np.zeros(2))
    assert zero.value() == 0.0
    assert zero.value(np.ones(2)) == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reward": [], "action": [], "pscore": []},
        {"reward": [1.0], "action": [0, 1], "pscore": [0.5]},
        {"reward": [np.nan], "action": [0], "pscore": [0.5]},
        {"reward": [1.0], "action": [0], "pscore": [0.0]},
        {"reward": [1.0], "action": [0], "pscore": [1.5]},
        {"reward": [1.0], "action": [-1], "pscore": [0.5]},
        {"reward": [1.0], "action": [3], "pscore": [0.5], "n_actions": 2},
    ],
)
def test_log_validation(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        BanditLog.from_arrays(**kwargs)  # type: ignore[arg-type]


def test_policy_shape_validation() -> None:
    log = BanditLog.from_arrays([1, 0], [0, 1], [0.5, 0.5])
    with pytest.raises(ValueError, match="action_dist"):
        importance_weights(log, np.ones((3, 1)))
    with pytest.raises(ValueError):
        importance_weights(log, -np.ones((2, 1)))
    with pytest.raises(ValueError, match="sums to zero"):
        model_terms(log, np.zeros((2, 1)), np.ones((2, 1)))
    with pytest.raises(ValueError, match="sums to zero"):
        model_terms(log, np.zeros((2, 2, 1)), np.ones((2, 1)))


def test_action_dist_helpers() -> None:
    log = BanditLog.from_arrays(
        [0, 1, 0, 1], [0, 0, 1, 2], [0.6, 0.4, 0.3, 0.2], [0, 0, 0, 1], n_actions=3
    )
    m = marginal_action_dist(log)
    assert m.shape == (3, 2)
    assert np.allclose(m.sum(axis=0), 1)
    assert m[0, 0] == pytest.approx(0.5 / 0.8)
    e = empirical_action_dist(log)
    assert e[:, 0].tolist() == pytest.approx([2 / 3, 1 / 3, 0])
    assert e[:, 1].tolist() == pytest.approx([0, 0, 1])
    u = uniform_action_dist(4, 3)
    assert np.allclose(u, 0.25)
    with pytest.raises(ValueError):
        uniform_action_dist(0, 1)


def test_item_position_model_shrinks_to_position_rate() -> None:
    log = BanditLog.from_arrays([1, 1, 0, 0], [0, 0, 1, 1], [0.5] * 4, [0, 0, 0, 0], 3, 1)
    q = item_position_model(log, prior=2.0)
    assert q[2, 0] == pytest.approx(0.5)  # never shown: the position rate
    assert q[0, 0] == pytest.approx((2 + 1) / 4)
    sub = item_position_model(log, rows=np.array([0, 1]), prior=0.0)
    assert sub[0, 0] == 1.0


def test_relative_error() -> None:
    assert relative_error(1.1, 1.0) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        relative_error(1.0, 0.0)


@given(
    st.integers(1, 200),
    st.integers(0, 10_000),
    st.floats(0.05, 1.0),
)
@settings(max_examples=100, deadline=None)
def test_snips_is_bounded_and_scale_invariant(n: int, seed: int, scale: float) -> None:
    rng = np.random.default_rng(seed)
    A = 4
    log = BanditLog.from_arrays(
        rng.random(n), rng.integers(0, A, n), rng.uniform(0.05, 1, n), n_actions=A
    )
    pi = rng.dirichlet(np.ones(A))[:, None]
    v = estimate("snips", log, pi)
    assert log.reward.min() - 1e-12 <= v <= log.reward.max() + 1e-12
    # Scaling every propensity leaves SNIPS unchanged and scales IPS.
    log2 = BanditLog.from_arrays(log.reward, log.action, log.pscore * scale, n_actions=A)
    assert estimate("snips", log2, pi) == pytest.approx(v)
    assert estimate("ips", log2, pi) == pytest.approx(estimate("ips", log, pi) / scale)


@given(st.integers(0, 10_000))
@settings(max_examples=50, deadline=None)
def test_estimates_are_linear_in_reward_shift(seed: int) -> None:
    log, _ = simulated(300, seed=seed)
    pi = target_policy(5, 2, seed=seed)
    q = item_position_model(log)
    shifted = BanditLog.from_arrays(log.reward + 1.0, log.action, log.pscore, log.position, 5, 2)
    # Shifting rewards and the model by 1 shifts self-normalized and model-based estimates by 1.
    for name in ("snips", "dm", "sndr"):
        a = estimate(name, log, pi, q)  # type: ignore[arg-type]
        b = estimate(name, shifted, pi, q + 1.0)  # type: ignore[arg-type]
        assert b == pytest.approx(a + 1.0)
