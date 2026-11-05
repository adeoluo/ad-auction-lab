"""Opt-in comparisons with reference libraries: ``make validation``."""

from __future__ import annotations

import numpy as np
import pytest

from ad_auction_lab.calibration import IsotonicCalibration
from ad_auction_lab.ctr import FTRLProximal
from ad_auction_lab.data import Table, split_rows
from ad_auction_lab.hashing import hash_rows
from ad_auction_lab.metrics import log_loss, roc_auc

pytestmark = pytest.mark.validation
sk = pytest.importorskip("sklearn.metrics")


def test_metrics_match_sklearn() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    p = np.round(rng.random(5000), 2)
    assert roc_auc(y, p) == pytest.approx(sk.roc_auc_score(y, p), abs=1e-12)
    assert log_loss(y, p) == pytest.approx(sk.log_loss(y, np.clip(p, 1e-15, 1 - 1e-15)), abs=1e-12)


def test_isotonic_matches_sklearn() -> None:
    from sklearn.isotonic import IsotonicRegression

    rng = np.random.default_rng(1)
    p = np.round(rng.random(3000), 3)
    y = (rng.random(3000) < p**2).astype(float)
    ours = IsotonicCalibration().fit(p, y)
    ref = IsotonicRegression(out_of_bounds="clip").fit(p, y)
    grid = np.linspace(-0.1, 1.1, 500)
    assert np.allclose(ours.transform(grid), ref.predict(grid), atol=1e-9)


def test_ftrl_close_to_sklearn_logistic_regression(clicks: Table) -> None:
    """Several FTRL passes approach the batch L2-regularized optimum on the same hashed features."""

    from scipy.sparse import csr_matrix
    from sklearn.linear_model import LogisticRegression

    tr, te = split_rows(len(clicks), (0.7, 0.3), seed=0)
    bits = 12
    idx = hash_rows(clicks.small, clicks.wide, bits)
    n, f = idx.shape
    X = csr_matrix(
        (np.ones(n * f), idx.reshape(-1), np.arange(0, n * f + 1, f)), shape=(n, 2**bits)
    )
    X.sum_duplicates()
    ref = LogisticRegression(C=1.0, max_iter=2000).fit(X[tr], clicks.labels[tr])
    p_ref = ref.predict_proba(X[te])[:, 1]
    m = FTRLProximal(alpha=0.05, l1=0.0, l2=1.0, bits=bits)
    m.fit(clicks, tr, epochs=10, seed=0)
    p = m.predict_proba(clicks, te)
    y = clicks.labels[te]
    assert abs(log_loss(y, p) - log_loss(y, p_ref)) < 0.01
    assert abs(roc_auc(y, p) - roc_auc(y, p_ref)) < 0.01


def _obp_inputs(seed: int, n: int = 3000, A: int = 7, K: int = 3):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    action = rng.integers(0, A, n)
    position = rng.integers(0, K, n)
    pscore = rng.uniform(0.02, 1.0, n)
    reward = rng.binomial(1, 0.2, n).astype(float)
    pi = rng.dirichlet(np.ones(A), (n, K)).transpose(0, 2, 1)
    q = rng.uniform(0, 1, (n, A, K))
    return action, position, pscore, reward, pi, q


@pytest.mark.parametrize("seed", range(5))
def test_ope_estimators_match_obp(seed: int) -> None:
    obp = pytest.importorskip("obp.ope")
    from ad_auction_lab.ope import BanditLog, estimate

    action, position, pscore, reward, pi, q = _obp_inputs(seed)
    log = BanditLog.from_arrays(reward, action, pscore, position, 7, 3)
    common = {
        "reward": reward,
        "action": action,
        "position": position,
        "pscore": pscore,
        "action_dist": pi,
    }
    tau = float(np.quantile(pi[np.arange(len(action)), action, position] / pscore, 0.9))
    refs = {
        "ips": (obp.InverseProbabilityWeighting(), {}),
        "snips": (obp.SelfNormalizedInverseProbabilityWeighting(), {}),
        "dm": (obp.DirectMethod(), {"estimated_rewards_by_reg_model": q}),
        "dr": (obp.DoublyRobust(), {"estimated_rewards_by_reg_model": q}),
        "sndr": (obp.SelfNormalizedDoublyRobust(), {"estimated_rewards_by_reg_model": q}),
        "switch-dr": (obp.SwitchDoublyRobust(lambda_=tau), {"estimated_rewards_by_reg_model": q}),
    }
    for name, (est, extra) in refs.items():
        kw = {**common, **extra}
        if name == "dm":
            kw.pop("reward")
            kw.pop("pscore")
        expected = est.estimate_policy_value(**kw)
        got = estimate(name, log, pi, q, tau=tau)  # type: ignore[arg-type]
        assert got == pytest.approx(expected, rel=1e-12, abs=1e-15), name
    clipped = obp.InverseProbabilityWeighting(lambda_=2.0).estimate_policy_value(**common)
    assert estimate("ips", log, pi, clip=2.0) == pytest.approx(clipped, rel=1e-12)
    dr_clip = obp.DoublyRobust(lambda_=2.0).estimate_policy_value(
        **common, estimated_rewards_by_reg_model=q
    )
    assert estimate("dr", log, pi, q, clip=2.0) == pytest.approx(dr_clip, rel=1e-12)


def test_ope_on_open_bandit_dataset_sample_matches_obp() -> None:
    """Same numbers as obp on the first 50,000 random-policy rounds of the real data, if present."""

    obp = pytest.importorskip("obp.ope")
    from pathlib import Path

    from ad_auction_lab.obd import load_obd
    from ad_auction_lab.ope import estimate, item_position_model, marginal_action_dist

    d = Path(__file__).resolve().parents[2] / ".bench-data" / "ad-auction-lab" / "obd"
    if not (d / "random_men.npz").exists():
        pytest.skip("Open Bandit Dataset not prepared")
    rnd = load_obd(d, "random", "men")
    log = rnd.log.take(np.arange(50_000))
    pi = marginal_action_dist(load_obd(d, "bts", "men").log)
    q = item_position_model(log)
    n = len(log)
    pi3 = np.broadcast_to(pi, (n, *pi.shape))
    q3 = np.broadcast_to(q, (n, *q.shape))
    kw = {
        "reward": log.reward,
        "action": log.action,
        "position": log.position,
        "pscore": log.pscore,
        "action_dist": pi3,
    }
    assert estimate("ips", log, pi) == pytest.approx(
        obp.InverseProbabilityWeighting().estimate_policy_value(**kw), rel=1e-12
    )
    assert estimate("dr", log, pi, q) == pytest.approx(
        obp.DoublyRobust().estimate_policy_value(**kw, estimated_rewards_by_reg_model=q3),
        rel=1e-12,
    )
