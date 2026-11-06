"""Open Bandit Dataset loading, reward features and cross-fitted reward models on a fixture."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ad_auction_lab.ctr import FactorizationMachine, FTRLProximal
from ad_auction_lab.obd import cross_fit_model_terms, load_obd, reward_table
from ad_auction_lab.ope import estimate, marginal_action_dist, model_terms, round_terms

A, N = 6, 4000


def write_fixture(d: Path, seed: int = 0) -> None:
    """A small dataset in the format of ``scripts/prepare_obd.py``."""

    rng = np.random.default_rng(seed)
    user = rng.integers(0, 2**32, (N, 4), dtype=np.uint64).astype(np.uint32)
    user[:, 0] = rng.integers(0, 3, N)  # a feature with a real effect
    affinity = rng.integers(0, 5, (N, A)).astype(np.uint8)
    item = rng.integers(0, A, N).astype(np.uint8)
    logit = -2.0 + 0.6 * (user[:, 0] == item % 3) + 0.3 * affinity[np.arange(N), item]
    click = (rng.random(N) < 1 / (1 + np.exp(-logit))).astype(np.uint8)
    np.savez(
        d / "random_all.npz",
        item=item,
        position=rng.integers(0, 3, N).astype(np.uint8),
        click=click,
        pscore=np.full(N, 1 / A),
        user=user,
        affinity=affinity,
    )
    p = rng.dirichlet(np.ones(A), N).max(axis=1)
    np.savez(
        d / "bts_all.npz",
        item=rng.integers(0, A, N).astype(np.uint8),
        position=rng.integers(0, 3, N).astype(np.uint8),
        click=(rng.random(N) < 0.1).astype(np.uint8),
        pscore=p,
    )
    np.savez(
        d / "item_all.npz",
        item_id=np.arange(A),
        feature_0=rng.normal(size=A),
        features=rng.integers(0, 2**31, (A, 3)),
    )


@pytest.fixture
def obd_dir(tmp_path: Path) -> Path:
    write_fixture(tmp_path)
    return tmp_path


def test_load_random_and_bts(obd_dir: Path) -> None:
    rnd = load_obd(obd_dir, "random", "all")
    bts = load_obd(obd_dir, "bts", "all")
    assert len(rnd.log) == N and rnd.has_context
    assert rnd.log.n_actions == A and rnd.log.n_positions == 3
    assert not bts.has_context
    assert rnd.item_features.shape == (A, 3)
    assert rnd.item_price_bucket.max() == A - 1
    with pytest.raises(ValueError):
        load_obd(obd_dir, "epsilon", "all")
    with pytest.raises(ValueError, match="no user context"):
        reward_table(bts, np.arange(5))


def test_reward_table_substitutes_actions(obd_dir: Path) -> None:
    rnd = load_obd(obd_dir, "random", "all")
    rows = np.arange(10)
    logged = reward_table(rnd, rows)
    assert logged.small.shape == (10, 4) and logged.wide.shape == (10, 11)
    assert logged.small[:, 1].tolist() == rnd.log.action[rows].tolist()
    t = reward_table(rnd, rows, 2)
    assert (t.small[:, 1] == 2).all()
    assert t.small[:, 2].tolist() == rnd.affinity[rows, 2].tolist()  # type: ignore[index]
    assert (t.wide[:, :4] == logged.wide[:, :4]).all()
    assert not (t.wide[:, 7:] == logged.wide[:, 7:]).all()
    assert t.labels.tolist() == logged.labels.tolist()


@pytest.mark.parametrize(
    ("make", "slack"),
    [(lambda: FTRLProximal(bits=12), 1.0), (lambda: FactorizationMachine(k=4, bits=12), 1.1)],
)
def test_cross_fit_terms_are_held_out_and_sensible(obd_dir: Path, make, slack: float) -> None:  # type: ignore[no-untyped-def]
    rnd = load_obd(obd_dir, "random", "all")
    pi = marginal_action_dist(load_obd(obd_dir, "bts", "all").log)
    q_pi, q_f = cross_fit_model_terms(rnd, pi, make, n_folds=2, epochs=2)
    assert q_pi.shape == (N,) and q_f.shape == (N,)
    assert np.all((q_pi > 0) & (q_pi < 1)) and np.all((q_f > 0) & (q_f < 1))
    # The linear model learns the planted effects; the FM (more parameters, 2,000 rows per fold)
    # stays within 10 percent of the constant predictor's squared error.
    y = rnd.log.reward
    base = y.mean()
    assert np.mean((y - q_f) ** 2) < slack * np.mean((y - base) ** 2)
    iw = rnd.log.pscore**-1 * pi[rnd.log.action, rnd.log.position]
    for name in ("dm", "dr", "sndr", "switch-dr"):
        v = round_terms(name, y, iw, q_pi, q_f, tau=10).value()  # type: ignore[arg-type]
        assert 0 < v < 1


def test_cross_fit_matches_constant_model(obd_dir: Path) -> None:
    """With a model that predicts a constant, the DM term is that constant."""

    class Constant:
        def fit(self, table: object, *, epochs: int = 1, seed: int = 0) -> None:
            return None

        def predict_proba(self, table: object) -> np.ndarray:
            return np.full(len(table), 0.25, dtype=np.float32)  # type: ignore[arg-type]

    rnd = load_obd(obd_dir, "random", "all")
    pi = np.full((A, 3), 1 / A)
    q_pi, q_f = cross_fit_model_terms(rnd, pi, Constant, n_folds=3)
    assert np.allclose(q_pi, 0.25) and np.allclose(q_f, 0.25)
    qp2, qf2 = model_terms(rnd.log, pi, np.full((A, 3), 0.25))
    assert np.allclose(q_pi, qp2) and np.allclose(q_f, qf2)
    assert estimate("dm", rnd.log, pi, np.full((A, 3), 0.25)) == pytest.approx(0.25)


def test_cross_fit_validation(obd_dir: Path) -> None:
    rnd = load_obd(obd_dir, "random", "all")
    with pytest.raises(ValueError):
        cross_fit_model_terms(rnd, np.ones((A, 3)), FTRLProximal, n_folds=1)
    with pytest.raises(ValueError):
        cross_fit_model_terms(rnd, np.ones((A, 2)), FTRLProximal)


def test_skips_actions_with_zero_probability(obd_dir: Path) -> None:
    rnd = load_obd(obd_dir, "random", "all")
    pi = np.zeros((A, 3))
    pi[0] = 1.0
    q_pi, _ = cross_fit_model_terms(rnd, pi, lambda: FTRLProximal(bits=10), n_folds=2)
    assert np.all(q_pi > 0)
