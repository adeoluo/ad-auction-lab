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
