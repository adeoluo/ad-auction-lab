"""The Rust core must reproduce the numpy reference implementations."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ad_auction_lab.backend import rust_available
from ad_auction_lab.ctr import FactorizationMachine, FTRLProximal
from ad_auction_lab.data import Table, synthetic_clicks

pytestmark = pytest.mark.skipif(not rust_available(), reason="Rust extension not built")


@given(st.integers(0, 1000), st.floats(0.01, 0.5), st.floats(0.0, 2.0), st.floats(0.0, 1.0))
@settings(max_examples=15, deadline=None)
def test_ftrl_parity(seed: int, alpha: float, l1: float, l2: float) -> None:
    t = synthetic_clicks(400, n_small=3, n_wide=4, seed=seed)
    rows = np.random.default_rng(seed).permutation(400)
    ms = [FTRLProximal(alpha=alpha, l1=l1, l2=l2, bits=10, backend=b) for b in ("python", "rust")]
    losses = [m.partial_fit(t, rows) for m in ms]
    assert losses[0] == pytest.approx(losses[1], rel=1e-9)
    assert np.allclose(ms[0].weights(), ms[1].weights(), atol=1e-6)
    assert np.allclose(ms[0].predict_proba(t), ms[1].predict_proba(t), atol=1e-6)


@given(st.integers(0, 1000), st.integers(1, 8), st.floats(0.01, 0.3))
@settings(max_examples=10, deadline=None)
def test_fm_parity(seed: int, k: int, lr: float) -> None:
    t = synthetic_clicks(300, n_small=2, n_wide=3, seed=seed)
    rows = np.random.default_rng(seed).permutation(300)
    ms = [
        FactorizationMachine(
            k=k, bits=8, lr=lr, l2_factor=1e-4, init_scale=0.05, seed=seed, backend=b
        )
        for b in ("python", "rust")
    ]
    losses = [m.partial_fit(t, rows) for m in ms]
    assert losses[0] == pytest.approx(losses[1], rel=1e-7)
    assert np.allclose(ms[0].predict_proba(t), ms[1].predict_proba(t), atol=1e-5)


def test_fm_initial_factors_match() -> None:
    ms = [FactorizationMachine(k=3, bits=6, seed=7, backend=b) for b in ("python", "rust")]
    a = ms[0]._get_state()["v"]
    b = ms[1]._get_state()["v"]
    assert np.array_equal(a, b)


def test_rust_rejects_bad_buffers() -> None:
    t = synthetic_clicks(10, seed=0)
    m = FTRLProximal(bits=8, backend="rust")
    with pytest.raises(ValueError):
        m._rs.fit(t.small, 3, t.wide, t.wide.shape[1], t.labels, np.arange(10, dtype=np.uint64))
    bad = Table(t.small, t.wide, t.labels)
    with pytest.raises(ValueError):
        m._rs.fit(
            bad.small,
            bad.small.shape[1],
            bad.wide,
            bad.wide.shape[1],
            bad.labels[:5].copy(),
            np.arange(10, dtype=np.uint64),
        )
