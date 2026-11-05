from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ad_auction_lab.backend import rust_available
from ad_auction_lab.ctr import FactorizationMachine, FTRLProximal, _parse_params, load_model
from ad_auction_lab.data import Table, split_rows
from ad_auction_lab.metrics import log_loss, roc_auc

BACKENDS = ["python", *(["rust"] if rust_available() else [])]


@pytest.mark.parametrize("backend", BACKENDS)
def test_ftrl_learns(clicks: Table, backend: str) -> None:
    tr, te = split_rows(len(clicks), (0.7, 0.3), seed=0)
    m = FTRLProximal(alpha=0.1, l1=0.1, bits=14, backend=backend)
    hist = m.fit(clicks, tr, epochs=2, valid_rows=te, seed=1)
    assert len(hist.epochs) == 2
    p = m.predict_proba(clicks, te)
    base = np.full(len(te), clicks.labels[tr].mean())
    assert log_loss(clicks.labels[te], p) < log_loss(clicks.labels[te], base) - 0.02
    assert roc_auc(clicks.labels[te], p) > 0.7
    w = m.weights()
    assert w.shape == (2**14 + 1,)


@pytest.mark.parametrize("backend", BACKENDS)
def test_fm_learns_and_beats_constant(clicks: Table, backend: str) -> None:
    tr, te = split_rows(len(clicks), (0.7, 0.3), seed=0)
    m = FactorizationMachine(k=4, bits=12, lr=0.05, backend=backend)
    m.fit(clicks, tr, epochs=2, seed=0)
    p = m.predict_proba(clicks, te)
    assert roc_auc(clicks.labels[te], p) > 0.7


def test_l1_sparsity(clicks: Table) -> None:
    dense = FTRLProximal(alpha=0.1, l1=0.0, bits=12, backend="python")
    sparse = FTRLProximal(alpha=0.1, l1=5.0, bits=12, backend="python")
    rows = np.arange(2000)
    dense.partial_fit(clicks, rows)
    sparse.partial_fit(clicks, rows)
    assert (sparse.weights() != 0).sum() < (dense.weights() != 0).sum()


@pytest.mark.parametrize("backend", BACKENDS)
def test_early_stopping_restores_best(clicks: Table, backend: str) -> None:
    tr, va = split_rows(len(clicks), (0.5, 0.5), seed=0)
    # A large learning rate overfits after the first epoch.
    m = FactorizationMachine(k=8, bits=12, lr=0.5, l2_factor=0.0, init_scale=0.1, backend=backend)
    hist = m.fit(clicks, tr, epochs=4, valid_rows=va, patience=1, seed=0)
    losses = [e.valid_logloss for e in hist.epochs]
    assert all(v is not None for v in losses)
    best = min(v for v in losses if v is not None)
    assert hist.best_epoch == 1 + losses.index(best)
    assert log_loss(clicks.labels[va], m.predict_proba(clicks, va)) == pytest.approx(best, abs=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load(tmp_path: Path, clicks: Table, backend: str) -> None:
    rows = np.arange(1000)
    for m in (
        FTRLProximal(bits=10, backend=backend),
        FactorizationMachine(k=2, bits=10, backend=backend),
    ):
        m.fit(clicks, rows, verbose=True)
        path = tmp_path / f"{m.kind}.npz"
        m.save(path)
        back = load_model(path, backend=backend)
        assert type(back) is type(m)
        assert np.allclose(back.predict_proba(clicks, rows), m.predict_proba(clicks, rows))
    with pytest.raises(ValueError, match="not an FTRL"):
        FTRLProximal.load(tmp_path / "fm.npz")
    with pytest.raises(ValueError, match="not a factorization"):
        FactorizationMachine.load(tmp_path / "ftrl.npz")


def test_load_rejects_unknown_kind(tmp_path: Path) -> None:
    np.savez(tmp_path / "x.npz", kind="other", params="{}")
    with pytest.raises(ValueError, match="unknown model kind"):
        load_model(tmp_path / "x.npz")


def test_parse_params_refuses_code() -> None:
    assert _parse_params("{'a': 1, 'b': 0.5}") == {"a": 1, "b": 0.5}
    with pytest.raises(ValueError):
        _parse_params("{'a': [1]}")
    with pytest.raises(ValueError):
        _parse_params("__import__('os').getcwd()")


def test_invalid_arguments(clicks: Table) -> None:
    with pytest.raises(ValueError):
        FTRLProximal(alpha=0)
    with pytest.raises(ValueError):
        FTRLProximal(bits=0)
    with pytest.raises(ValueError):
        FactorizationMachine(k=0)
    with pytest.raises(ValueError):
        FactorizationMachine(lr=0)
    with pytest.raises(ValueError):
        FTRLProximal(backend="gpu")
    m = FTRLProximal(bits=8, backend="python")
    with pytest.raises(ValueError, match="out of range"):
        m.predict_proba(clicks, np.array([len(clicks)]))
    with pytest.raises(ValueError, match="epochs"):
        m.fit(clicks, epochs=0)


def test_state_size_checked() -> None:
    m = FTRLProximal(bits=8, backend="python")
    with pytest.raises(ValueError, match="size"):
        m._set_state({"z": np.zeros(3), "n": np.zeros(3)})
    f = FactorizationMachine(k=2, bits=8, backend="python")
    bad = {
        "w0": 0.0,
        "g0": 0.0,
        "w": np.zeros(3),
        "gw": np.zeros(3),
        "v": np.zeros(3),
        "gv": np.zeros(3),
    }
    with pytest.raises(ValueError, match="size"):
        f._set_state(bad)


def test_single_block_tables() -> None:
    rng = np.random.default_rng(0)
    wide_only = Table(
        np.zeros((200, 0), np.uint8),
        rng.integers(0, 5, (200, 2), dtype=np.uint32),
        rng.integers(0, 2, 200, dtype=np.uint8),
    )
    small_only = Table(
        rng.integers(0, 5, (200, 2), dtype=np.uint8),
        np.zeros((200, 0), np.uint32),
        wide_only.labels,
    )
    for backend in BACKENDS:
        for t in (wide_only, small_only):
            m = FTRLProximal(bits=8, backend=backend)
            m.fit(t)
            assert m.predict_proba(t).shape == (200,)
