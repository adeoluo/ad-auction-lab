"""Hashed click-through-rate models: FTRL-Proximal logistic regression and a factorization machine.

Both read a :class:`~ad_auction_lab.data.Table` and hash every ``(field, value)`` token into
``2^bits`` buckets on the fly, so the 11.46 million Criteo rows are never expanded into a sparse
matrix. Training is one sequential pass per epoch in the order given (the Rust core, or the numpy
reference with ``backend="python"``); ``fit`` shuffles the training rows each epoch with a seeded
permutation and can stop early on a validation log loss.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ad_auction_lab.backend import extension, resolve_backend
from ad_auction_lab.ctr_reference import FmReference, FtrlReference
from ad_auction_lab.data import Table
from ad_auction_lab.metrics import log_loss


def _rows(table: Table, rows: NDArray[np.integer] | None) -> NDArray[np.uint64]:
    if rows is None:
        return np.arange(len(table), dtype=np.uint64)
    r = np.ascontiguousarray(rows, dtype=np.uint64)
    if r.size and int(np.max(r)) >= len(table):
        raise ValueError("row index out of range")
    return r


def _blocks(table: Table) -> tuple[Any, int, Any, int]:
    small = table.small if table.small.flags.c_contiguous else np.ascontiguousarray(table.small)
    wide = table.wide if table.wide.flags.c_contiguous else np.ascontiguousarray(table.wide)
    if small.shape[1] == 0:
        small = np.zeros(0, dtype=np.uint8)
    if wide.shape[1] == 0:
        wide = np.zeros(0, dtype=np.uint32)
    return small, table.small.shape[1], wide, table.wide.shape[1]


@dataclass
class EpochLog:
    epoch: int
    train_progressive_logloss: float
    valid_logloss: float | None
    seconds: float


@dataclass
class History:
    epochs: list[EpochLog] = field(default_factory=list)
    best_epoch: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"best_epoch": self.best_epoch, "epochs": [asdict(e) for e in self.epochs]}


class _HashedModel:
    """Shared training loop (epochs, shuffling, early stopping) for the two learners."""

    kind = "base"

    def __init__(self, bits: int, backend: str) -> None:
        if not 1 <= bits <= 28:
            raise ValueError("bits must be between 1 and 28")
        self.bits = bits
        self.backend = resolve_backend(backend)

    def partial_fit(self, table: Table, rows: NDArray[np.integer] | None = None) -> float:
        """One sequential pass over ``rows`` (in order); returns the progressive log loss."""

        r = _rows(table, rows)
        if self.backend == "rust":
            small, ns, wide, nw = _blocks(table)
            return float(self._rs.fit(small, ns, wide, nw, _labels(table), r))
        return float(self._ref.fit(table, r))

    def predict_proba(
        self, table: Table, rows: NDArray[np.integer] | None = None
    ) -> NDArray[np.float32]:
        """Click probabilities for ``rows`` (all rows by default)."""

        r = _rows(table, rows)
        if self.backend == "rust":
            small, ns, wide, nw = _blocks(table)
            return np.frombuffer(self._rs.predict(small, ns, wide, nw, r), dtype=np.float32)
        out: NDArray[np.float32] = self._ref.predict(table, r)
        return out

    def fit(
        self,
        table: Table,
        rows: NDArray[np.integer] | None = None,
        *,
        epochs: int = 1,
        valid_rows: NDArray[np.integer] | None = None,
        patience: int = 0,
        seed: int = 0,
        shuffle: bool = True,
        verbose: bool = False,
    ) -> History:
        """Train for up to ``epochs`` passes.

        With ``valid_rows`` the validation log loss is computed after each epoch; with
        ``patience > 0`` training stops after that many epochs without improvement and the best
        state is restored.
        """

        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        r = _rows(table, rows)
        rng = np.random.default_rng(seed)
        hist = History()
        best, best_state, bad = float("inf"), None, 0
        for epoch in range(1, epochs + 1):
            order = r[rng.permutation(len(r))] if shuffle else r
            t0 = time.perf_counter()
            train_loss = self.partial_fit(table, order)
            seconds = time.perf_counter() - t0
            valid = None
            if valid_rows is not None:
                vr = _rows(table, valid_rows)
                valid = log_loss(table.labels[vr], self.predict_proba(table, vr))
            hist.epochs.append(EpochLog(epoch, train_loss, valid, seconds))
            if verbose:
                v = "" if valid is None else f"  valid logloss {valid:.5f}"
                msg = f"epoch {epoch}: train progressive logloss {train_loss:.5f}{v}"
                print(f"{msg}  ({seconds:.1f} s)")
            if valid is None:
                hist.best_epoch = epoch
                continue
            if valid < best:
                best, hist.best_epoch, bad = valid, epoch, 0
                if patience and epoch < epochs:
                    best_state = self._get_state()
            else:
                bad += 1
                if patience and bad >= patience:
                    break
        if best_state is not None and hist.best_epoch != len(hist.epochs):
            self._set_state(best_state)
        return hist

    # state handling is model specific
    _rs: Any
    _ref: Any

    def _get_state(self) -> dict[str, Any]:  # pragma: no cover - overridden
        raise NotImplementedError

    def _set_state(self, state: dict[str, Any]) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def params(self) -> dict[str, Any]:  # pragma: no cover - overridden
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        """Save parameters and learner state to an ``.npz`` file."""

        state = self._get_state()
        np.savez(Path(path), kind=self.kind, params=repr(self.params()), **state)


def _labels(table: Table) -> NDArray[np.uint8]:
    y = table.labels
    return y if y.flags.c_contiguous else np.ascontiguousarray(y)


class FTRLProximal(_HashedModel):
    """Logistic regression on hashed features trained with FTRL-Proximal.

    Defaults (``alpha=0.05, beta=1, l1=1, l2=0, bits=22``) are the ones used for Criteo in the
    benchmarks; ``alpha`` and ``l1`` were chosen on the validation split.
    """

    kind = "ftrl"

    def __init__(
        self,
        alpha: float = 0.05,
        beta: float = 1.0,
        l1: float = 1.0,
        l2: float = 0.0,
        bits: int = 22,
        backend: str = "auto",
    ) -> None:
        super().__init__(bits, backend)
        if alpha <= 0 or beta < 0 or l1 < 0 or l2 < 0:
            raise ValueError("alpha must be positive and beta, l1, l2 non-negative")
        self.alpha, self.beta, self.l1, self.l2 = alpha, beta, l1, l2
        if self.backend == "rust":
            self._rs = extension().FtrlModel(alpha, beta, l1, l2, bits)
        else:
            self._ref = FtrlReference(alpha, beta, l1, l2, bits)

    def params(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "l1": self.l1,
            "l2": self.l2,
            "bits": self.bits,
        }

    def weights(self) -> NDArray[np.float32]:
        """Weights of the ``2^bits`` buckets followed by the bias."""

        if self.backend == "rust":
            return np.frombuffer(self._rs.weights(), dtype=np.float32)
        out: NDArray[np.float32] = self._ref.weights()
        return out

    def _get_state(self) -> dict[str, Any]:
        if self.backend == "rust":
            z = np.frombuffer(self._rs.get_z(), dtype=np.float32)
            n = np.frombuffer(self._rs.get_n(), dtype=np.float32)
        else:
            z, n = self._ref.z.copy(), self._ref.n.copy()
        return {"z": z, "n": n}

    def _set_state(self, state: dict[str, Any]) -> None:
        z = np.ascontiguousarray(state["z"], dtype=np.float32)
        n = np.ascontiguousarray(state["n"], dtype=np.float32)
        if self.backend == "rust":
            self._rs.set_state(z, n)
        else:
            if z.shape != self._ref.z.shape or n.shape != self._ref.n.shape:
                raise ValueError("state does not match the model size")
            self._ref.z, self._ref.n = z.copy(), n.copy()

    @classmethod
    def load(cls, path: str | Path, backend: str = "auto") -> FTRLProximal:
        with np.load(Path(path), allow_pickle=False) as f:
            if str(f["kind"]) != cls.kind:
                raise ValueError(f"{path} is not an FTRL model")
            params = _parse_params(str(f["params"]))
            m = cls(**params, backend=backend)
            m._set_state({"z": f["z"], "n": f["n"]})
        return m


class FactorizationMachine(_HashedModel):
    """Second-order factorization machine on hashed features, trained with AdaGrad."""

    kind = "fm"

    def __init__(
        self,
        k: int = 16,
        bits: int = 22,
        lr: float = 0.05,
        l2_linear: float = 0.0,
        l2_factor: float = 1e-5,
        init_scale: float = 0.01,
        seed: int = 0,
        backend: str = "auto",
    ) -> None:
        super().__init__(bits, backend)
        if not 1 <= k <= 256:
            raise ValueError("k must be between 1 and 256")
        if lr <= 0 or l2_linear < 0 or l2_factor < 0 or init_scale < 0:
            raise ValueError("lr must be positive and l2 and init_scale non-negative")
        self.k, self.lr, self.l2_linear, self.l2_factor = k, lr, l2_linear, l2_factor
        self.init_scale, self.seed = init_scale, seed
        if self.backend == "rust":
            self._rs = extension().FmModel(k, bits, lr, l2_linear, l2_factor, init_scale, seed)
        else:
            self._ref = FmReference(k, bits, lr, l2_linear, l2_factor, init_scale, seed)

    def params(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "bits": self.bits,
            "lr": self.lr,
            "l2_linear": self.l2_linear,
            "l2_factor": self.l2_factor,
            "init_scale": self.init_scale,
            "seed": self.seed,
        }

    def _get_state(self) -> dict[str, Any]:
        if self.backend == "rust":
            s = self._rs.state()
            return {
                "w0": np.float64(s["w0"]),
                "g0": np.float64(s["g0"]),
                **{k: np.frombuffer(s[k], dtype=np.float32) for k in ("w", "gw", "v", "gv")},
            }
        r = self._ref
        return {
            "w0": np.float64(r.w0),
            "g0": np.float64(r.g0),
            "w": r.w.copy(),
            "gw": r.gw.copy(),
            "v": r.v.reshape(-1).copy(),
            "gv": r.gv.reshape(-1).copy(),
        }

    def _set_state(self, state: dict[str, Any]) -> None:
        arrays = {
            k: np.ascontiguousarray(state[k], dtype=np.float32).reshape(-1)
            for k in ("w", "gw", "v", "gv")
        }
        w0, g0 = float(state["w0"]), float(state["g0"])
        if self.backend == "rust":
            self._rs.set_state(w0, g0, arrays["w"], arrays["gw"], arrays["v"], arrays["gv"])
            return
        r = self._ref
        size = 1 << self.bits
        if arrays["w"].size != size or arrays["v"].size != size * self.k:
            raise ValueError("state does not match the model size")
        r.w0, r.g0 = w0, g0
        r.w, r.gw = arrays["w"].copy(), arrays["gw"].copy()
        r.v = arrays["v"].reshape(size, self.k).copy()
        r.gv = arrays["gv"].reshape(size, self.k).copy()

    @classmethod
    def load(cls, path: str | Path, backend: str = "auto") -> FactorizationMachine:
        with np.load(Path(path), allow_pickle=False) as f:
            if str(f["kind"]) != cls.kind:
                raise ValueError(f"{path} is not a factorization machine")
            params = _parse_params(str(f["params"]))
            m = cls(**params, backend=backend)
            m._set_state({k: f[k] for k in ("w0", "g0", "w", "gw", "v", "gv")})
        return m


def _parse_params(text: str) -> dict[str, Any]:
    """Parse the ``repr`` of a flat dict of numbers without evaluating code."""

    import ast

    value = ast.literal_eval(text)
    if not isinstance(value, dict) or not all(
        isinstance(k, str) and isinstance(v, int | float) for k, v in value.items()
    ):
        raise ValueError("model parameters must be a flat dict of numbers")
    return value


def load_model(path: str | Path, backend: str = "auto") -> FTRLProximal | FactorizationMachine:
    """Load either model type from ``path``."""

    with np.load(Path(path), allow_pickle=False) as f:
        kind = str(f["kind"])
    if kind == FTRLProximal.kind:
        return FTRLProximal.load(path, backend)
    if kind == FactorizationMachine.kind:
        return FactorizationMachine.load(path, backend)
    raise ValueError(f"unknown model kind {kind!r}")
