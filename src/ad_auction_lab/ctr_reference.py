"""Readable numpy reference implementations of the hashed CTR learners.

They follow ``rust/src/ftrl.rs`` and ``rust/src/fm.rs`` step by step (float32 storage, float64
arithmetic, the same update order), so the parity tests can require the same predictions. They
are slow (a Python loop per row) and meant for tests and small data.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from ad_auction_lab.data import Table
from ad_auction_lab.hashing import hash_rows, mix64


def sigmoid(s: float) -> float:
    s = min(max(s, -35.0), 35.0)
    return 1.0 / (1.0 + math.exp(-s))


def log_loss_one(p: float, y: float) -> float:
    p = min(max(p, 1e-15), 1 - 1e-15)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


class FtrlReference:
    """FTRL-Proximal logistic regression (McMahan et al., 2013), one row at a time."""

    def __init__(self, alpha: float, beta: float, l1: float, l2: float, bits: int) -> None:
        self.alpha, self.beta, self.l1, self.l2, self.bits = alpha, beta, l1, l2, bits
        size = (1 << bits) + 1
        self.z = np.zeros(size, dtype=np.float32)
        self.n = np.zeros(size, dtype=np.float32)

    def _weight(self, i: int, l1: float, l2: float) -> float:
        z = float(self.z[i])
        if abs(z) <= l1:
            return 0.0
        sign = 1.0 if z > 0 else -1.0
        return -(z - sign * l1) / ((self.beta + math.sqrt(float(self.n[i]))) / self.alpha + l2)

    def _update(self, i: int, wi: float, g: float, g2: float) -> None:
        n_old = float(self.n[i])
        sigma = (math.sqrt(n_old + g2) - math.sqrt(n_old)) / self.alpha
        self.z[i] = np.float32(float(self.z[i]) + g - sigma * wi)
        self.n[i] = np.float32(n_old + g2)

    def fit(self, table: Table, rows: NDArray[np.integer]) -> float:
        bias = 1 << self.bits
        idx = hash_rows(table.small[rows], table.wide[rows], self.bits)
        total = 0.0
        for k, r in enumerate(rows):
            wb = self._weight(bias, 0.0, 0.0)
            w = [self._weight(int(i), self.l1, self.l2) for i in idx[k]]
            p = sigmoid(wb + sum(w))
            y = float(table.labels[r])
            total += log_loss_one(p, y)
            g = p - y
            for wi, i in zip(w, idx[k], strict=True):
                self._update(int(i), wi, g, g * g)
            self._update(bias, wb, g, g * g)
        return total / len(rows) if len(rows) else 0.0

    def weights(self) -> NDArray[np.float32]:
        bias = 1 << self.bits
        w = np.array(
            [self._weight(i, self.l1, self.l2) for i in range(bias)] + [self._weight(bias, 0, 0)]
        )
        return w.astype(np.float32)

    def predict(self, table: Table, rows: NDArray[np.integer]) -> NDArray[np.float32]:
        w = self.weights()
        idx = hash_rows(table.small[rows], table.wide[rows], self.bits)
        s = float(w[-1]) + w[idx].astype(np.float64).sum(axis=1)
        return (1 / (1 + np.exp(-np.clip(s, -35, 35)))).astype(np.float32)


def fm_init(seed: int, size: int, scale: float) -> NDArray[np.float32]:
    """Initial factors: uniform in [-scale, scale] from a SplitMix64 stream (as in Rust)."""

    j = np.arange(size, dtype=np.uint64)
    with np.errstate(over="ignore"):
        x = np.uint64(seed) ^ (j * np.uint64(0x9E3779B97F4A7C15))
    u = (mix64(x) >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))
    return ((2.0 * u - 1.0) * scale).astype(np.float32)


class FmReference:
    """Second-order factorization machine with AdaGrad, one row at a time."""

    EPS = 1e-8

    def __init__(
        self,
        k: int,
        bits: int,
        lr: float,
        l2_linear: float,
        l2_factor: float,
        init_scale: float,
        seed: int,
    ) -> None:
        self.k, self.bits, self.lr = k, bits, lr
        self.l2_linear, self.l2_factor = l2_linear, l2_factor
        size = 1 << bits
        self.w0 = 0.0
        self.g0 = 0.0
        self.w = np.zeros(size, dtype=np.float32)
        self.gw = np.zeros(size, dtype=np.float32)
        self.v = fm_init(seed, size * k, init_scale).reshape(size, k)
        self.gv = np.zeros((size, k), dtype=np.float32)

    def _score(self, idx: NDArray[np.int64]) -> tuple[float, NDArray[np.float64]]:
        vv = self.v[idx].astype(np.float64)
        s = vv.sum(axis=0)
        inter = float((s * s).sum() - (vv * vv).sum())
        return self.w0 + float(self.w[idx].astype(np.float64).sum()) + 0.5 * inter, s

    def fit(self, table: Table, rows: NDArray[np.integer]) -> float:
        idx = hash_rows(table.small[rows], table.wide[rows], self.bits)
        total = 0.0
        for k, r in enumerate(rows):
            score, s = self._score(idx[k])
            p = sigmoid(score)
            y = float(table.labels[r])
            total += log_loss_one(p, y)
            g = p - y
            self.g0 += g * g
            self.w0 -= self.lr * g / math.sqrt(self.g0 + self.EPS)
            for i in idx[k]:
                wi = float(self.w[i])
                gi = g + self.l2_linear * wi
                acc = float(self.gw[i]) + gi * gi
                self.gw[i] = np.float32(acc)
                self.w[i] = np.float32(wi - self.lr * gi / math.sqrt(acc + self.EPS))
                vif = self.v[i].astype(np.float64)
                gf = g * (s - vif) + self.l2_factor * vif
                acc_v = self.gv[i].astype(np.float64) + gf * gf
                self.gv[i] = acc_v.astype(np.float32)
                self.v[i] = (vif - self.lr * gf / np.sqrt(acc_v + self.EPS)).astype(np.float32)
        return total / len(rows) if len(rows) else 0.0

    def predict(self, table: Table, rows: NDArray[np.integer]) -> NDArray[np.float32]:
        idx = hash_rows(table.small[rows], table.wide[rows], self.bits)
        vv = self.v[idx].astype(np.float64)
        s = vv.sum(axis=1)
        inter = (s * s).sum(axis=1) - (vv * vv).sum(axis=(1, 2))
        score = self.w0 + self.w[idx].astype(np.float64).sum(axis=1) + 0.5 * inter
        return (1 / (1 + np.exp(-np.clip(score, -35, 35)))).astype(np.float32)
