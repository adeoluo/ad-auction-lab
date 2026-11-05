"""Post-hoc calibration of click probabilities: Platt scaling and isotonic regression.

Both are fitted on a validation split and applied to test predictions. Platt scaling fits
``sigmoid(a * logit(p) + b)`` by Newton's method on the log loss; isotonic regression fits a
non-decreasing step function with pool adjacent violators and interpolates linearly between the
fitted points, as ``sklearn.isotonic.IsotonicRegression(out_of_bounds="clip")``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _logit(p: NDArray[np.float64]) -> NDArray[np.float64]:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p) - np.log1p(-p)


def _sigmoid(s: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1 / (1 + np.exp(-np.clip(s, -35, 35)))


def _platt_loss(a: float, b: float, x: NDArray[np.float64], t: NDArray[np.float64]) -> float:
    s = a * x + b
    # log(1 + e^s) - t s, computed stably
    return float((np.logaddexp(0, s) - t * s).sum())


class PlattScaling:
    """Logistic recalibration on the logit of the prediction."""

    def __init__(self, max_iter: int = 100, tol: float = 1e-10) -> None:
        self.max_iter, self.tol = max_iter, tol
        self.a, self.b = 1.0, 0.0

    def fit(self, p: ArrayLike, y: ArrayLike) -> PlattScaling:
        x = _logit(np.asarray(p, dtype=np.float64))
        t = np.asarray(y, dtype=np.float64)
        if x.shape != t.shape or x.size == 0:
            raise ValueError("predictions and labels must be non-empty and the same length")
        a, b = 1.0, 0.0
        loss = _platt_loss(a, b, x, t)
        for _ in range(self.max_iter):
            q = _sigmoid(a * x + b)
            g = np.array([((q - t) * x).sum(), (q - t).sum()])
            w = q * (1 - q)
            h = np.array([[(w * x * x).sum(), (w * x).sum()], [(w * x).sum(), w.sum()]])
            step = np.linalg.solve(h + 1e-9 * np.eye(2), g)
            # Backtracking keeps every Newton step a descent step.
            scale = 1.0
            while scale > 1e-8:
                na, nb = a - scale * step[0], b - scale * step[1]
                new = _platt_loss(na, nb, x, t)
                if new <= loss:
                    break
                scale /= 2
            else:
                break
            a, b, loss = na, nb, new
            if np.abs(scale * step).max() < self.tol:
                break
        self.a, self.b = float(a), float(b)
        return self

    def transform(self, p: ArrayLike) -> NDArray[np.float64]:
        x = _logit(np.asarray(p, dtype=np.float64))
        return _sigmoid(self.a * x + self.b)


class IsotonicCalibration:
    """Non-decreasing calibration map fitted with pool adjacent violators."""

    def __init__(self) -> None:
        self.x: NDArray[np.float64] = np.zeros(0)
        self.y: NDArray[np.float64] = np.zeros(0)

    def fit(self, p: ArrayLike, y: ArrayLike) -> IsotonicCalibration:
        x = np.asarray(p, dtype=np.float64)
        t = np.asarray(y, dtype=np.float64)
        if x.shape != t.shape or x.size == 0:
            raise ValueError("predictions and labels must be non-empty and the same length")
        order = np.argsort(x, kind="mergesort")
        xs, ts = x[order], t[order]
        # Collapse equal x values first (their mean label), as scikit-learn does.
        ux, start = np.unique(xs, return_index=True)
        sums = np.add.reduceat(ts, start)
        counts = np.diff(np.r_[start, len(xs)]).astype(np.float64)
        vals: list[float] = []
        wts: list[float] = []
        lens: list[int] = []
        for s, c in zip(sums, counts, strict=True):
            vals.append(s / c)
            wts.append(c)
            lens.append(1)
            while len(vals) > 1 and vals[-2] > vals[-1]:
                w = wts[-2] + wts[-1]
                v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
                n = lens[-2] + lens[-1]
                vals[-2:], wts[-2:], lens[-2:] = [v], [w], [n]
        self.x = ux
        self.y = np.repeat(np.array(vals), lens)
        return self

    def transform(self, p: ArrayLike) -> NDArray[np.float64]:
        if self.x.size == 0:
            raise RuntimeError("fit the calibrator first")
        return np.interp(np.asarray(p, dtype=np.float64), self.x, self.y)
