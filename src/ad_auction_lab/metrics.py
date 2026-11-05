"""Click model metrics: log loss, ROC AUC, normalized entropy, calibration and significance tests.

* Normalized entropy (He et al., 2014) is the log loss divided by the entropy of a constant
  predictor at the base rate, so 1.0 means "no better than predicting the average CTR".
* ``delong`` compares two AUCs on the same rows with the fast DeLong algorithm (Sun and Xu, 2014).
* ``paired_loss_test`` compares two models' per-row log losses with a paired t interval.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

EPS = 1e-15


def _yp(y: ArrayLike, p: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    ya = np.asarray(y, dtype=np.float64).reshape(-1)
    pa = np.asarray(p, dtype=np.float64).reshape(-1)
    if ya.shape != pa.shape or ya.size == 0:
        raise ValueError("labels and predictions must be non-empty and the same length")
    if not np.isfinite(pa).all() or (pa < 0).any() or (pa > 1).any():
        raise ValueError("predictions must be probabilities in [0, 1]")
    if not np.isin(ya, (0.0, 1.0)).all():
        raise ValueError("labels must be 0 or 1")
    return ya, pa


def row_log_loss(y: ArrayLike, p: ArrayLike) -> NDArray[np.float64]:
    """Per-row log loss with probabilities clipped to [1e-15, 1 - 1e-15]."""

    ya, pa = _yp(y, p)
    pa = np.clip(pa, EPS, 1 - EPS)
    return -(ya * np.log(pa) + (1 - ya) * np.log1p(-pa))


def log_loss(y: ArrayLike, p: ArrayLike) -> float:
    """Mean log loss (natural log), as ``sklearn.metrics.log_loss``."""

    return float(row_log_loss(y, p).mean())


def entropy(rate: float) -> float:
    """Binary entropy (nats) of a Bernoulli(rate) label."""

    r = min(max(rate, EPS), 1 - EPS)
    return -(r * math.log(r) + (1 - r) * math.log(1 - r))


def normalized_entropy(y: ArrayLike, p: ArrayLike, base_rate: float | None = None) -> float:
    """Log loss divided by the entropy of the base rate (default: the mean label of ``y``)."""

    ya, _ = _yp(y, p)
    rate = float(ya.mean()) if base_rate is None else base_rate
    return log_loss(y, p) / entropy(rate)


def midrank(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Ranks 1..n with ties given their average rank."""

    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    n = len(x)
    starts = np.flatnonzero(np.r_[True, xs[1:] != xs[:-1]])
    ends = np.r_[starts[1:], n]
    avg = (starts + ends + 1) / 2.0
    ranks_sorted = np.repeat(avg, ends - starts)
    out = np.empty(n, dtype=np.float64)
    out[order] = ranks_sorted
    return out


def roc_auc(y: ArrayLike, p: ArrayLike) -> float:
    """Area under the ROC curve (Mann-Whitney statistic, ties count one half)."""

    ya, pa = _yp(y, p)
    pos = ya == 1
    m, n = int(pos.sum()), int((~pos).sum())
    if m == 0 or n == 0:
        raise ValueError("AUC needs both classes")
    r = midrank(pa)
    return float((r[pos].sum() - m * (m + 1) / 2) / (m * n))


@dataclass
class Calibration:
    """Calibration summary: mean prediction over mean label, and expected calibration error."""

    mean_prediction: float
    mean_label: float
    ratio: float
    ece: float
    bins: list[dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def calibration(y: ArrayLike, p: ArrayLike, n_bins: int = 10) -> Calibration:
    """Reliability table on equal-count bins of the predictions, and the ECE over those bins."""

    ya, pa = _yp(y, p)
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    order = np.argsort(pa, kind="mergesort")
    groups = np.array_split(order, min(n_bins, len(order)))
    bins, ece = [], 0.0
    for g in groups:
        mp, my = float(pa[g].mean()), float(ya[g].mean())
        ece += len(g) / len(pa) * abs(mp - my)
        bins.append({"count": float(len(g)), "mean_prediction": mp, "mean_label": my})
    mp, my = float(pa.mean()), float(ya.mean())
    return Calibration(mp, my, mp / my if my > 0 else float("inf"), ece, bins)


def _normal_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2))


@dataclass
class Comparison:
    """Difference ``a - b`` with a 95% interval and a two-sided p-value."""

    metric: str
    a: float
    b: float
    difference: float
    ci_low: float
    ci_high: float
    p_value: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def delong(y: ArrayLike, p_a: ArrayLike, p_b: ArrayLike) -> Comparison:
    """DeLong test for the difference of two correlated AUCs (fast algorithm of Sun and Xu)."""

    ya, pa = _yp(y, p_a)
    _, pb = _yp(y, p_b)
    pos = ya == 1
    m, n = int(pos.sum()), int((~pos).sum())
    if m < 2 or n < 2:
        raise ValueError("DeLong needs at least two rows of each class")
    v10, v01, aucs = [], [], []
    for s in (pa, pb):
        tz = midrank(s)
        tx = midrank(s[pos])
        ty = midrank(s[~pos])
        auc = (tz[pos].sum() - m * (m + 1) / 2) / (m * n)
        v10.append((tz[pos] - tx) / n)
        v01.append(1 - (tz[~pos] - ty) / m)
        aucs.append(auc)
    s10 = np.cov(np.vstack(v10))
    s01 = np.cov(np.vstack(v01))
    s = s10 / m + s01 / n
    var = float(s[0, 0] + s[1, 1] - 2 * s[0, 1])
    diff = float(aucs[0] - aucs[1])
    se = math.sqrt(max(var, 0.0))
    p = 1.0 if se == 0 else 2 * _normal_sf(abs(diff) / se)
    return Comparison(
        "auc", float(aucs[0]), float(aucs[1]), diff, diff - 1.96 * se, diff + 1.96 * se, p
    )


def paired_loss_test(y: ArrayLike, p_a: ArrayLike, p_b: ArrayLike) -> Comparison:
    """Paired test on per-row log losses: mean difference, normal 95% interval, two-sided p."""

    la = row_log_loss(y, p_a)
    lb = row_log_loss(y, p_b)
    d = la - lb
    n = len(d)
    if n < 2:
        raise ValueError("need at least two rows")
    se = float(d.std(ddof=1) / math.sqrt(n))
    diff = float(d.mean())
    p = 1.0 if se == 0 else 2 * _normal_sf(abs(diff) / se)
    return Comparison(
        "logloss", float(la.mean()), float(lb.mean()), diff, diff - 1.96 * se, diff + 1.96 * se, p
    )


def summary(y: ArrayLike, p: ArrayLike, base_rate: float | None = None) -> dict[str, float]:
    """Log loss, AUC, normalized entropy, calibration ratio and ECE in one dict."""

    c = calibration(y, p)
    return {
        "logloss": log_loss(y, p),
        "auc": roc_auc(y, p),
        "normalized_entropy": normalized_entropy(y, p, base_rate),
        "calibration_ratio": c.ratio,
        "ece": c.ece,
        "rows": float(np.asarray(y).size),
    }
