"""Bid shading for first-price auctions.

In a first-price auction a bidder with value ``v`` who bids ``b`` wins when ``b`` beats the highest
competing bid ``M`` and then pays ``b``, so its expected surplus is ``(v - b) G(b)`` with
``G(b) = P(M < b)`` the bid landscape (win-rate curve). Bidding ``v`` earns nothing; the surplus
maximizing bid is the shaded bid ``b*(v) = argmax_b (v - b) G(b)``.

* :class:`EmpiricalLandscape` estimates ``G`` by the empirical CDF of observed market prices and
  solves the argmax exactly over the observed prices (the optimum of a step function lies on a
  step). Vectorized over many values at once with a running maximum over the sorted prices.
* :class:`LogNormalLandscape` fits ``log M ~ Normal(mu, sigma)`` and solves the first order
  condition ``v - b = G(b) / g(b)`` by bisection, a parametric shading model in the spirit of the
  production systems in Zhou et al. (2021, "An efficient deep distribution network for bid
  shading in first-price auctions").
* :func:`symmetric_equilibrium_bid` is the Bayes Nash equilibrium with ``n`` i.i.d.
  Uniform(0, 1) bidders, ``b(v) = (n - 1) v / n`` (Vickrey, 1961), used to validate the solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, erfc, exp, log, pi, sqrt

import numpy as np
from numpy.typing import ArrayLike, NDArray

F64 = NDArray[np.float64]


def symmetric_equilibrium_bid(values: ArrayLike, n_bidders: int) -> F64:
    """First-price equilibrium bid with ``n_bidders`` i.i.d. Uniform(0, 1) values."""

    if n_bidders < 1:
        raise ValueError("n_bidders must be at least 1")
    out: F64 = np.asarray(values, dtype=np.float64) * (n_bidders - 1) / n_bidders
    return out


def surplus(values: ArrayLike, bids: ArrayLike, market: ArrayLike) -> F64:
    """Realized first-price surplus ``(v - b) 1[b > m]`` for each row (ties lose)."""

    v = np.asarray(values, dtype=np.float64)
    b = np.asarray(bids, dtype=np.float64)
    m = np.asarray(market, dtype=np.float64)
    out: F64 = np.where(b > m, v - b, 0.0)
    return out


@dataclass(frozen=True)
class EmpiricalLandscape:
    """Win-rate curve from a sample of highest competing bids (market prices)."""

    prices: F64

    @classmethod
    def fit(cls, market: ArrayLike) -> EmpiricalLandscape:
        m = np.sort(np.asarray(market, dtype=np.float64).reshape(-1))
        if m.size == 0 or np.any(~np.isfinite(m)) or np.any(m < 0):
            raise ValueError(
                "market prices must be a non-empty array of finite non-negative values"
            )
        return cls(m)

    def win_rate(self, bids: ArrayLike) -> F64:
        """``G(b) = P(M < b)`` (strict, so a tie loses)."""

        b = np.asarray(bids, dtype=np.float64)
        out: F64 = np.searchsorted(self.prices, b, side="left") / self.prices.size
        return out

    def shade(self, values: ArrayLike, eps: float = 1e-9) -> F64:
        """Exact surplus-maximizing bids for ``values`` under the empirical landscape.

        Bidding just above the ``k``-th smallest price ``m_k`` wins with probability ``k / N``, so
        the optimum is ``m_k + eps`` for ``k = argmax_k (v - m_k) k / N`` (or 0 if no ``k`` is
        profitable). Using distinct price levels and their counts this is a maximum of lines
        ``k v - k m_k`` in ``v``, i.e. an upper envelope; values are processed in sorted order and
        the envelope is evaluated with a convex hull trick in ``O((N + V) log N)``.
        """

        v = np.asarray(values, dtype=np.float64)
        flat = v.reshape(-1)
        levels, counts = np.unique(self.prices, return_counts=True)
        k = np.cumsum(counts).astype(np.float64)  # wins when bidding just above levels[i]
        slope, intercept = k, -k * levels
        hull = _upper_hull(slope, intercept)
        hs, hi_ = slope[hull], intercept[hull]
        # Breakpoints between consecutive hull lines.
        xs = (hi_[:-1] - hi_[1:]) / (hs[1:] - hs[:-1])
        pos = np.searchsorted(xs, flat, side="right")
        best_val = hs[pos] * flat + hi_[pos]
        bid = levels[hull][pos] + eps
        out = np.where(best_val > 0, bid, 0.0)
        return out.reshape(v.shape)

    def shade_bruteforce(self, values: ArrayLike, eps: float = 1e-9) -> F64:
        """Quadratic reference for :meth:`shade` (tests only)."""

        v = np.asarray(values, dtype=np.float64).reshape(-1)
        levels, counts = np.unique(self.prices, return_counts=True)
        k = np.cumsum(counts) / self.prices.size
        gain = (v[:, None] - levels[None, :]) * k[None, :]
        i = np.argmax(gain, axis=1)
        best = gain[np.arange(v.size), i]
        out: F64 = np.where(best > 0, levels[i] + eps, 0.0)
        return out


def _upper_hull(slope: F64, intercept: F64) -> NDArray[np.int64]:
    """Indices of the lines on the upper envelope of ``y = slope x + intercept`` for ``x >= 0``.

    ``slope`` must be strictly increasing (true for cumulative win counts).
    """

    hull: list[int] = []

    def bad(a: int, b: int, c: int) -> bool:
        # Line b is never strictly above both a and c.
        lhs = float((intercept[c] - intercept[a]) * (slope[b] - slope[a]))
        return lhs >= float((intercept[b] - intercept[a]) * (slope[c] - slope[a]))

    for i in range(slope.size):
        while len(hull) >= 2 and bad(hull[-2], hull[-1], i):
            hull.pop()
        hull.append(i)
    # Drop leading lines that only win for x < 0.
    while len(hull) >= 2 and intercept[hull[0]] <= intercept[hull[1]]:
        hull.pop(0)
    return np.asarray(hull, dtype=np.int64)


@dataclass(frozen=True)
class LogNormalLandscape:
    """Parametric win-rate curve ``G(b) = Phi((log b - mu) / sigma)``."""

    mu: float
    sigma: float

    @classmethod
    def fit(cls, market: ArrayLike) -> LogNormalLandscape:
        m = np.asarray(market, dtype=np.float64).reshape(-1)
        m = m[m > 0]
        if m.size < 2:
            raise ValueError("need at least two positive market prices")
        lm = np.log(m)
        sigma = float(lm.std())
        if sigma <= 0:
            raise ValueError("market prices have no spread")
        return cls(float(lm.mean()), sigma)

    def win_rate(self, bids: ArrayLike) -> F64:
        b = np.asarray(bids, dtype=np.float64)
        with np.errstate(divide="ignore"):
            z = (np.log(np.maximum(b, 1e-300)) - self.mu) / self.sigma
        out: F64 = 0.5 * (1 + _erf(z / sqrt(2)))
        return np.where(b > 0, out, 0.0)

    def _shade_one(self, v: float) -> float:
        if v <= 0:
            return 0.0

        def foc(b: float) -> float:
            # Sign of the derivative of (v - b) G(b): (v - b) g(b) / G(b) - 1, with G from erfc so
            # that the far left tail does not underflow to 0 / 0.
            z = (log(b) - self.mu) / self.sigma
            cdf = 0.5 * erfc(-z / sqrt(2))
            pdf = exp(-0.5 * z * z) / (b * self.sigma * sqrt(2 * pi))
            return (v - b) * pdf / cdf - 1 if cdf > 0 else 1.0

        # (v - b) g(b) / G(b) decreases in b for the log-normal (its reversed hazard rate
        # decreases), so the first order condition has one root on (0, v), found by bisection.
        lo, hi = max(v * 1e-12, exp(self.mu - 30 * self.sigma)), v
        if lo >= hi or foc(lo) <= 0:
            return 0.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if foc(mid) > 0:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-12 * v:
                break
        return 0.5 * (lo + hi)

    def shade(self, values: ArrayLike, grid_size: int = 4096) -> F64:
        """Optimal bids: exact bisection on a log-spaced grid of values, interpolated in between."""

        v = np.asarray(values, dtype=np.float64)
        flat = v.reshape(-1)
        pos = flat[flat > 0]
        if pos.size == 0:
            return np.zeros_like(v)
        lo, hi = float(np.min(pos)), float(np.max(pos))
        if hi / lo < 1 + 1e-9 or pos.size <= grid_size:
            out = np.array([self._shade_one(float(x)) for x in flat])
            return out.reshape(v.shape)
        grid = np.geomspace(lo, hi, grid_size)
        ratios = np.array([self._shade_one(float(g)) / g for g in grid])
        r = np.interp(np.log(np.maximum(flat, lo)), np.log(grid), ratios)
        out = np.where(flat > 0, r * flat, 0.0)
        return out.reshape(v.shape)


def _erf(x: F64) -> F64:
    """Element-wise ``math.erf`` (numpy has no error function and scipy is optional)."""

    out: F64 = np.vectorize(erf, otypes=[np.float64])(x)
    return out


def shading_report(
    values: ArrayLike, market: ArrayLike, bids: dict[str, ArrayLike]
) -> dict[str, dict[str, float]]:
    """Surplus, win rate and spend of each bidding rule on the same impressions."""

    v = np.asarray(values, dtype=np.float64)
    m = np.asarray(market, dtype=np.float64)
    out: dict[str, dict[str, float]] = {}
    for name, b in bids.items():
        bb = np.asarray(b, dtype=np.float64)
        win = bb > m
        out[name] = {
            "surplus": float(np.where(win, v - bb, 0.0).sum()),
            "win_rate": float(win.mean()),
            "spend": float(np.where(win, bb, 0.0).sum()),
            "value_won": float(np.where(win, v, 0.0).sum()),
        }
    return out
