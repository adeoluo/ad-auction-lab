"""Position auctions for sponsored search and display slots: GSP, VCG and first price.

Model. Bidder ``i`` reports a bid ``b_i`` per click and has a quality score ``q_i`` (its predicted
click-through rate). Slot ``j`` has a position effect ``alpha_j`` (non-increasing), and an ad in
slot ``j`` is clicked with probability ``alpha_j q_i`` (the separable model of Varian, 2007 and
Edelman, Ostrovsky and Schwarz, 2007). Bidders are ranked by the score ``s_i = q_i^lambda b_i``,
where the squashing exponent ``lambda`` (Lahaie and Pennock, 2007) trades efficiency
(``lambda = 1``, rank by expected value per impression) against revenue (``lambda < 1`` helps
bidders with low predicted CTR).

* **GSP** (generalized second price): the bidder in slot ``j`` pays per click the smallest bid that
  keeps its rank, ``p_j = max(s_{j+1} / q_j^lambda, r)``, where ``r`` is the per-click reserve and
  ``s_{j+1}`` the next eligible score. With one slot this is the Vickrey auction.
* **VCG**: an affine maximizer. The allocation maximizes ``sum_j alpha_j q_i^lambda (b_i - r)``
  (welfare with weights ``q^(lambda - 1)`` and a seller cost ``r`` per click), and each bidder pays
  ``r`` per click plus its weighted externality on the others, which in a position auction is
  ``E_j = sum_{k > j} (alpha_{k-1} - alpha_k) s'_k`` with ``s'_k = q^lambda (b - r)`` the k-th
  shifted score and ``alpha_K = 0``. The per-click price is ``r + E_j / (alpha_j q_i^lambda)``.
  Affine maximizers with this payment are dominant-strategy truthful for every ``lambda`` and
  ``r``, which ``tests/test_auctions.py`` checks numerically.
* **First price**: GSP ranking, and each winner pays its own bid per click.

Every function is vectorized over a batch of ``T`` independent auctions given as ``(T, n)``
arrays; a single auction is a batch of one. Ties are broken by bidder index (lower wins).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

Mechanism = Literal["gsp", "vcg", "first_price"]
MECHANISMS: tuple[Mechanism, ...] = ("gsp", "vcg", "first_price")

F64 = NDArray[np.float64]


@dataclass(frozen=True)
class AuctionResult:
    """Outcome of ``T`` auctions with ``K`` slots.

    ``winner[t, j]`` is the bidder index in slot ``j`` (``-1`` if the slot stays empty) and
    ``price[t, j]`` the price per click that bidder pays (0 for an empty slot).
    """

    winner: NDArray[np.int64]
    price: F64
    slot_ctr: F64
    quality: F64

    @property
    def filled(self) -> NDArray[np.bool_]:
        return self.winner >= 0

    def clicks(self) -> F64:
        """Expected clicks ``alpha_j q_i`` of each slot, ``(T, K)``."""

        q = np.take_along_axis(self.quality, np.maximum(self.winner, 0), axis=1)
        out: F64 = np.where(self.filled, self.slot_ctr[None, :] * q, 0.0)
        return out

    def revenue(self) -> F64:
        """Expected seller revenue per auction, ``(T,)``."""

        out: F64 = (self.clicks() * self.price).sum(axis=1)
        return out

    def welfare(self, values: ArrayLike) -> F64:
        """Expected welfare ``sum_j alpha_j q_i v_i`` per auction under true per-click values."""

        v = _as_2d(values, self.quality.shape)
        vw = np.take_along_axis(v, np.maximum(self.winner, 0), axis=1)
        out: F64 = (self.clicks() * vw).sum(axis=1)
        return out

    def utilities(self, values: ArrayLike) -> F64:
        """Expected utility ``alpha_j q_i (v_i - p_j)`` of every bidder, ``(T, n)``."""

        v = _as_2d(values, self.quality.shape)
        T, n = self.quality.shape
        out = np.zeros((T, n))
        rows = np.repeat(np.arange(T), self.winner.shape[1])
        who = self.winner.reshape(-1)
        keep = who >= 0
        surplus = self.clicks() * (
            np.take_along_axis(v, np.maximum(self.winner, 0), axis=1) - self.price
        )
        np.add.at(out, (rows[keep], who[keep]), surplus.reshape(-1)[keep])
        return out

    def payments(self) -> F64:
        """Expected payment of every bidder, ``(T, n)``."""

        T, n = self.quality.shape
        out = np.zeros((T, n))
        rows = np.repeat(np.arange(T), self.winner.shape[1])
        who = self.winner.reshape(-1)
        keep = who >= 0
        np.add.at(out, (rows[keep], who[keep]), (self.clicks() * self.price).reshape(-1)[keep])
        return out


def _as_2d(x: ArrayLike, shape: tuple[int, ...] | None = None) -> F64:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2:
        raise ValueError("expected a (n,) or (T, n) array")
    if shape is not None and a.shape != shape:
        a = np.broadcast_to(a, shape).astype(np.float64)
    return a


def _check_slots(slot_ctr: ArrayLike) -> F64:
    a = np.asarray(slot_ctr, dtype=np.float64).reshape(-1)
    if a.size == 0:
        raise ValueError("at least one slot is needed")
    if np.any(~np.isfinite(a)) or np.any(a < 0) or np.any(np.diff(a) > 1e-12):
        raise ValueError("slot click-through rates must be finite, non-negative, non-increasing")
    return a


def rank(scores: F64, eligible: NDArray[np.bool_]) -> NDArray[np.int64]:
    """Bidder indices ordered by descending score; ineligible bidders go last."""

    key = np.where(eligible, scores, -np.inf)
    out: NDArray[np.int64] = np.argsort(-key, axis=1, kind="stable").astype(np.int64)
    return out


def run_auctions(
    bids: ArrayLike,
    quality: ArrayLike,
    slot_ctr: ArrayLike,
    mechanism: Mechanism = "gsp",
    reserve: float = 0.0,
    squash: float = 1.0,
) -> AuctionResult:
    """Run ``T`` independent position auctions; ``bids`` and ``quality`` are ``(T, n)`` or ``(n,)``.

    ``reserve`` is a per-click reserve price and ``squash`` the exponent ``lambda`` on quality in
    the ranking score. Returns the winners and per-click prices of every slot.
    """

    b = _as_2d(bids)
    q = _as_2d(quality, b.shape)
    alpha = _check_slots(slot_ctr)
    if mechanism not in MECHANISMS:
        raise ValueError(f"unknown mechanism {mechanism!r}; expected one of {MECHANISMS}")
    if not np.isfinite(reserve) or reserve < 0:
        raise ValueError("reserve must be finite and non-negative")
    if not np.isfinite(squash) or squash < 0:
        raise ValueError("squash must be finite and non-negative")
    if np.any(~np.isfinite(b)) or np.any(b < 0) or np.any(~np.isfinite(q)) or np.any(q < 0):
        raise ValueError("bids and quality scores must be finite and non-negative")

    T = b.shape[0]
    K = alpha.size
    w = q**squash
    if mechanism == "vcg":
        scores = w * (b - reserve)
        eligible = (b >= reserve) & (scores > 0)
    else:
        scores = w * b
        eligible = (b >= reserve) & (scores > 0)
    order = rank(scores, eligible)
    # Pad with one empty column so that "next score" and "slots beyond n" are uniform.
    padded = np.concatenate([order, np.full((T, K + 1), -1, dtype=np.int64)], axis=1)[:, : K + 1]
    ok = np.concatenate(
        [np.take_along_axis(eligible, order, axis=1), np.zeros((T, K + 1), bool)], 1
    )[:, : K + 1]
    sorted_scores = np.where(
        ok, np.take_along_axis(scores, np.maximum(padded, 0), axis=1), 0.0
    )  # (T, K + 1)
    winner = np.where(ok[:, :K], padded[:, :K], -1)
    w_win = np.take_along_axis(w, np.maximum(winner, 0), axis=1)
    safe_w = np.where(w_win > 0, w_win, 1.0)

    if mechanism == "gsp":
        price = np.maximum(sorted_scores[:, 1:] / safe_w, reserve)
    elif mechanism == "first_price":
        price = np.take_along_axis(b, np.maximum(winner, 0), axis=1)
    else:
        a_ext = np.concatenate([alpha, [0.0]])
        # d_k = (alpha_{k-1} - alpha_k) s'_k for k = 1..K; E_j = sum_{k > j} d_k.
        d = (a_ext[:-1] - a_ext[1:])[None, :] * sorted_scores[:, 1:]
        ext = np.cumsum(d[:, ::-1], axis=1)[:, ::-1]
        denom = alpha[None, :] * safe_w
        price = reserve + np.divide(ext, denom, out=np.zeros_like(ext), where=denom > 0)
    price = np.where(winner >= 0, price, 0.0)
    return AuctionResult(winner, price, alpha, q)


def vcg_bruteforce(
    bids: ArrayLike,
    quality: ArrayLike,
    slot_ctr: ArrayLike,
    reserve: float = 0.0,
    squash: float = 1.0,
) -> tuple[NDArray[np.int64], F64]:
    """Reference VCG for one auction: exhaustive assignment and Clarke pivot payments.

    Enumerates every assignment of bidders to slots (small ``n`` only) to maximize the weighted
    welfare ``sum alpha_j q^lambda (b - r)``; the payment of each winner is ``r`` per click plus
    ``(W_{-i} - W_{-i}(chosen)) / q_i^lambda`` per unit of ``alpha_j``. Returns ``(winner, price)``
    for the ``K`` slots, matching :func:`run_auctions` with ``mechanism="vcg"``.
    """

    from itertools import permutations

    b = np.asarray(bids, dtype=np.float64).reshape(-1)
    q = np.asarray(quality, dtype=np.float64).reshape(-1)
    alpha = _check_slots(slot_ctr)
    n, K = b.size, alpha.size
    if n > 8:
        raise ValueError("brute force is limited to 8 bidders")
    w = q**squash
    val = np.where(b >= reserve, w * (b - reserve), 0.0)

    def best(exclude: int) -> tuple[float, tuple[int, ...]]:
        people = [i for i in range(n) if i != exclude and val[i] > 0]
        top, arg = 0.0, tuple([-1] * K)
        for m in range(min(K, len(people)) + 1):
            for perm in permutations(people, m):
                total = float(sum(alpha[j] * val[i] for j, i in enumerate(perm)))
                if total > top + 1e-15:
                    top, arg = total, tuple(list(perm) + [-1] * (K - m))
        return top, arg

    welfare, assign = best(-1)
    winner = np.array(assign, dtype=np.int64)
    price = np.zeros(K)
    for j, i in enumerate(assign):
        if i < 0:
            continue
        without_i, _ = best(i)
        others = welfare - alpha[j] * val[i]
        price[j] = reserve + (without_i - others) / (alpha[j] * w[i])
    return winner, price


def gsp_envy_free_bids(
    values: ArrayLike, quality: ArrayLike, slot_ctr: ArrayLike, squash: float = 1.0
) -> F64:
    """Lowest locally envy-free equilibrium bids of GSP (Edelman, Ostrovsky and Schwarz, 2007).

    Bidders are sorted by score ``q^lambda v``. In score units the bidder in position ``k``
    (0-based, ``k >= 1``) bids
    ``B_k = alpha_k / alpha_{k-1} B_{k+1} + (1 - alpha_k / alpha_{k-1}) s_k``, the first loser
    bids its value, and the top bidder bids its value. Under this equilibrium GSP has
    the same allocation and revenue as truthful VCG (their Theorem 2), which the tests check.
    Returns per-click bids for one auction in the original bidder order.
    """

    v = np.asarray(values, dtype=np.float64).reshape(-1)
    q = np.asarray(quality, dtype=np.float64).reshape(-1)
    alpha = _check_slots(slot_ctr)
    w = q**squash
    s = w * v
    order = np.argsort(-s, kind="stable")
    n, K = v.size, alpha.size
    score_bid = s[order].copy()
    for k in range(min(K, n - 1), 0, -1):
        if k >= K:
            continue
        nxt = score_bid[k + 1] if k + 1 < n else 0.0
        ratio = alpha[k] / alpha[k - 1] if alpha[k - 1] > 0 else 0.0
        score_bid[k] = ratio * nxt + (1 - ratio) * s[order][k]
    out = np.empty(n)
    out[order] = np.divide(score_bid, w[order], out=np.zeros(n), where=w[order] > 0)
    return out


def deviation_gain(
    values: ArrayLike,
    quality: ArrayLike,
    slot_ctr: ArrayLike,
    mechanism: Mechanism,
    bidder: int,
    factors: ArrayLike = (0.0, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 1.05, 1.1, 1.25, 1.5, 2.0, 4.0),
    reserve: float = 0.0,
    squash: float = 1.0,
    others: ArrayLike | None = None,
) -> F64:
    """Best utility gain of ``bidder`` from shading its bid, per auction (``(T,)``, never negative).

    The bidder compares the truthful bid ``v`` with every ``factor * v`` while the others bid
    ``others`` (truthfully when ``None``). In a dominant-strategy truthful mechanism the result is 0
    up to rounding for every auction; positive values are profitable deviations.
    """

    v = _as_2d(values)
    q = _as_2d(quality, v.shape)
    base = v.copy() if others is None else _as_2d(others, v.shape).copy()
    base[:, bidder] = v[:, bidder]
    truthful = run_auctions(base, q, slot_ctr, mechanism, reserve, squash).utilities(v)[:, bidder]
    best = truthful.copy()
    for f in np.asarray(factors, dtype=np.float64).reshape(-1):
        dev = base.copy()
        dev[:, bidder] = f * v[:, bidder]
        u = run_auctions(dev, q, slot_ctr, mechanism, reserve, squash).utilities(v)[:, bidder]
        best = np.maximum(best, u)
    out: F64 = best - truthful
    return out


def myerson_reserve_uniform(high: float = 1.0) -> float:
    """Optimal reserve for i.i.d. Uniform(0, high) values: the root of ``r = (1 - F(r)) / f(r)``."""

    if high <= 0:
        raise ValueError("high must be positive")
    return high / 2


def myerson_reserve_lognormal(mu: float, sigma: float, tol: float = 1e-12) -> float:
    """Optimal (Myerson, 1981) reserve for i.i.d. log-normal values.

    Solves ``r = (1 - F(r)) / f(r)`` by bisection; the log-normal hazard rate makes
    ``r - (1 - F(r)) / f(r)`` increasing on the bracket used, so the root is unique there.
    """

    from math import erf, exp, log, pi, sqrt

    if sigma <= 0:
        raise ValueError("sigma must be positive")

    def phi(r: float) -> float:
        z = (log(r) - mu) / sigma
        sf = 0.5 * (1 - erf(z / sqrt(2)))
        pdf = exp(-0.5 * z * z) / (r * sigma * sqrt(2 * pi))
        return r - sf / pdf if pdf > 0 else r

    lo, hi = exp(mu - 8 * sigma), exp(mu + 8 * sigma)
    for _ in range(300):
        mid = sqrt(lo * hi)
        if phi(mid) > 0:
            hi = mid
        else:
            lo = mid
        if hi / lo - 1 < tol:
            break
    return sqrt(lo * hi)


def empirical_reserve(values: ArrayLike, grid: ArrayLike | None = None) -> tuple[float, float]:
    """Revenue-maximizing posted price for a sample of values: ``argmax_r r P(V >= r)``.

    Returns ``(reserve, revenue per bidder)``. Without ``grid`` every sample value is a candidate,
    which gives the exact empirical optimum in ``O(n log n)``.
    """

    v = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    if v.size == 0:
        raise ValueError("no values")
    if grid is None:
        cand = v
        survive = (v.size - np.arange(v.size)) / v.size
    else:
        cand = np.asarray(grid, dtype=np.float64).reshape(-1)
        survive = (v.size - np.searchsorted(v, cand, side="left")) / v.size
    rev = cand * survive
    i = int(np.argmax(rev))
    return float(cand[i]), float(rev[i])
