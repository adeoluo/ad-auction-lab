"""Budget pacing in a repeated multi-advertiser second-price auction market.

Each of ``n`` advertisers has a value per click ``v_i``, a budget ``B_i`` for a horizon of ``T``
auctions, and a predicted CTR ``c_{t,i}`` in auction ``t``; it bids ``m_{t,i} v_i`` per click, i.e.
an eCPM of ``m_{t,i} v_i c_{t,i}``. The highest eCPM wins if it beats both an external market
eCPM ``x_t`` (the best bid from outside the simulated advertisers) and the reserve, and pays the
larger of the second eCPM, ``x_t`` and the reserve. Value and spend are expectations per
impression. The pacing multiplier ``m`` comes from:

* ``none``: ``m = 1`` (bid the value until the budget runs out).
* ``pid``: a PID controller on the normalized spend error
  ``e = (B t / T - spend_t) / B`` updated every ``interval`` auctions,
  ``m = clip(exp(kp e + ki sum e + kd (e - e_prev)), m_min, 1)``.
* ``dual``: dual (Lagrangian) descent on the budget constraint (Balseiro and Gur, 2019). The
  Lagrangian of ``max sum value s.t. sum spend <= B`` gives the bid ``v / (1 + mu)``, and after
  every auction ``mu <- max(0, mu - eta (rho - z_t) / rho)`` with ``rho = B / T`` the target
  spend per auction and ``z_t`` the spend in auction ``t``. With i.i.d. markets this achieves
  ``O(sqrt(T))`` regret against the best fixed multiplier in hindsight.

The simulator runs in the Rust core (``rust/src/pacing.rs``); :func:`simulate_reference` is the
plain Python reference used by the parity tests. :func:`hindsight_optimum` gives the fractional
knapsack upper bound for one advertiser facing an exogenous market.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ad_auction_lab.backend import extension, resolve_backend

PacerKind = Literal["none", "pid", "dual"]
F64 = NDArray[np.float64]


@dataclass(frozen=True)
class Pacer:
    """Pacing rule and its parameters (see the module docstring)."""

    kind: PacerKind = "dual"
    kp: float = 2.0
    ki: float = 0.05
    kd: float = 0.0
    interval: int = 1000
    m_min: float = 0.0
    step: float = 0.002

    def __post_init__(self) -> None:
        if self.kind not in ("none", "pid", "dual"):
            raise ValueError("pacer kind must be 'none', 'pid' or 'dual'")
        if self.interval < 1:
            raise ValueError("interval must be positive")
        if self.step < 0 or not 0 <= self.m_min <= 1:
            raise ValueError("step must be non-negative and m_min in [0, 1]")


@dataclass
class PacingResult:
    value: F64
    spend: F64
    wins: NDArray[np.int64]
    exhausted_at: NDArray[np.int64]
    trajectory: F64
    revenue: float
    market_wins: int
    unsold: int
    budgets: F64 = field(repr=False)
    horizon: int = 0

    def summary(self) -> dict[str, Any]:
        """Totals and pacing quality: utilization, exhaustion time, tracking error."""

        b = self.budgets
        util = np.divide(self.spend, b, out=np.zeros_like(b), where=b > 0)
        c = self.trajectory.shape[0]
        ideal = np.arange(1, c + 1)[:, None] / c * b[None, :]
        track = np.divide(
            np.abs(self.trajectory - ideal), b[None, :], out=np.zeros_like(ideal), where=b > 0
        )
        return {
            "value": float(self.value.sum()),
            "spend": float(self.spend.sum()),
            "revenue": self.revenue,
            "roi": float(self.value.sum() / self.spend.sum()) if self.spend.sum() > 0 else 0.0,
            "utilization": float(util.mean()),
            "exhausted_share": float(np.mean(self.exhausted_at < self.horizon)),
            "mean_exhaustion_time": float(np.mean(self.exhausted_at) / max(self.horizon, 1)),
            "max_tracking_error": float(track.max()),
            "mean_tracking_error": float(track.mean()),
            "market_wins": self.market_wins,
            "unsold": self.unsold,
        }


def _check(
    values: ArrayLike, budgets: ArrayLike, ctr: ArrayLike, market: ArrayLike
) -> tuple[F64, F64, NDArray[np.float32], NDArray[np.float32]]:
    v = np.ascontiguousarray(values, dtype=np.float64).reshape(-1)
    b = np.ascontiguousarray(budgets, dtype=np.float64).reshape(-1)
    c = np.ascontiguousarray(ctr, dtype=np.float32)
    m = np.ascontiguousarray(market, dtype=np.float32).reshape(-1)
    if c.ndim == 1:
        c = c.reshape(-1, 1)
    if c.ndim != 2 or c.shape != (m.size, v.size):
        raise ValueError("ctr must be (T, n_advertisers) and market (T,)")
    return v, b, c, m


def simulate(
    values: ArrayLike,
    budgets: ArrayLike,
    ctr: ArrayLike,
    market: ArrayLike,
    pacer: Pacer | None = None,
    reserve: float = 0.0,
    checkpoints: int = 24,
    backend: str = "auto",
) -> PacingResult:
    """Run the market for ``T = len(market)`` auctions with every advertiser using ``pacer``."""

    p = pacer or Pacer()
    v, b, c, m = _check(values, budgets, ctr, market)
    if resolve_backend(backend) == "python":
        return simulate_reference(v, b, c, m, p, reserve, checkpoints)
    d = extension().simulate_pacing(
        v,
        b,
        c.reshape(-1),
        m,
        float(reserve),
        p.kind,
        checkpoints,
        kp=p.kp,
        ki=p.ki,
        kd=p.kd,
        interval=p.interval,
        m_min=p.m_min,
        step=p.step,
    )
    cps = max(checkpoints, 1)
    return PacingResult(
        value=np.asarray(d["value"]),
        spend=np.asarray(d["spend"]),
        wins=np.asarray(d["wins"], dtype=np.int64),
        exhausted_at=np.asarray(d["exhausted_at"], dtype=np.int64),
        trajectory=np.frombuffer(d["trajectory"], dtype=np.float64).reshape(cps, v.size),
        revenue=float(d["revenue"]),
        market_wins=int(d["market_wins"]),
        unsold=int(d["unsold"]),
        budgets=b,
        horizon=m.size,
    )


def simulate_reference(
    values: ArrayLike,
    budgets: ArrayLike,
    ctr: ArrayLike,
    market: ArrayLike,
    pacer: Pacer | None = None,
    reserve: float = 0.0,
    checkpoints: int = 24,
) -> PacingResult:
    """Plain Python version of the Rust simulator, line for line (slow; for parity tests)."""

    p = pacer or Pacer()
    v, b, c, mk = _check(values, budgets, ctr, market)
    if not math.isfinite(reserve) or reserve < 0:
        raise ValueError("reserve must be non-negative")
    if v.size == 0 or b.size != v.size:
        raise ValueError("values and budgets must be non-empty and the same length")
    if np.any(~np.isfinite(v)) or np.any(v < 0) or np.any(~np.isfinite(b)) or np.any(b < 0):
        raise ValueError("values and budgets must be finite and non-negative")
    n, T = v.size, mk.size
    cps = max(checkpoints, 1)
    value, spend = [0.0] * n, [0.0] * n
    wins = [0] * n
    exhausted = [T] * n
    traj = np.zeros((cps, n))
    revenue, market_wins, unsold = 0.0, 0, 0
    mult, mu = [1.0] * n, [0.0] * n
    integral, prev = [0.0] * n, [0.0] * n
    rho = [float(x) / max(T, 1) for x in b]
    nxt = 0
    for t in range(T):
        best, second, winner = 0.0, 0.0, -1
        for i in range(n):
            factor = 1.0 if p.kind == "none" else mult[i] if p.kind == "pid" else 1 / (1 + mu[i])
            ecpm = float(v[i]) * factor * float(c[t, i])
            if ecpm <= 0 or float(b[i]) - spend[i] < ecpm:
                if ecpm > 0 and exhausted[i] == T:
                    exhausted[i] = t
                continue
            if ecpm > best:
                second, best, winner = best, ecpm, i
            elif ecpm > second:
                second = ecpm
        floor = max(float(mk[t]), reserve)
        paid, price = -1, 0.0
        if winner >= 0 and best > floor:
            price = max(second, floor)
            spend[winner] += price
            value[winner] += float(v[winner]) * float(c[t, winner])
            wins[winner] += 1
            revenue += price
            paid = winner
        elif float(mk[t]) >= reserve and float(mk[t]) > 0:
            market_wins += 1
        else:
            unsold += 1
        if p.kind == "dual":
            for i in range(n):
                if rho[i] <= 0:
                    continue
                z = price if i == paid else 0.0
                mu[i] = max(mu[i] - p.step * (rho[i] - z) / rho[i], 0.0)
        elif p.kind == "pid" and (t + 1) % p.interval == 0:
            for i in range(n):
                if b[i] <= 0:
                    continue
                err = (float(b[i]) * (t + 1) / T - spend[i]) / float(b[i])
                integral[i] += err
                u = p.kp * err + p.ki * integral[i] + p.kd * (err - prev[i])
                prev[i] = err
                mult[i] = min(max(math.exp(u), p.m_min), 1.0)
        while nxt < cps and (t + 1) * cps >= (nxt + 1) * T:
            traj[nxt] = spend
            nxt += 1
    while nxt < cps:
        traj[nxt] = spend
        nxt += 1
    return PacingResult(
        value=np.asarray(value),
        spend=np.asarray(spend),
        wins=np.asarray(wins, dtype=np.int64),
        exhausted_at=np.asarray(exhausted, dtype=np.int64),
        trajectory=traj,
        revenue=revenue,
        market_wins=market_wins,
        unsold=unsold,
        budgets=b,
        horizon=T,
    )


def hindsight_optimum(value: ArrayLike, price: ArrayLike, budget: float) -> dict[str, float]:
    """Best value one advertiser could get with full knowledge of the future (LP relaxation).

    With exogenous prices, buying impression ``t`` costs ``price_t`` and yields ``value_t``; the
    LP ``max sum x_t value_t s.t. sum x_t price_t <= B, 0 <= x <= 1`` is a fractional knapsack,
    solved by taking impressions in decreasing ``value / price`` order (free ones first). Its
    optimum upper-bounds every online policy, so ``achieved / optimum`` is a competitive ratio.
    Also returns the best fixed multiplier ``1 / (1 + mu*)``, where ``mu*`` is the LP dual.
    """

    val = np.asarray(value, dtype=np.float64).reshape(-1)
    pr = np.asarray(price, dtype=np.float64).reshape(-1)
    if val.shape != pr.shape or np.any(pr < 0) or np.any(val < 0) or budget < 0:
        raise ValueError("value and price must be non-negative arrays of equal length")
    useful = val > 0
    val, pr = val[useful], pr[useful]
    free = pr <= 0
    total = float(val[free].sum())
    val, pr = val[~free], pr[~free]
    ratio = val / pr
    order = np.argsort(-ratio, kind="stable")
    cum = np.cumsum(pr[order])
    k = int(np.searchsorted(cum, budget, side="right"))
    total += float(val[order[:k]].sum())
    spent = float(cum[k - 1]) if k else 0.0
    mu = 0.0
    if k < order.size:
        frac = (budget - spent) / pr[order[k]]
        total += frac * float(val[order[k]])
        spent = budget
        mu = max(float(ratio[order[k]]) - 1.0, 0.0)
    return {"value": total, "spend": spent, "multiplier": 1.0 / (1.0 + mu), "dual": mu}


@dataclass
class MarketSample:
    """Inputs of :func:`simulate` built from real price and CTR samples."""

    values: F64
    budgets: F64
    ctr: NDArray[np.float32]
    market: NDArray[np.float32]
    unpaced_spend: F64


def build_market(
    prices: ArrayLike,
    ctr_pool: ArrayLike,
    horizon: int,
    n_advertisers: int,
    budget_share: float = 0.3,
    value_spread: tuple[float, float] = (0.5, 2.0),
    seed: int = 0,
) -> MarketSample:
    """A market of ``horizon`` auctions from real data.

    The external market eCPM is a contiguous, time-ordered window of observed prices (so daily
    and weekly patterns are kept), each advertiser's CTR in each auction is drawn from
    ``ctr_pool`` (predicted CTRs of a click model on held-out impressions), and values per click
    are ``u_i * median(price) / median(ctr)`` with ``u_i`` log-uniform on ``value_spread``. Each
    budget is ``budget_share`` times what the advertiser spends without a budget, so
    ``budget_share < 1`` makes every budget binding.
    """

    p = np.asarray(prices, dtype=np.float32).reshape(-1)
    pool = np.asarray(ctr_pool, dtype=np.float32).reshape(-1)
    if horizon < 1 or horizon > p.size or n_advertisers < 1 or pool.size == 0:
        raise ValueError("need 1 <= horizon <= len(prices), n_advertisers >= 1, a CTR pool")
    if not budget_share > 0:
        raise ValueError("budget_share must be positive")
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, p.size - horizon + 1))
    market = np.ascontiguousarray(p[start : start + horizon])
    ctr = pool[rng.integers(0, pool.size, (horizon, n_advertisers))]
    scale = float(np.median(market)) / max(float(np.median(pool)), 1e-12)
    lo, hi = value_spread
    values = scale * np.exp(rng.uniform(np.log(lo), np.log(hi), n_advertisers))
    free = simulate(values, np.full(n_advertisers, 1e30), ctr, market, Pacer("none"))
    budgets = np.maximum(free.spend * budget_share, 1e-12)
    return MarketSample(values, budgets, ctr, market, free.spend)
