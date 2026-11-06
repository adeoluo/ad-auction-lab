"""Off-policy estimators for (item, position) bandit logs: IPS, SNIPS, DM, DR, SNDR, Switch-DR.

A log has ``n`` rounds; in round ``i`` the behavior policy showed item (action) ``a_i`` at position
``k_i`` with probability (propensity) ``p_i`` and observed the reward ``r_i`` (a click). An
evaluation policy ``pi(a | x, k)`` is given as an ``(n_actions, n_positions)`` array when it
ignores the context or as ``(n, n_actions, n_positions)`` otherwise. The importance weight of a
round is ``w_i = pi(a_i | x_i, k_i) / p_i``. With a reward model ``q(x, a, k)``:

* IPS (Horvitz and Thompson, 1952): ``V = mean(w r)``; unbiased when ``p > 0`` wherever
  ``pi > 0``, but high variance when weights are large. ``clip`` caps ``w`` (Strehl et al., 2010).
* SNIPS (Swaminathan and Joachims, 2015): ``V = sum(w r) / sum(w)``, biased but consistent and
  bounded in ``[min r, max r]``.
* DM: ``V = mean(q(x_i, pi, k_i))`` with ``q(x, pi, k) = sum_a pi(a | x, k) q(x, a, k)``; low
  variance, biased by model error.
* DR (Dudik et al., 2011): ``V = mean(q(x_i, pi, k_i) + w_i (r_i - q(x_i, a_i, k_i)))``; unbiased
  if either the propensities or the reward model are correct.
* SNDR: DM term plus the self-normalized correction ``sum(w (r - q)) / sum(w)``.
* Switch-DR (Wang et al., 2017): DR on rounds with ``w <= tau`` and DM alone where ``w > tau``,
  ``V = mean(q(x_i, pi, k_i) + 1[w_i <= tau] w_i (r_i - q(x_i, a_i, k_i)))``.

These definitions follow the Open Bandit Pipeline 0.5.7 (Saito et al., 2021) exactly, including
the normalization of ``pi`` over actions at each position in the DM term, so the validation tests
can require agreement to floating point precision.

Each estimator is written as ``mean(a) + sum(b) / sum(c)`` of per-round terms
(:class:`RoundTerms`), which makes the Poisson bootstrap (:func:`bootstrap`) a pair of weighted
sums per replicate instead of a re-run of the estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

F64 = NDArray[np.float64]
Estimator = Literal["ips", "snips", "dm", "dr", "sndr", "switch-dr"]
ESTIMATORS: tuple[Estimator, ...] = ("ips", "snips", "dm", "dr", "sndr", "switch-dr")
NEEDS_MODEL = frozenset({"dm", "dr", "sndr", "switch-dr"})


@dataclass(frozen=True)
class BanditLog:
    """Logged rounds of a ranking (slate) bandit."""

    reward: F64
    action: NDArray[np.int64]
    position: NDArray[np.int64]
    pscore: F64
    n_actions: int
    n_positions: int

    @classmethod
    def from_arrays(
        cls,
        reward: ArrayLike,
        action: ArrayLike,
        pscore: ArrayLike,
        position: ArrayLike | None = None,
        n_actions: int | None = None,
        n_positions: int | None = None,
    ) -> BanditLog:
        r = np.asarray(reward, dtype=np.float64).reshape(-1)
        a = np.asarray(action, dtype=np.int64).reshape(-1)
        p = np.asarray(pscore, dtype=np.float64).reshape(-1)
        k = np.zeros_like(a) if position is None else np.asarray(position, dtype=np.int64)
        k = k.reshape(-1)
        n = r.size
        if n == 0 or a.size != n or p.size != n or k.size != n:
            raise ValueError("reward, action, pscore and position must be non-empty, same length")
        if np.any(~np.isfinite(r)):
            raise ValueError("rewards must be finite")
        if np.any(~(p > 0)) or np.any(p > 1):
            raise ValueError("propensity scores must be in (0, 1]")
        if np.min(a) < 0 or np.min(k) < 0:
            raise ValueError("actions and positions must be non-negative")
        na = int(np.max(a)) + 1 if n_actions is None else n_actions
        nk = int(np.max(k)) + 1 if n_positions is None else n_positions
        if np.max(a) >= na or np.max(k) >= nk:
            raise ValueError("action or position out of range")
        return cls(r, a, k, p, na, nk)

    def __len__(self) -> int:
        return int(self.reward.size)

    def take(self, rows: NDArray[np.integer]) -> BanditLog:
        return BanditLog(
            self.reward[rows],
            self.action[rows],
            self.position[rows],
            self.pscore[rows],
            self.n_actions,
            self.n_positions,
        )


def _check_policy(log: BanditLog, pi: NDArray[np.floating]) -> None:
    shape = (log.n_actions, log.n_positions)
    if pi.shape != shape and pi.shape != (len(log), *shape):
        raise ValueError(f"action_dist must be {shape} or (n, *{shape}), got {pi.shape}")
    if np.any(pi < 0) or np.any(~np.isfinite(pi)):
        raise ValueError("action_dist must be finite and non-negative")


def importance_weights(log: BanditLog, action_dist: ArrayLike) -> F64:
    """``w_i = pi(a_i | x_i, k_i) / p_i``."""

    pi = np.asarray(action_dist, dtype=np.float64)
    _check_policy(log, pi)
    if pi.ndim == 2:
        num = pi[log.action, log.position]
    else:
        num = pi[np.arange(len(log)), log.action, log.position]
    out: F64 = num / log.pscore
    return out


def model_terms(log: BanditLog, action_dist: ArrayLike, q_hat: ArrayLike) -> tuple[F64, F64]:
    """``(q(x_i, pi, k_i), q(x_i, a_i, k_i))`` from a reward model of the same shape options.

    ``q_hat`` is ``(n_actions, n_positions)`` or ``(n, n_actions, n_positions)``. For large
    contextual problems compute the two vectors directly and skip this helper.
    """

    pi = np.asarray(action_dist, dtype=np.float64)
    q = np.asarray(q_hat, dtype=np.float64)
    _check_policy(log, pi)
    _check_policy(log, np.abs(q))
    if pi.ndim == 2 and q.ndim == 2:
        # Context-free policy and model: one value per position, no (n, n_actions) temporaries.
        tot = pi.sum(axis=0)
        if np.any(tot[np.unique(log.position)] <= 0):
            raise ValueError("action_dist sums to zero at a logged position")
        per_pos = (pi * q).sum(axis=0) / np.where(tot > 0, tot, 1.0)
        return per_pos[log.position], q[log.action, log.position]
    q_pi = np.empty(len(log))
    q_f = np.empty(len(log))
    step = max(1, 4_000_000 // max(log.n_actions, 1))
    for start in range(0, len(log), step):
        idx = np.arange(start, min(start + step, len(log)))
        k = log.position[idx]
        pi_k = pi[:, k].T if pi.ndim == 2 else pi[idx, :, k]
        q_k = q[:, k].T if q.ndim == 2 else q[idx, :, k]
        tot = pi_k.sum(axis=1)
        if np.any(tot <= 0):
            raise ValueError("action_dist sums to zero at a logged position")
        q_pi[idx] = (pi_k * q_k).sum(axis=1) / tot
        a = log.action[idx]
        q_f[idx] = q[a, k] if q.ndim == 2 else q[idx, a, k]
    return q_pi, q_f


@dataclass(frozen=True)
class RoundTerms:
    """Per-round terms such that the estimate is ``mean(a) + sum(b) / sum(c)``."""

    a: F64
    b: F64
    c: F64

    def value(self, weights: F64 | None = None) -> float:
        if weights is None:
            ratio = float(self.b.sum() / self.c.sum()) if self.c.sum() > 0 else 0.0
            return float(self.a.mean()) + ratio
        wa = float(weights @ self.a) / float(weights.sum())
        wc = float(weights @ self.c)
        return wa + (float(weights @ self.b) / wc if wc > 0 else 0.0)


def round_terms(
    estimator: Estimator,
    reward: F64,
    iw: F64,
    q_pi: F64 | None = None,
    q_factual: F64 | None = None,
    tau: float = np.inf,
    clip: float = np.inf,
) -> RoundTerms:
    """Per-round terms of ``estimator``; model-based ones need ``q_pi`` and ``q_factual``.

    ``clip`` caps importance weights for IPS and DR (obp's ``lambda_``); ``tau`` is the Switch-DR
    threshold.
    """

    if estimator not in ESTIMATORS:
        raise ValueError(f"unknown estimator {estimator!r}; expected one of {ESTIMATORS}")
    if estimator in NEEDS_MODEL and (q_pi is None or q_factual is None):
        raise ValueError(f"{estimator} needs a reward model (q_pi and q_factual)")
    if not clip > 0 or not tau >= 0:
        raise ValueError("clip must be positive and tau non-negative")
    zero = np.zeros_like(reward)
    one = np.ones_like(reward)
    if estimator == "ips":
        return RoundTerms(np.minimum(iw, clip) * reward, zero, one)
    if estimator == "snips":
        return RoundTerms(zero, iw * reward, iw)
    assert q_pi is not None and q_factual is not None
    if estimator == "dm":
        return RoundTerms(q_pi, zero, one)
    if estimator == "dr":
        return RoundTerms(q_pi + np.minimum(iw, clip) * (reward - q_factual), zero, one)
    if estimator == "sndr":
        return RoundTerms(q_pi, iw * (reward - q_factual), iw)
    keep = (iw <= tau).astype(np.float64)
    return RoundTerms(q_pi + keep * iw * (reward - q_factual), zero, one)


def estimate(
    estimator: Estimator,
    log: BanditLog,
    action_dist: ArrayLike,
    q_hat: ArrayLike | None = None,
    tau: float = np.inf,
    clip: float = np.inf,
) -> float:
    """Policy value estimate of ``action_dist`` from ``log``."""

    iw = importance_weights(log, action_dist)
    q_pi = q_f = None
    if q_hat is not None:
        q_pi, q_f = model_terms(log, action_dist, q_hat)
    return round_terms(estimator, log.reward, iw, q_pi, q_f, tau, clip).value()


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float
    std: float

    def to_dict(self) -> dict[str, float]:
        return {"estimate": self.estimate, "low": self.low, "high": self.high, "std": self.std}


def bootstrap(
    terms: RoundTerms,
    n_boot: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
    chunk: int = 8,
) -> tuple[Interval, F64]:
    """Poisson bootstrap percentile interval for an estimator given its per-round terms.

    Each replicate weights round ``i`` by an independent Poisson(1) count (Hanley and MacGibbon,
    2006; the standard large-scale approximation of resampling with replacement). Returns the
    interval and the replicate estimates. At most about 30 million counts are held at a time.
    """

    if n_boot < 2 or not 0 < alpha < 1:
        raise ValueError("n_boot must be at least 2 and alpha in (0, 1)")
    rng = np.random.default_rng(seed)
    n = terms.a.size
    reps = np.empty(n_boot)
    M = np.stack([terms.a, terms.b, terms.c, np.ones(n)]).T  # (n, 4)
    chunk = max(1, min(chunk, 30_000_000 // max(n, 1)))
    done = 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        W = rng.poisson(1.0, size=(m, n)).astype(np.float64)
        S = W @ M
        ratio = np.divide(S[:, 1], S[:, 2], out=np.zeros(m), where=S[:, 2] > 0)
        reps[done : done + m] = S[:, 0] / S[:, 3] + ratio
        done += m
    lo, hi = np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(terms.value(), float(lo), float(hi), float(reps.std(ddof=1))), reps


def relative_error(estimate_: float, truth: float) -> float:
    """``|estimate - truth| / |truth|``, the metric of the Open Bandit Dataset paper."""

    if truth == 0:
        raise ValueError("truth must be non-zero")
    return abs(estimate_ - truth) / abs(truth)


def marginal_action_dist(log: BanditLog) -> F64:
    """Context-free action distribution of the behavior policy, ``(n_actions, n_positions)``.

    The mean logged propensity of each (item, position) pair over the rounds where it was shown;
    this is how a static summary of a non-contextual policy (such as Bernoulli Thompson sampling
    on the Open Bandit Dataset) is recovered from its own log. Pairs never shown get 0, and each
    position is renormalized to sum to 1.
    """

    A, K = log.n_actions, log.n_positions
    flat = log.action * K + log.position
    s = np.bincount(flat, weights=log.pscore, minlength=A * K)
    c = np.bincount(flat, minlength=A * K)
    pi = np.divide(s, c, out=np.zeros(A * K), where=c > 0).reshape(A, K)
    tot = pi.sum(axis=0, keepdims=True)
    out: F64 = np.divide(pi, tot, out=np.zeros_like(pi), where=tot > 0)
    return out


def empirical_action_dist(log: BanditLog) -> F64:
    """Frequency of each (item, position) pair among the logged rounds at that position."""

    A, K = log.n_actions, log.n_positions
    c = np.bincount(log.action * K + log.position, minlength=A * K).reshape(A, K).astype(float)
    tot = c.sum(axis=0, keepdims=True)
    out: F64 = np.divide(c, tot, out=np.zeros_like(c), where=tot > 0)
    return out


def uniform_action_dist(n_actions: int, n_positions: int) -> F64:
    """The uniform random ranking: each item at each position with probability ``1 / n_actions``."""

    if n_actions < 1 or n_positions < 1:
        raise ValueError("n_actions and n_positions must be positive")
    return np.full((n_actions, n_positions), 1.0 / n_actions)


def item_position_model(
    log: BanditLog, rows: NDArray[np.integer] | None = None, prior: float = 10.0
) -> F64:
    """Context-free reward model ``q(a, k)``: click rate of each (item, position) pair.

    Smoothed towards the position's click rate with ``prior`` pseudo-rounds (empirical Bayes
    shrinkage), so rarely shown pairs do not get extreme estimates.
    """

    sub = log if rows is None else log.take(rows)
    A, K = sub.n_actions, sub.n_positions
    flat = sub.action * K + sub.position
    clicks = np.bincount(flat, weights=sub.reward, minlength=A * K).reshape(A, K)
    shown = np.bincount(flat, minlength=A * K).reshape(A, K).astype(float)
    pos_rate = np.divide(
        clicks.sum(axis=0), shown.sum(axis=0), out=np.zeros(K), where=shown.sum(axis=0) > 0
    )
    den = shown + prior
    num = clicks + prior * pos_rate[None, :]
    out: F64 = np.divide(num, den, out=np.broadcast_to(pos_rate, num.shape).copy(), where=den > 0)
    return out
