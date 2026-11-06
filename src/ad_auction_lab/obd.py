"""The Open Bandit Dataset (Saito et al., 2021) as bandit logs, and its contextual reward model.

``scripts/prepare_obd.py`` converts the full dataset (ZOZOTOWN fashion recommendations, 7 days,
three campaigns, logged by a uniform random policy and by Bernoulli Thompson sampling) into
``.npz`` files. Each round shows an item (action) in one of three positions; the reward is a
click. The random logs keep the user context: four hashed user features and the user-item
affinity counts.

The direct method and the doubly robust estimators need ``q(x, a, k)``, the click probability of
item ``a`` at position ``k`` for user ``x``. :func:`reward_table` turns rounds into a
:class:`~ad_auction_lab.data.Table` for the hashed CTR models of :mod:`ad_auction_lab.ctr`
(user features, item, item features, position, the user's affinity for the item and
user-by-item crosses), and :func:`cross_fit_model_terms` fits the model on ``K - 1`` folds and
predicts the held-out fold for every candidate item, so no round is scored by a model that saw it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ad_auction_lab.data import Table
from ad_auction_lab.ope import BanditLog

POLICIES = ("random", "bts")
CAMPAIGNS = ("all", "men", "women")
N_POSITIONS = 3


@dataclass(frozen=True)
class ObdLog:
    log: BanditLog
    user: NDArray[np.uint32] | None
    affinity: NDArray[np.uint8] | None
    item_features: NDArray[np.uint32]
    item_price_bucket: NDArray[np.uint8]

    @property
    def has_context(self) -> bool:
        return self.user is not None and self.affinity is not None


def load_obd(directory: str | Path, policy: str, campaign: str) -> ObdLog:
    """Load ``<policy>_<campaign>.npz`` and the item features written by ``prepare_obd.py``."""

    if policy not in POLICIES or campaign not in CAMPAIGNS:
        raise ValueError(f"policy must be one of {POLICIES} and campaign one of {CAMPAIGNS}")
    d = Path(directory)
    with np.load(d / f"{policy}_{campaign}.npz") as f:
        arrays = {k: f[k] for k in f.files}
    with np.load(d / f"item_{campaign}.npz") as f:
        item_ids = f["item_id"]
        feats = f["features"].astype(np.uint32)
        price = f["feature_0"]
    n_items = int(item_ids.max()) + 1
    item_features = np.zeros((n_items, feats.shape[1]), dtype=np.uint32)
    item_features[item_ids] = feats
    ranks = np.searchsorted(np.unique(price), price)
    bucket = np.zeros(n_items, dtype=np.uint8)
    bucket[item_ids] = np.minimum(ranks, 255).astype(np.uint8)
    log = BanditLog.from_arrays(
        arrays["click"],
        arrays["item"],
        arrays["pscore"],
        arrays["position"],
        n_actions=n_items,
        n_positions=N_POSITIONS,
    )
    return ObdLog(log, arrays.get("user"), arrays.get("affinity"), item_features, bucket)


def _mix(a: NDArray[np.uint32], b: NDArray[np.integer]) -> NDArray[np.uint32]:
    x = a.astype(np.uint64) * np.uint64(0x9E3779B1) + b.astype(np.uint64) * np.uint64(0x85EBCA77)
    x ^= x >> np.uint64(29)
    out: NDArray[np.uint32] = (x & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    return out


def reward_table(
    obd: ObdLog,
    rows: NDArray[np.integer],
    action: NDArray[np.integer] | int | None = None,
) -> Table:
    """Features of ``rows`` with their logged item, or with ``action`` substituted.

    ``small`` (uint8): position, item, the user's affinity for the item, item price bucket.
    ``wide`` (uint32): 4 user features, 3 item features, 4 user-by-item crosses.
    """

    if not obd.has_context:
        raise ValueError("this log has no user context (only the random logs keep it)")
    assert obd.user is not None and obd.affinity is not None
    r = np.asarray(rows, dtype=np.int64)
    log = obd.log
    a = log.action[r] if action is None else np.broadcast_to(np.asarray(action), r.shape)
    a = np.asarray(a, dtype=np.int64)
    user = obd.user[r]
    small = np.stack(
        [
            log.position[r].astype(np.uint8),
            a.astype(np.uint8),
            obd.affinity[r, a],
            obd.item_price_bucket[a],
        ],
        axis=1,
    )
    crosses = [_mix(user[:, j], a) for j in range(user.shape[1])]
    wide = np.concatenate([user, obd.item_features[a], np.stack(crosses, axis=1)], axis=1)
    labels = log.reward[r].astype(np.uint8)
    return Table(np.ascontiguousarray(small), np.ascontiguousarray(wide, dtype=np.uint32), labels)


class RewardModel(Protocol):
    """Anything with the ``fit`` and ``predict_proba`` of the hashed CTR models."""

    def fit(self, table: Table, *, epochs: int = ..., seed: int = ...) -> object: ...

    def predict_proba(self, table: Table) -> NDArray[np.float32]: ...


def cross_fit_model_terms(
    obd: ObdLog,
    action_dist: NDArray[np.float64],
    make_model: Callable[[], RewardModel],
    n_folds: int = 2,
    epochs: int = 1,
    seed: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Cross-fitted ``(q(x_i, pi, k_i), q(x_i, a_i, k_i))`` from a hashed CTR model.

    Rounds are split into ``n_folds`` random folds; for each fold a fresh model is trained on the
    others and predicts every candidate item at the logged position of each held-out round.
    ``action_dist`` is the context-free ``(n_actions, n_positions)`` evaluation policy.
    """

    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    log = obd.log
    n = len(log)
    pi = np.asarray(action_dist, dtype=np.float64)
    if pi.shape != (log.n_actions, log.n_positions):
        raise ValueError("action_dist must be (n_actions, n_positions)")
    fold = np.random.default_rng(seed).integers(0, n_folds, n)
    q_pi = np.zeros(n)
    q_f = np.zeros(n)
    for f in range(n_folds):
        train = np.flatnonzero(fold != f)
        test = np.flatnonzero(fold == f)
        model = make_model()
        model.fit(reward_table(obd, train), epochs=epochs, seed=seed + f)
        q_f[test] = model.predict_proba(reward_table(obd, test))
        k = log.position[test]
        tot = pi[:, k].sum(axis=0)
        acc = np.zeros(test.size)
        for a in range(log.n_actions):
            weight = pi[a, k]
            if not np.any(weight > 0):
                continue
            acc += weight * model.predict_proba(reward_table(obd, test, a))
        q_pi[test] = acc / tot
    return q_pi, q_f
