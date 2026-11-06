"""Auction incentives, reserve prices, bid shading and budget pacing on real market data.

Market data: the Criteo Attribution Modeling for Bidding dataset (16,468,027 impressions won by
Criteo over 31 days, each with the price paid). Quality scores (predicted CTRs) come from the
factorization machine's predictions on the Criteo test split (``ctr_criteo.py final`` writes them
to ``criteo_test_ctr.npy``).

    uv run python benchmarks/markets.py incentives   # profitable deviations per mechanism
    uv run python benchmarks/markets.py reserves     # Myerson and empirical reserves
    uv run python benchmarks/markets.py shading      # first-price bid shading, time split
    uv run python benchmarks/markets.py pacing       # PID and dual pacing, 5 seeds
    uv run python benchmarks/markets.py speed        # Rust simulator against the reference
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, write

from ad_auction_lab.auctions import (
    MECHANISMS,
    deviation_gain,
    empirical_reserve,
    gsp_envy_free_bids,
    myerson_reserve_lognormal,
    run_auctions,
    vcg_bruteforce,
)
from ad_auction_lab.bidding import EmpiricalLandscape, LogNormalLandscape, shading_report
from ad_auction_lab.pacing import (
    Pacer,
    build_market,
    hindsight_optimum,
    simulate,
    simulate_reference,
)

ATTR = DATA / "attribution"
SLOTS = {1: [1.0], 3: [1.0, 0.6, 0.35], 5: [1.0, 0.7, 0.5, 0.35, 0.25]}
PACERS = {
    "none": Pacer("none"),
    "pid": Pacer("pid", kp=2.0, ki=0.05, interval=1000),
    "dual": Pacer("dual", step=0.002),
}


def prices() -> np.ndarray:
    out: np.ndarray = np.load(ATTR / "cost.npy", mmap_mode="r")
    return out


def ctr_pool() -> np.ndarray:
    f = DATA / "criteo_test_ctr.npy"
    if not f.exists():
        raise SystemExit("run `benchmarks/ctr_criteo.py final` first (writes the CTR pool)")
    out: np.ndarray = np.load(f)
    return out


def value_sample(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """Per-click values: resampled positive market prices divided by resampled CTRs.

    An advertiser who just breaks even at the observed eCPM ``price`` with CTR ``c`` values a
    click at ``price / c``; resampling both gives a heavy-tailed, data-driven value distribution.
    """

    p = prices()
    pos = p[p > 0]
    pool = ctr_pool()
    v = pos[rng.integers(0, pos.size, shape)] / pool[rng.integers(0, pool.size, shape)]
    out: np.ndarray = v / np.median(v)
    return out


def cmd_incentives(a: argparse.Namespace) -> None:
    rng = np.random.default_rng(0)
    pool = ctr_pool()
    rows = []
    for n in (3, 5, 10):
        for k, alpha in SLOTS.items():
            if k >= n:
                continue
            v = value_sample(rng, (a.auctions, n))
            q = pool[rng.integers(0, pool.size, (a.auctions, n))].astype(np.float64)
            r_star = empirical_reserve(v.reshape(-1))[0]
            for mech in MECHANISMS:
                for reserve in (0.0, r_star):
                    for squash in (1.0, 0.5):
                        t0 = time.perf_counter()
                        gains = np.stack(
                            [
                                deviation_gain(v, q, alpha, mech, i, reserve=reserve, squash=squash)
                                for i in range(n)
                            ],
                            axis=1,
                        )
                        res = run_auctions(v, q, alpha, mech, reserve, squash)
                        welfare = res.welfare(v)
                        util = res.utilities(v)
                        rows.append(
                            {
                                "bidders": n,
                                "slots": k,
                                "mechanism": mech,
                                "reserve": reserve,
                                "squash": squash,
                                "auctions": a.auctions,
                                "share_profitable": float(np.mean(gains > 1e-9 * (1 + util))),
                                "max_gain": float(gains.max()),
                                "gain_over_welfare": float(gains.sum() / welfare.sum()),
                                "truthful_revenue": float(res.revenue().mean()),
                                "truthful_welfare": float(welfare.mean()),
                                "seconds": time.perf_counter() - t0,
                            }
                        )
                        print(rows[-1], flush=True)
    # EOS: GSP at the lowest envy-free equilibrium has VCG's allocation and revenue.
    eos = []
    for n, k in ((5, 3), (10, 5)):
        alpha = SLOTS[k]
        v = value_sample(rng, (a.equilibria, n))
        q = pool[rng.integers(0, pool.size, (a.equilibria, n))].astype(np.float64)
        b = np.stack([gsp_envy_free_bids(v[t], q[t], alpha) for t in range(a.equilibria)])
        gsp = run_auctions(b, q, alpha, "gsp")
        vcg = run_auctions(v, q, alpha, "vcg")
        truthful_gsp = run_auctions(v, q, alpha, "gsp")
        rel = np.abs(gsp.revenue() - vcg.revenue()) / np.maximum(vcg.revenue(), 1e-300)
        eos.append(
            {
                "bidders": n,
                "slots": k,
                "auctions": a.equilibria,
                "same_allocation_share": float(np.mean(np.all(gsp.winner == vcg.winner, axis=1))),
                "max_relative_revenue_difference": float(rel.max()),
                "vcg_revenue": float(vcg.revenue().mean()),
                "gsp_equilibrium_revenue": float(gsp.revenue().mean()),
                "gsp_truthful_revenue": float(truthful_gsp.revenue().mean()),
            }
        )
    # VCG closed form against exhaustive search with Clarke payments.
    worst, checked = 0.0, 0
    for n, k in ((4, 3), (6, 3), (6, 5)):
        alpha = SLOTS[k]
        for _ in range(a.bruteforce):
            v = value_sample(rng, (n,))
            q = pool[rng.integers(0, pool.size, n)].astype(np.float64)
            for reserve, squash in ((0.0, 1.0), (0.5, 0.5)):
                w, p = vcg_bruteforce(v, q, alpha, reserve, squash)
                r = run_auctions(v, q, alpha, "vcg", reserve, squash)
                assert np.array_equal(w, r.winner[0])
                worst = max(worst, float(np.max(np.abs(p - r.price[0]))))
                checked += 1
    write(
        "auction_incentives",
        {
            "deviation_factors": "0, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 1.05, 1.1, 1.25, 1.5, 2, 4",
            "slot_ctr": SLOTS,
            "rows": rows,
            "gsp_envy_free_equilibrium": eos,
            "vcg_bruteforce": {"auctions": checked, "max_abs_price_difference": worst},
        },
    )


def cmd_reserves(a: argparse.Namespace) -> None:
    """Second-price auctions with i.i.d. values: revenue with and without a reserve."""

    rng = np.random.default_rng(1)
    rows = []
    for n in (1, 2, 3, 5, 10):
        train = value_sample(rng, (a.train,))
        logv = np.log(train)
        mu, sigma = float(logv.mean()), float(logv.std())
        reserves = {
            "none": 0.0,
            "myerson (log-normal fit)": myerson_reserve_lognormal(mu, sigma),
            "empirical (train sample)": empirical_reserve(train)[0],
        }
        test = value_sample(rng, (a.auctions, n))
        ones = np.ones_like(test)
        for name, r in reserves.items():
            res = run_auctions(test, ones, [1.0], "gsp", reserve=r)
            rev = res.revenue()
            rows.append(
                {
                    "bidders": n,
                    "reserve_rule": name,
                    "reserve": r,
                    "revenue_mean": float(rev.mean()),
                    "revenue_se": float(rev.std(ddof=1) / np.sqrt(rev.size)),
                    "welfare_mean": float(res.welfare(test).mean()),
                    "sold_share": float(np.mean(res.winner[:, 0] >= 0)),
                }
            )
            print(rows[-1], flush=True)
    write("auction_reserves", {"train_values": a.train, "test_auctions": a.auctions, "rows": rows})


def cmd_shading(a: argparse.Namespace) -> None:
    """Fit landscapes on days 1 to 21, evaluate surplus on days 22 to 31."""

    p = np.asarray(prices(), dtype=np.float64)
    ts = np.load(ATTR / "timestamp.npy", mmap_mode="r")
    camp = np.load(ATTR / "campaign.npy", mmap_mode="r")
    cut = int(np.searchsorted(ts, 21 * 86400))
    train, test = p[:cut], p[cut:]
    rng = np.random.default_rng(0)
    out: dict[str, Any] = {"train_rows": int(cut), "test_rows": int(p.size - cut)}
    t0 = time.perf_counter()
    emp = EmpiricalLandscape.fit(train)
    fit_s = time.perf_counter() - t0
    logn = LogNormalLandscape.fit(train[train > 0])
    oracle = EmpiricalLandscape.fit(test)
    med = float(np.median(train))
    sub = test[rng.integers(0, test.size, a.eval)]
    rows = []
    for mult in (0.5, 1.0, 2.0, 5.0, 20.0):
        v = np.full(sub.size, mult * med)
        bids = {
            "truthful": v,
            "empirical landscape": emp.shade(v),
            "log-normal landscape": logn.shade(v),
            "oracle (test landscape)": oracle.shade(v),
        }
        rep = shading_report(v, sub, bids)
        best = rep["oracle (test landscape)"]["surplus"]
        for name, r in rep.items():
            rows.append(
                {
                    "value_over_median_price": mult,
                    "rule": name,
                    "bid": float(np.asarray(bids[name])[0]),
                    **r,
                    "surplus_share_of_oracle": r["surplus"] / best if best > 0 else 0.0,
                }
            )
        print(rows[-5:], flush=True)
    out["global"] = rows
    t0 = time.perf_counter()
    big = emp.shade(value_sample(rng, (1_000_000,)) * med)
    out["shade_1m_values_seconds"] = time.perf_counter() - t0
    out["landscape_fit_seconds"] = fit_s
    out["shaded_mean"] = float(big.mean())
    # Per-campaign landscapes for the 20 campaigns with the most impressions.
    c_train, c_test = np.asarray(camp[:cut]), np.asarray(camp[cut:])
    ids, counts = np.unique(c_train, return_counts=True)
    top = ids[np.argsort(-counts)[:20]]
    per = []
    for c in top:
        tr, te = train[c_train == c], test[c_test == c]
        if te.size < 1000:
            continue
        m = float(np.median(tr))
        v = np.full(te.size, 2.0 * m)
        own = EmpiricalLandscape.fit(tr).shade(v)
        pooled = emp.shade(v)
        rep = shading_report(v, te, {"campaign": own, "pooled": pooled, "truthful": v})
        orc = shading_report(v, te, {"oracle": EmpiricalLandscape.fit(te).shade(v)})["oracle"]
        per.append(
            {
                "campaign": int(c),
                "train_rows": int(tr.size),
                "test_rows": int(te.size),
                **{
                    f"{k}_surplus_share": r["surplus"] / orc["surplus"] if orc["surplus"] else 0
                    for k, r in rep.items()
                },
            }
        )
    out["per_campaign_value_2x_median"] = per
    write("bid_shading", out)


def cmd_pacing(a: argparse.Namespace) -> None:
    p = prices()
    pool = ctr_pool()
    rows = []
    for n in (5, 10, 20):
        for share in (0.1, 0.3, 0.6):
            for seed in range(a.seeds):
                m = build_market(p, pool, a.horizon, n, share, seed=seed)
                for name, pacer in PACERS.items():
                    t0 = time.perf_counter()
                    s = simulate(m.values, m.budgets, m.ctr, m.market, pacer).summary()
                    rows.append(
                        {
                            "advertisers": n,
                            "budget_share": share,
                            "seed": seed,
                            "pacer": name,
                            "seconds": time.perf_counter() - t0,
                            **s,
                        }
                    )
            print(n, share, rows[-1], flush=True)
    # One advertiser against the external market: compare with the hindsight LP optimum.
    single = []
    for share in (0.05, 0.1, 0.3, 0.6):
        for seed in range(a.seeds):
            m = build_market(p, pool, a.horizon, 1, share, seed=seed)
            v_t = m.values[0] * m.ctr[:, 0].astype(np.float64)
            # A second-price win costs the market eCPM; ties go to the market.
            opt = hindsight_optimum(v_t, m.market.astype(np.float64), float(m.budgets[0]))
            for name, pacer in PACERS.items():
                r = simulate(m.values, m.budgets, m.ctr, m.market, pacer)
                single.append(
                    {
                        "budget_share": share,
                        "seed": seed,
                        "pacer": name,
                        "value": float(r.value[0]),
                        "hindsight_value": opt["value"],
                        "competitive_ratio": float(r.value[0]) / opt["value"],
                        "utilization": float(r.spend[0] / m.budgets[0]),
                        "exhausted_at": float(r.exhausted_at[0] / a.horizon),
                    }
                )
        print(share, single[-3:], flush=True)
    write(
        "pacing",
        {
            "horizon": a.horizon,
            "pacers": {k: vars(v) for k, v in PACERS.items()},
            "rows": rows,
            "single_advertiser": single,
        },
    )


def cmd_speed(a: argparse.Namespace) -> None:
    p = prices()
    pool = ctr_pool()
    rows = []
    for n in (1, 10, 50):
        m = build_market(p, pool, a.horizon, n, 0.3, seed=0)
        for name, pacer in PACERS.items():
            t0 = time.perf_counter()
            r1 = simulate(m.values, m.budgets, m.ctr, m.market, pacer, backend="rust")
            rust = time.perf_counter() - t0
            t0 = time.perf_counter()
            r2 = simulate_reference(m.values, m.budgets, m.ctr, m.market, pacer)
            py = time.perf_counter() - t0
            rows.append(
                {
                    "advertisers": n,
                    "pacer": name,
                    "auctions": a.horizon,
                    "rust_auctions_per_s": a.horizon / rust,
                    "python_auctions_per_s": a.horizon / py,
                    "speedup": py / rust,
                    "max_abs_value_difference": float(np.max(np.abs(r1.value - r2.value))),
                }
            )
            print(rows[-1], flush=True)
    big = build_market(p, pool, 10_000_000, 10, 0.3, seed=0)
    t0 = time.perf_counter()
    simulate(big.values, big.budgets, big.ctr, big.market, PACERS["dual"])
    write(
        "pacing_speed",
        {"rows": rows, "rust_10m_auctions_10_advertisers_seconds": time.perf_counter() - t0},
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("incentives")
    s.add_argument("--auctions", type=int, default=20_000)
    s.add_argument("--equilibria", type=int, default=2_000)
    s.add_argument("--bruteforce", type=int, default=300)
    s.set_defaults(func=cmd_incentives)
    s = sub.add_parser("reserves")
    s.add_argument("--train", type=int, default=200_000)
    s.add_argument("--auctions", type=int, default=1_000_000)
    s.set_defaults(func=cmd_reserves)
    s = sub.add_parser("shading")
    s.add_argument("--eval", type=int, default=2_000_000)
    s.set_defaults(func=cmd_shading)
    s = sub.add_parser("pacing")
    s.add_argument("--horizon", type=int, default=1_000_000)
    s.add_argument("--seeds", type=int, default=5)
    s.set_defaults(func=cmd_pacing)
    s = sub.add_parser("speed")
    s.add_argument("--horizon", type=int, default=100_000)
    s.set_defaults(func=cmd_speed)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
