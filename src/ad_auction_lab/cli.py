"""Command line interface: ``python -m ad_auction_lab <command>``.

Commands
--------
``auction``     run one position auction and print the slots, prices and revenue.
``incentives``  measure the best gain from misreporting under each mechanism.
``shade``       first-price bid shading against the Criteo attribution market prices.
``pace``        budget pacing in a simulated market built from real prices and CTRs.
``train-ctr``   train FTRL or a factorization machine on Criteo and report test metrics.
``ope``         off-policy evaluation on the Open Bandit Dataset against the on-policy truth.
``serve``       start the HTTP service.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ad_auction_lab import __version__
from ad_auction_lab.auctions import MECHANISMS, deviation_gain, run_auctions

DEFAULT_DATA = Path("../.bench-data/ad-auction-lab")


def _floats(text: str) -> list[float]:
    try:
        return [float(x) for x in text.split(",") if x.strip()]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"expected comma-separated numbers, got {text!r}") from e


def _print(obj: object, as_json: bool) -> None:
    if as_json:
        print(json.dumps(obj, indent=2))


def cmd_auction(a: argparse.Namespace) -> int:
    q = a.quality or [1.0] * len(a.bids)
    if len(q) != len(a.bids):
        print("error: --quality needs one value per bid", file=sys.stderr)
        return 2
    rows = []
    for mech in MECHANISMS if a.mechanism == "all" else [a.mechanism]:
        r = run_auctions(a.bids, q, a.slots, mech, a.reserve, a.squash)
        rows.append(
            {
                "mechanism": mech,
                "winners": r.winner[0].tolist(),
                "price_per_click": [round(float(p), 6) for p in r.price[0]],
                "revenue": float(r.revenue()[0]),
            }
        )
    if a.json:
        _print(rows, True)
        return 0
    print(f"bids {a.bids}  quality {q}  slot CTRs {a.slots}  reserve {a.reserve}")
    print(f"{'mechanism':<12} {'slot':>4} {'bidder':>6} {'price/click':>12}")
    for row in rows:
        for j, (w, p) in enumerate(zip(row["winners"], row["price_per_click"], strict=True)):
            who = "-" if w < 0 else str(w)
            print(f"{row['mechanism']:<12} {j:>4} {who:>6} {p:>12.4f}")
        print(f"{'':<12} expected revenue per impression {row['revenue']:.4f}")
    return 0


def cmd_incentives(a: argparse.Namespace) -> int:
    rng = np.random.default_rng(a.seed)
    v = rng.lognormal(0, 0.5, (a.auctions, a.bidders))
    q = rng.beta(2, 30, (a.auctions, a.bidders))
    out = {}
    for mech in MECHANISMS:
        gains = np.stack(
            [deviation_gain(v, q, a.slots, mech, i, reserve=a.reserve) for i in range(a.bidders)]
        )
        welfare = run_auctions(v, q, a.slots, mech, a.reserve).welfare(v)
        out[mech] = {
            "share_with_profitable_deviation": float(np.mean(gains > 1e-12)),
            "max_gain": float(np.max(gains)),
            "total_gain_over_welfare": float(np.sum(gains) / max(float(np.sum(welfare)), 1e-300)),
        }
    if a.json:
        _print(out, True)
        return 0
    print(f"{a.auctions} auctions, {a.bidders} bidders, slot CTRs {a.slots}, others truthful")
    print(f"{'mechanism':<12} {'profitable deviation':>21} {'max gain':>10} {'gain/welfare':>13}")
    for mech, s in out.items():
        print(
            f"{mech:<12} {s['share_with_profitable_deviation']:>20.1%} "
            f"{s['max_gain']:>10.2e} {s['total_gain_over_welfare']:>12.1%}"
        )
    return 0


def _attribution_prices(data: Path) -> np.ndarray:
    f = data / "attribution" / "cost.npy"
    if not f.exists():
        raise SystemExit(f"error: {f} not found; run `make data` first")
    out: np.ndarray = np.load(f, mmap_mode="r")
    return out


def cmd_shade(a: argparse.Namespace) -> int:
    from ad_auction_lab.bidding import EmpiricalLandscape, LogNormalLandscape

    prices = np.asarray(_attribution_prices(a.data), dtype=np.float64)
    rng = np.random.default_rng(a.seed)
    sample = prices[rng.integers(0, prices.size, a.sample)]
    emp = EmpiricalLandscape.fit(sample)
    logn = LogNormalLandscape.fit(sample)
    med = float(np.median(prices))
    values = np.asarray(a.values or [0.5 * med, med, 2 * med, 5 * med, 20 * med])
    b_emp, b_ln = emp.shade(values), logn.shade(values)
    print(f"market prices: {prices.size:,} Criteo impressions, median {med:.3e}")
    print(f"{'value':>10} {'bid (empirical)':>16} {'win rate':>9} {'bid (log-normal)':>17}")
    for v, be, bl in zip(values, b_emp, b_ln, strict=True):
        wr = float(emp.win_rate([be])[0])
        print(f"{v:>10.3e} {be:>16.3e} {wr:>9.1%} {bl:>17.3e}")
    return 0


def cmd_pace(a: argparse.Namespace) -> int:
    from ad_auction_lab.pacing import Pacer, build_market, simulate

    prices = _attribution_prices(a.data)
    pool_file = a.data / "criteo_test_ctr.npy"
    pool = np.load(pool_file) if pool_file.exists() else np.random.default_rng(1).beta(2, 60, 10**6)
    m = build_market(prices, pool, a.horizon, a.advertisers, a.budget_share, seed=a.seed)
    pacers = {
        "none": Pacer("none"),
        "pid": Pacer("pid", kp=a.kp, ki=a.ki, interval=a.interval),
        "dual": Pacer("dual", step=a.step),
    }
    print(
        f"{a.horizon:,} auctions, {a.advertisers} advertisers, budgets {a.budget_share:.0%} "
        "of unconstrained spend"
    )
    print(f"{'pacer':<6} {'value':>10} {'spend':>10} {'used':>6} {'lasts':>6} {'track err':>10}")
    for name, p in pacers.items():
        t0 = time.perf_counter()
        s = simulate(m.values, m.budgets, m.ctr, m.market, p).summary()
        ms = (time.perf_counter() - t0) * 1e3
        print(
            f"{name:<6} {s['value']:>10.4f} {s['spend']:>10.4f} {s['utilization']:>6.1%} "
            f"{s['mean_exhaustion_time']:>6.0%} {s['max_tracking_error']:>10.3f}  ({ms:.0f} ms)"
        )
    return 0


def cmd_train_ctr(a: argparse.Namespace) -> int:
    from ad_auction_lab.ctr import FactorizationMachine, FTRLProximal
    from ad_auction_lab.data import load_criteo, split_rows
    from ad_auction_lab.metrics import summary

    t = load_criteo(a.data / "criteo")
    train, _, test = split_rows(len(t), seed=0)
    if a.train_rows:
        train = train[: a.train_rows]
    test = test[: a.test_rows] if a.test_rows else test
    model = (
        FTRLProximal(alpha=a.alpha, l1=a.l1, bits=a.bits)
        if a.model == "ftrl"
        else FactorizationMachine(k=a.k, bits=a.bits, lr=a.lr)
    )
    print(f"training {a.model} on {len(train):,} Criteo rows, {a.epochs} epoch(s)", flush=True)
    model.fit(t, train, epochs=a.epochs, seed=a.seed, verbose=True)
    p = model.predict_proba(t, test)
    s = summary(t.labels[test], p)
    print(
        f"test rows {len(test):,}: logloss {s['logloss']:.5f}  AUC {s['auc']:.5f}  "
        f"NE {s['normalized_entropy']:.4f}  calibration {s['calibration_ratio']:.4f}"
    )
    if a.out:
        model.save(a.out)
        print(f"saved {a.out}")
    return 0


def cmd_ope(a: argparse.Namespace) -> int:
    from ad_auction_lab.obd import load_obd
    from ad_auction_lab.ope import (
        ESTIMATORS,
        bootstrap,
        importance_weights,
        item_position_model,
        marginal_action_dist,
        model_terms,
        relative_error,
        round_terms,
        uniform_action_dist,
    )

    d = a.data / "obd"
    behavior = "random" if a.target == "bts" else "bts"
    log = load_obd(d, behavior, a.campaign).log
    target_log = load_obd(d, a.target, a.campaign).log
    truth = float(target_log.reward.mean())
    pi = (
        marginal_action_dist(target_log)
        if a.target == "bts"
        else uniform_action_dist(log.n_actions, log.n_positions)
    )
    iw = importance_weights(log, pi)
    q_pi, q_f = model_terms(log, pi, item_position_model(log))
    tau = float(np.quantile(iw, 0.99))
    print(
        f"estimate the {a.target} policy from {len(log):,} {behavior} rounds "
        f"({a.campaign}); on-policy truth {truth:.6f} from {len(target_log):,} rounds"
    )
    print(f"{'estimator':<10} {'estimate':>10} {'95% CI':>23} {'rel. error':>11}")
    for name in ESTIMATORS:
        t = round_terms(name, log.reward, iw, q_pi, q_f, tau=tau)
        ci, _ = bootstrap(t, n_boot=a.boot, seed=a.seed)
        print(
            f"{name:<10} {ci.estimate:>10.6f}  [{ci.low:.6f}, {ci.high:.6f}] "
            f"{relative_error(ci.estimate, truth):>10.2%}"
        )
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    from ad_auction_lab.ctr import load_model
    from ad_auction_lab.service import AuctionService, serve

    model = load_model(a.model) if a.model else None
    server = serve(AuctionService(model=model), a.host, a.port)
    print(f"serving on http://{a.host}:{server.server_address[1]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - interactive
        pass
    finally:
        server.server_close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ad_auction_lab", description=__doc__.splitlines()[0])
    p.add_argument("--version", action="version", version=f"ad-auction-lab {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("auction", help="run one position auction")
    s.add_argument("--bids", type=_floats, required=True)
    s.add_argument("--quality", type=_floats, default=None)
    s.add_argument("--slots", type=_floats, default=[1.0])
    s.add_argument("--mechanism", choices=[*MECHANISMS, "all"], default="all")
    s.add_argument("--reserve", type=float, default=0.0)
    s.add_argument("--squash", type=float, default=1.0)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_auction)

    s = sub.add_parser("incentives", help="profitable deviations per mechanism")
    s.add_argument("--auctions", type=int, default=2000)
    s.add_argument("--bidders", type=int, default=5)
    s.add_argument("--slots", type=_floats, default=[0.3, 0.2, 0.1])
    s.add_argument("--reserve", type=float, default=0.0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_incentives)

    s = sub.add_parser("shade", help="first-price bid shading on Criteo market prices")
    s.add_argument("--data", type=Path, default=DEFAULT_DATA)
    s.add_argument("--values", type=_floats, default=None)
    s.add_argument("--sample", type=int, default=1_000_000)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_shade)

    s = sub.add_parser("pace", help="budget pacing on a market built from real data")
    s.add_argument("--data", type=Path, default=DEFAULT_DATA)
    s.add_argument("--horizon", type=int, default=1_000_000)
    s.add_argument("--advertisers", type=int, default=10)
    s.add_argument("--budget-share", type=float, default=0.3)
    s.add_argument("--kp", type=float, default=2.0)
    s.add_argument("--ki", type=float, default=0.05)
    s.add_argument("--interval", type=int, default=1000)
    s.add_argument("--step", type=float, default=0.002)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_pace)

    s = sub.add_parser("train-ctr", help="train a hashed CTR model on Criteo")
    s.add_argument("--data", type=Path, default=DEFAULT_DATA)
    s.add_argument("--model", choices=["ftrl", "fm"], default="ftrl")
    s.add_argument("--train-rows", type=int, default=0, help="0 means the full training split")
    s.add_argument("--test-rows", type=int, default=0, help="0 means the full test split")
    s.add_argument("--epochs", type=int, default=1)
    s.add_argument("--bits", type=int, default=22)
    s.add_argument("--alpha", type=float, default=0.05)
    s.add_argument("--l1", type=float, default=1.0)
    s.add_argument("--k", type=int, default=16)
    s.add_argument("--lr", type=float, default=0.05)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", type=Path, default=None)
    s.set_defaults(func=cmd_train_ctr)

    s = sub.add_parser("ope", help="off-policy evaluation on the Open Bandit Dataset")
    s.add_argument("--data", type=Path, default=DEFAULT_DATA)
    s.add_argument("--campaign", choices=["all", "men", "women"], default="all")
    s.add_argument("--target", choices=["bts", "random"], default="bts")
    s.add_argument("--boot", type=int, default=100)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_ope)

    s = sub.add_parser("serve", help="start the HTTP service")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--model", type=Path, default=None, help="a model saved by train-ctr --out")
    s.set_defaults(func=cmd_serve)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code: int = args.func(args)
    return code
