"""Off-policy evaluation on the full Open Bandit Dataset, against on-policy truth and obp.

For each campaign (all, men, women) the policy value of Bernoulli Thompson sampling (BTS) is
estimated from the uniform random log and, in the other direction, the value of the random policy
from the BTS log. The ground truth is the click rate the target policy actually achieved in its
own log (on-policy), as in Saito et al. (2021). Reward models: the smoothed (item, position) click
rate, and on the random logs (the only ones with user context) cross-fitted hashed FTRL and FM
models with user, item and user-by-item features.

    uv run python benchmarks/ope_obd.py main        # every campaign and direction, bootstrap CIs
    uv run python benchmarks/ope_obd.py curve       # relative error against log size
    uv run python benchmarks/ope_obd.py ablate      # Switch-DR threshold and IPS clipping
    uv run python benchmarks/ope_obd.py obp         # agreement with and speed against obp 0.5.7
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

from ad_auction_lab.ctr import FactorizationMachine, FTRLProximal
from ad_auction_lab.obd import CAMPAIGNS, cross_fit_model_terms, load_obd
from ad_auction_lab.ope import (
    ESTIMATORS,
    NEEDS_MODEL,
    BanditLog,
    bootstrap,
    importance_weights,
    item_position_model,
    marginal_action_dist,
    model_terms,
    relative_error,
    round_terms,
    uniform_action_dist,
)

OBD = DATA / "obd"
TAU_QUANTILE = 0.99


def setting(campaign: str, target: str) -> tuple[Any, BanditLog, np.ndarray, float, int]:
    behavior = "random" if target == "bts" else "bts"
    obd = load_obd(OBD, behavior, campaign)
    target_log = load_obd(OBD, target, campaign).log
    truth = float(target_log.reward.mean())
    pi = (
        marginal_action_dist(target_log)
        if target == "bts"
        else uniform_action_dist(obd.log.n_actions, obd.log.n_positions)
    )
    return obd, obd.log, pi, truth, len(target_log)


def evaluate(
    log: BanditLog,
    pi: np.ndarray,
    truth: float,
    models: dict[str, tuple[np.ndarray, np.ndarray] | None],
    n_boot: int,
    seed: int = 0,
) -> list[dict[str, Any]]:
    iw = importance_weights(log, pi)
    tau = float(np.quantile(iw, TAU_QUANTILE))
    rows = []
    for model, terms in models.items():
        for est in ESTIMATORS:
            if (est in NEEDS_MODEL) != (terms is not None):
                continue
            q_pi, q_f = terms if terms is not None else (None, None)
            t = round_terms(est, log.reward, iw, q_pi, q_f, tau=tau)
            ci, reps = bootstrap(t, n_boot=n_boot, seed=seed)
            re_reps = np.abs(reps - truth) / truth
            rows.append(
                {
                    "estimator": est,
                    "reward_model": model,
                    "estimate": ci.estimate,
                    "ci_low": ci.low,
                    "ci_high": ci.high,
                    "std": ci.std,
                    "relative_error": relative_error(ci.estimate, truth),
                    "bootstrap_relative_error_mean": float(re_reps.mean()),
                    "bootstrap_relative_error_std": float(re_reps.std(ddof=1)),
                    "covers_truth": bool(ci.low <= truth <= ci.high),
                }
            )
    return rows


def cmd_main(a: argparse.Namespace) -> None:
    out = []
    for campaign in CAMPAIGNS:
        for target in ("bts", "random"):
            t0 = time.perf_counter()
            obd, log, pi, truth, n_target = setting(campaign, target)
            iw = importance_weights(log, pi)
            models: dict[str, Any] = {"none": None}
            models["item-position"] = model_terms(log, pi, item_position_model(log))
            timing = {}
            if obd.has_context and not a.no_models:
                for name, make in (
                    ("ftrl", lambda: FTRLProximal(alpha=0.05, l1=1.0, bits=20)),
                    ("fm", lambda: FactorizationMachine(k=8, bits=20, lr=0.05)),
                ):
                    s = time.perf_counter()
                    models[name] = cross_fit_model_terms(obd, pi, make, n_folds=2, seed=0)
                    timing[name] = time.perf_counter() - s
            rows = evaluate(log, pi, truth, models, a.boot)
            n = len(log)
            se_truth = float(np.sqrt(truth * (1 - truth) / n_target))
            res = {
                "campaign": campaign,
                "target": target,
                "behavior": "random" if target == "bts" else "bts",
                "rounds": n,
                "target_rounds": n_target,
                "truth": truth,
                "truth_se": se_truth,
                "behavior_ctr": float(log.reward.mean()),
                "max_weight": float(iw.max()),
                "mean_weight": float(iw.mean()),
                "effective_sample_size": float(iw.sum() ** 2 / (iw**2).sum()),
                "switch_tau": float(np.quantile(iw, TAU_QUANTILE)),
                "model_fit_seconds": timing,
                "seconds": time.perf_counter() - t0,
                "estimates": rows,
            }
            out.append(res)
            print(campaign, target, f"truth {truth:.6f}", flush=True)
            for r in rows:
                print(
                    f"  {r['estimator']:<10} {r['reward_model']:<14} {r['estimate']:.6f} "
                    f"RE {r['relative_error']:.4f} covers {r['covers_truth']}",
                    flush=True,
                )
    write("ope_obd", {"n_boot": a.boot, "switch_tau_quantile": TAU_QUANTILE, "settings": out})


def cmd_curve(a: argparse.Namespace) -> None:
    out = []
    for target in ("bts", "random"):
        _, log, pi, truth, _ = setting("all", target)
        rng = np.random.default_rng(0)
        for n in (10_000, 30_000, 100_000, 300_000, 1_000_000):
            if n > len(log):
                continue
            errs: dict[str, list[float]] = {e: [] for e in ESTIMATORS}
            for _ in range(a.repeats):
                sub = log.take(np.sort(rng.choice(len(log), n, replace=False)))
                iw = importance_weights(sub, pi)
                q_pi, q_f = model_terms(sub, pi, item_position_model(sub))
                tau = float(np.quantile(iw, TAU_QUANTILE))
                for e in ESTIMATORS:
                    v = round_terms(e, sub.reward, iw, q_pi, q_f, tau=tau).value()
                    errs[e].append(relative_error(v, truth))
            row = {
                "target": target,
                "rounds": n,
                "repeats": a.repeats,
                **{
                    e: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1))}
                    for e, v in errs.items()
                },
            }
            out.append(row)
            print(target, n, {e: round(row[e]["mean"], 4) for e in ESTIMATORS}, flush=True)
    write("ope_curve", {"campaign": "all", "reward_model": "item-position", "curve": out})


def cmd_ablate(a: argparse.Namespace) -> None:
    out = []
    for campaign in CAMPAIGNS:
        for target in ("bts", "random"):
            _, log, pi, truth, _ = setting(campaign, target)
            iw = importance_weights(log, pi)
            q_pi, q_f = model_terms(log, pi, item_position_model(log))
            for q in (0.5, 0.9, 0.99, 0.999, 1.0):
                tau = float(np.quantile(iw, q))
                v = round_terms("switch-dr", log.reward, iw, q_pi, q_f, tau=tau).value()
                c = round_terms("ips", log.reward, iw, clip=tau).value()
                out.append(
                    {
                        "campaign": campaign,
                        "target": target,
                        "quantile": q,
                        "threshold": tau,
                        "switch_dr": v,
                        "switch_dr_relative_error": relative_error(v, truth),
                        "clipped_ips": c,
                        "clipped_ips_relative_error": relative_error(c, truth),
                    }
                )
            for prior in (0.0, 1.0, 10.0, 100.0, 1000.0):
                qp, qf = model_terms(log, pi, item_position_model(log, prior=prior))
                for e in ("dm", "dr"):
                    v = round_terms(e, log.reward, iw, qp, qf).value()
                    out.append(
                        {
                            "campaign": campaign,
                            "target": target,
                            "prior": prior,
                            "estimator": e,
                            "estimate": v,
                            "relative_error": relative_error(v, truth),
                        }
                    )
    write("ope_ablation", {"rows": out})


def cmd_obp(a: argparse.Namespace) -> None:
    import obp.ope as ref

    from ad_auction_lab.ope import estimate

    out = []
    for campaign in CAMPAIGNS:
        for target in ("bts", "random"):
            _, log, pi, truth, _ = setting(campaign, target)
            n = min(a.rows, len(log))
            sub = log.take(np.arange(n))
            q = item_position_model(sub)
            iw = importance_weights(sub, pi)
            tau = float(np.quantile(iw, TAU_QUANTILE))
            pi3 = np.ascontiguousarray(np.broadcast_to(pi, (n, *pi.shape)))
            q3 = np.ascontiguousarray(np.broadcast_to(q, (n, *q.shape)))
            kw = {
                "reward": sub.reward,
                "action": sub.action,
                "position": sub.position,
                "pscore": sub.pscore,
                "action_dist": pi3,
            }
            refs = {
                "ips": (ref.InverseProbabilityWeighting(), {}),
                "snips": (ref.SelfNormalizedInverseProbabilityWeighting(), {}),
                "dm": (ref.DirectMethod(), {"estimated_rewards_by_reg_model": q3}),
                "dr": (ref.DoublyRobust(), {"estimated_rewards_by_reg_model": q3}),
                "sndr": (ref.SelfNormalizedDoublyRobust(), {"estimated_rewards_by_reg_model": q3}),
                "switch-dr": (
                    ref.SwitchDoublyRobust(lambda_=tau),
                    {"estimated_rewards_by_reg_model": q3},
                ),
            }
            for name, (est, extra) in refs.items():
                args = {**kw, **extra}
                if name == "dm":
                    args = {k: v for k, v in args.items() if k not in ("reward", "pscore")}
                t0 = time.perf_counter()
                expected = float(est.estimate_policy_value(**args))
                t_ref = time.perf_counter() - t0
                t0 = time.perf_counter()
                got = estimate(name, sub, pi, q if name != "ips" and name != "snips" else None, tau)
                t_ours = time.perf_counter() - t0
                t0 = time.perf_counter()
                est.estimate_interval(**args, n_bootstrap_samples=a.boot, random_state=0)
                t_ref_ci = time.perf_counter() - t0
                t0 = time.perf_counter()
                qp, qf = model_terms(sub, pi, q)
                bootstrap(round_terms(name, sub.reward, iw, qp, qf, tau=tau), n_boot=a.boot)
                t_ours_ci = time.perf_counter() - t0
                out.append(
                    {
                        "campaign": campaign,
                        "target": target,
                        "rounds": n,
                        "estimator": name,
                        "ours": got,
                        "obp": expected,
                        "abs_difference": abs(got - expected),
                        "ours_seconds": t_ours,
                        "obp_seconds": t_ref,
                        "ours_bootstrap_seconds": t_ours_ci,
                        "obp_bootstrap_seconds": t_ref_ci,
                        "truth": truth,
                    }
                )
                print(out[-1], flush=True)
            del pi3, q3
    write("ope_obp", {"n_boot": a.boot, "obp_version": "0.5.7", "rows": out})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("main")
    s.add_argument("--boot", type=int, default=200)
    s.add_argument("--no-models", action="store_true")
    s.set_defaults(func=cmd_main)
    s = sub.add_parser("curve")
    s.add_argument("--repeats", type=int, default=30)
    s.set_defaults(func=cmd_curve)
    sub.add_parser("ablate").set_defaults(func=cmd_ablate)
    s = sub.add_parser("obp")
    s.add_argument("--rows", type=int, default=200_000)
    s.add_argument("--boot", type=int, default=100)
    s.set_defaults(func=cmd_obp)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
