"""Print the README and EVALUATION_REPORT tables from ``benchmarks/results/*.json``.

uv run python benchmarks/report.py            # every table
uv run python benchmarks/report.py ctr ope    # selected sections
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, read


def have(name: str) -> bool:
    return (RESULTS / f"{name}.json").exists()


def ms(values: list[float], digits: int = 4) -> str:
    a = np.asarray(values, dtype=float)
    if a.size == 1:
        return f"{a[0]:.{digits}f}"
    return f"{a.mean():.{digits}f} ± {a.std(ddof=1):.{digits}f}"


def table(header: list[str], rows: list[list[Any]]) -> None:
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|")
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")
    print()


# Zhu et al., "Open Benchmarking for Click-Through Rate Prediction" (CIKM 2021), BARS leaderboard
# https://openbenchmark.github.io/BARS/CTR/leaderboard/criteo_x4.html (full Criteo, random 8:1:1).
PUBLISHED = [
    ("LR", "0.4568", "0.7934"),
    ("FM", "0.4431", "0.8086"),
    ("FFM", "0.4407", "0.8113"),
    ("DeepFM", "0.4376", "0.8143"),
]


def ctr() -> None:
    if have("ctr_sweep"):
        print("### CTR sweep (validation log loss, best per model)\n")
        rows = []
        for name in ("ctr_sweep", "ctr_sweep_fm"):
            if not have(name):
                continue
            for run in read(name)["runs"]:
                for e in run["epochs"]:
                    p = run["params"]
                    kind = "fm" if "k" in p else "ftrl"
                    rows.append((e["valid"]["logloss"], kind, p, e["epoch"], e["valid"]["auc"]))
        rows.sort(key=lambda r: r[0])
        best: dict[str, Any] = {}
        for r in rows:
            best.setdefault(r[1], r)
        top = sorted(rows)[:8]
        table(
            ["Model", "Params", "Epoch", "Valid logloss", "Valid AUC"],
            [[r[1], r[2], r[3], f"{r[0]:.5f}", f"{r[4]:.5f}"] for r in top],
        )
        for kind, r in best.items():
            print(f"best {kind}: {r[2]} epoch {r[3]} logloss {r[0]:.5f}\n")
    if have("ctr_final"):
        f = read("ctr_final")
        print(f"### CTR final ({f['rows']}, base rate {f['base_rate']:.4f})\n")
        rows = []
        names = {"ftrl": "LR, FTRL-Proximal", "fm": "Factorization machine"}
        for kind in ("ftrl", "fm"):
            runs = f[kind]
            t = [r["test"] for r in runs]
            train_s = [sum(e["seconds"] for e in r["epochs"]) for r in runs]
            rows.append(
                [
                    names[kind],
                    ms([x["logloss"] for x in t]),
                    ms([x["auc"] for x in t]),
                    ms([x["normalized_entropy"] for x in t]),
                    ms([x["calibration_ratio"] for x in t], 3),
                    ms([x["ece"] for x in t]),
                    ms(train_s, 1),
                    f"{np.mean([r['predict_rows_per_s'] for r in runs]) / 1e6:.2f}M",
                ]
            )
        table(
            [
                "Model",
                "Test logloss",
                "Test AUC",
                "NE",
                "Calibration",
                "ECE",
                "Train s",
                "Predict rows/s",
            ],
            rows,
        )
        print("Calibrated (fit on validation):\n")
        rows = []
        for kind in ("ftrl", "fm"):
            for cal in ("platt", "isotonic"):
                t = [r["calibrated"][cal] for r in f[kind]]
                rows.append(
                    [
                        f"{names[kind]} + {cal}",
                        ms([x["logloss"] for x in t]),
                        ms([x["auc"] for x in t]),
                        ms([x["normalized_entropy"] for x in t]),
                        ms([x["calibration_ratio"] for x in t], 3),
                        ms([x["ece"] for x in t]),
                    ]
                )
        table(["Model", "Test logloss", "Test AUC", "NE", "Calibration", "ECE"], rows)
        d, p = f["fm_vs_ftrl_auc"], f["fm_vs_ftrl_logloss"]
        print(
            f"FM - FTRL AUC {d['difference']:+.5f} [{d['ci_low']:+.5f}, {d['ci_high']:+.5f}] "
            f"p={d['p_value']:.3g}; logloss {p['difference']:+.5f} "
            f"[{p['ci_low']:+.5f}, {p['ci_high']:+.5f}] p={p['p_value']:.3g}\n"
        )
        print(
            "FTRL nonzero weights:",
            [r.get("nonzero_weights") for r in f["ftrl"]],
            "peak RSS MB",
            round(f["peak_rss_mb"]),
            "\n",
        )
        rel = f["fm"][0]["reliability"]["bins"]
        print("FM reliability (20 bins, seed 0): mean prediction / mean label")
        print(", ".join(f"{b['mean_prediction']:.3f}/{b['mean_label']:.3f}" for b in rel), "\n")
    if have("ctr_final"):
        f = read("ctr_final")
        print("### Against published Criteo baselines (quoted, not rerun)\n")
        rows = [
            [
                f"This repo: {names[k]} (1 in 4 sample, {f['rows']['train']:,} train rows)",
                ms([r["test"]["logloss"] for r in f[k]]),
                ms([r["test"]["auc"] for r in f[k]]),
            ]
            for k in ("ftrl", "fm")
        ]
        bars = "BARS Criteo_x4_001, 36.7M train rows"
        rows += [[f"{m} ({bars})", ll, auc] for m, ll, auc in PUBLISHED]
        table(["Model", "Test logloss", "Test AUC"], rows)
    for which in ("all", "sklearn", "torch"):
        if not have(f"ctr_reference_{which}"):
            continue
        r = read(f"ctr_reference_{which}")
        print(f"### References ({which})\n")
        rows = []
        if "constant_base_rate" in r:
            c = r["constant_base_rate"]
            rows.append(["constant base rate", f"{c['logloss']:.5f}", f"{c['auc']:.5f}", "", ""])
        if "sklearn_sgd_logistic" in r:
            s = r["sklearn_sgd_logistic"]
            ch = s["chosen"]
            sec = next(x["seconds"] for x in s["runs"] if all(x.get(k) == v for k, v in ch.items()))
            lr = ch["learning_rate"] + (f" {ch['eta0']}" if "eta0" in ch else "")
            rows.append(
                [
                    f"scikit-learn SGDClassifier (alpha {ch['alpha']}, {lr})",
                    f"{s['test']['logloss']:.5f}",
                    f"{s['test']['auc']:.5f}",
                    f"{s['test']['normalized_entropy']:.4f}",
                    f"{sec:.0f}",
                ]
            )
        if "torch_fm" in r:
            s = r["torch_fm"]
            ch = s["chosen"]
            sec = sum(
                x["seconds"] for x in s["runs"] if x["lr"] == ch["lr"] and x["epoch"] <= ch["epoch"]
            )
            rows.append(
                [
                    f"PyTorch FM (Adagrad lr {ch['lr']}, {ch['epoch']} epochs)",
                    f"{s['test']['logloss']:.5f}",
                    f"{s['test']['auc']:.5f}",
                    f"{s['test']['normalized_entropy']:.4f}",
                    f"{sec:.0f}",
                ]
            )
        table(["Model", "Test logloss", "Test AUC", "NE", "Train s"], rows)
    if have("ctr_ablation"):
        a = read("ctr_ablation")
        print("### CTR ablations (test split, chosen passes: FTRL 2, FM 1)\n")
        for key, col in (
            ("bits", "bits"),
            ("fields", "fields"),
            ("train_size", "train_rows"),
            ("fm_k", "k"),
            ("l1", "l1"),
        ):
            if key not in a:
                continue
            rows = []
            for r in a[key]:
                e = r["epochs"][-1]
                rows.append(
                    [
                        r.get("model", "fm"),
                        r[col],
                        f"{e['test']['logloss']:.5f}",
                        f"{e['test']['auc']:.5f}",
                        f"{e['test']['normalized_entropy']:.4f}",
                        f"{e['rows_per_s'] / 1e6:.2f}M",
                        r.get("nonzero_weights", ""),
                    ]
                )
            table(["Model", col, "logloss", "AUC", "NE", "train rows/s", "nonzero"], rows)
    if have("ctr_speed"):
        print("### Rust core against the Python reference\n")
        rows = []
        for r in read("ctr_speed")["results"]:
            rows.append(
                [
                    r["model"],
                    f"{r['rust_train_rows_per_s']:,.0f}",
                    f"{r['python_train_rows_per_s']:,.0f}",
                    f"{r['speedup_train']:.0f}x",
                    f"{r['rust_predict_rows_per_s']:,.0f}",
                    f"{r['python_predict_rows_per_s']:,.0f}",
                    f"{r['max_abs_prediction_difference']:.1e}",
                ]
            )
        table(
            [
                "Model",
                "Rust train rows/s",
                "Python train rows/s",
                "Speedup",
                "Rust predict rows/s",
                "Python predict rows/s",
                "Max abs diff",
            ],
            rows,
        )


def ope() -> None:
    if have("ope_obd"):
        r = read("ope_obd")
        print(f"### OPE on the Open Bandit Dataset ({r['n_boot']} Poisson bootstrap replicates)\n")
        for s in r["settings"]:
            print(
                f"**{s['campaign']}: {s['target']} from {s['behavior']}** ({s['rounds']:,} rounds;"
                f" truth {s['truth']:.6f} ± {1.96 * s['truth_se']:.6f} from "
                f"{s['target_rounds']:,} rounds; ESS {s['effective_sample_size']:,.0f}, "
                f"max weight {s['max_weight']:.2f})\n"
            )
            rows = []
            for e in s["estimates"]:
                rows.append(
                    [
                        e["estimator"],
                        e["reward_model"],
                        f"{e['estimate']:.6f}",
                        f"[{e['ci_low']:.6f}, {e['ci_high']:.6f}]",
                        f"{e['relative_error']:.4f}",
                        f"{e['bootstrap_relative_error_mean']:.4f} ± "
                        f"{e['bootstrap_relative_error_std']:.4f}",
                        "yes" if e["covers_truth"] else "no",
                    ]
                )
            table(
                ["Estimator", "Reward model", "Estimate", "95% CI", "RE", "Bootstrap RE", "Covers"],
                rows,
            )
        # Compact summary: relative error per estimator, best reward model, per setting.
        print("### OPE summary (relative error, best available reward model)\n")
        rows = []
        for s in r["settings"]:
            per: dict[str, float] = {}
            for e in s["estimates"]:
                if e["reward_model"] in ("none", "item-position"):
                    per[e["estimator"]] = e["relative_error"]
            rows.append(
                [f"{s['campaign']}: {s['target']} from {s['behavior']}", f"{s['truth']:.5f}"]
                + [
                    f"{per[k]:.3f}"
                    for k in ("ips", "snips", "dm", "dr", "sndr", "switch-dr")
                    if k in per
                ]
            )
        table(["Setting", "Truth", "IPS", "SNIPS", "DM", "DR", "SNDR", "Switch-DR"], rows)
    if have("ope_curve"):
        c = read("ope_curve")
        print("### OPE relative error against log size (all campaign, mean over repeats)\n")
        rows = []
        for r in c["curve"]:
            rows.append(
                [r["target"], f"{r['rounds']:,}"]
                + [f"{r[e]['mean']:.3f}" for e in ("ips", "snips", "dm", "dr", "sndr", "switch-dr")]
            )
        table(["Target", "Rounds", "IPS", "SNIPS", "DM", "DR", "SNDR", "Switch-DR"], rows)
    if have("ope_ablation"):
        rows_t, rows_p = [], []
        for r in read("ope_ablation")["rows"]:
            name = f"{r['campaign']}: {r['target']}"
            if "quantile" in r:
                rows_t.append(
                    [
                        name,
                        r["quantile"],
                        f"{r['threshold']:.2f}",
                        f"{r['switch_dr_relative_error']:.4f}",
                        f"{r['clipped_ips_relative_error']:.4f}",
                    ]
                )
            else:
                rows_p.append([name, r["estimator"], r["prior"], f"{r['relative_error']:.4f}"])
        print("### Switch-DR threshold and IPS clipping\n")
        table(["Setting", "Weight quantile", "Threshold", "Switch-DR RE", "Clipped IPS RE"], rows_t)
        print("### Reward model smoothing prior\n")
        table(["Setting", "Estimator", "Prior", "RE"], rows_p)
    if have("ope_obp"):
        o = read("ope_obp")
        rows = []
        worst = max(r["abs_difference"] for r in o["rows"])
        agg: dict[str, list[Any]] = defaultdict(list)
        for r in o["rows"]:
            agg[r["estimator"]].append(r)
        for est, rs in agg.items():
            rows.append(
                [
                    est,
                    len(rs),
                    f"{max(x['abs_difference'] for x in rs):.1e}",
                    f"{np.median([x['obp_seconds'] / x['ours_seconds'] for x in rs]):.0f}x",
                    f"{np.median([boot_ratio(x) for x in rs]):.1f}x",
                ]
            )
        print(f"### Agreement with obp {o['obp_version']} (worst abs difference {worst:.1e})\n")
        table(
            [
                "Estimator",
                "Settings",
                "Max abs diff",
                "Point estimate speedup",
                "Bootstrap speedup",
            ],
            rows,
        )


def boot_ratio(x: dict[str, float]) -> float:
    return x["obp_bootstrap_seconds"] / x["ours_bootstrap_seconds"]


def markets() -> None:
    if have("auction_incentives"):
        r = read("auction_incentives")
        print("### Incentives (share of bidder-auctions with a profitable deviation)\n")
        rows = []
        for x in r["rows"]:
            rows.append(
                [
                    x["mechanism"],
                    x["bidders"],
                    x["slots"],
                    "yes" if x["reserve"] > 0 else "no",
                    x["squash"],
                    f"{x['share_profitable']:.1%}",
                    f"{x['gain_over_welfare']:.2%}",
                    f"{x['truthful_revenue']:.4f}",
                    f"{x['truthful_welfare']:.4f}",
                ]
            )
        table(
            [
                "Mechanism",
                "Bidders",
                "Slots",
                "Reserve",
                "Squash",
                "Profitable",
                "Gain/welfare",
                "Revenue",
                "Welfare",
            ],
            rows,
        )
        agg: dict[tuple[str, bool], list[float]] = defaultdict(list)
        for x in r["rows"]:
            agg[(x["mechanism"], x["reserve"] > 0)].append(x["share_profitable"])
        print(
            "Summary:",
            {
                f"{k[0]}{' +reserve' if k[1] else ''}": f"{min(v):.1%}..{max(v):.1%}"
                for k, v in agg.items()
            },
            "\n",
        )
        print("EOS equilibrium:", r["gsp_envy_free_equilibrium"], "\n")
        print("VCG brute force:", r["vcg_bruteforce"], "\n")
    if have("auction_reserves"):
        rows = []
        for x in read("auction_reserves")["rows"]:
            rows.append(
                [
                    x["bidders"],
                    x["reserve_rule"],
                    f"{x['reserve']:.3f}",
                    f"{x['revenue_mean']:.4f} ± {1.96 * x['revenue_se']:.4f}",
                    f"{x['welfare_mean']:.4f}",
                    f"{x['sold_share']:.1%}",
                ]
            )
        print("### Reserve prices (second price, one slot)\n")
        table(["Bidders", "Reserve rule", "Reserve", "Revenue (95% CI)", "Welfare", "Sold"], rows)
    if have("bid_shading"):
        b = read("bid_shading")
        print(
            f"### Bid shading (train {b['train_rows']:,} / test {b['test_rows']:,} impressions)\n"
        )
        rows = []
        for x in b["global"]:
            rows.append(
                [
                    x["value_over_median_price"],
                    x["rule"],
                    f"{x['win_rate']:.1%}",
                    f"{x['surplus_share_of_oracle']:.1%}",
                ]
            )
        table(["Value / median price", "Rule", "Win rate", "Surplus vs oracle"], rows)
        per = b["per_campaign_value_2x_median"]
        print(
            "Per campaign (value 2x campaign median, n=",
            len(per),
            "): campaign landscape",
            ms([p["campaign_surplus_share"] for p in per], 3),
            "pooled",
            ms([p["pooled_surplus_share"] for p in per], 3),
            "\n",
        )
    if have("pacing"):
        p = read("pacing")
        print(f"### Pacing ({p['horizon']:,} auctions, mean ± std over seeds)\n")
        groups: dict[tuple[int, float, str], list[dict[str, Any]]] = defaultdict(list)
        for x in p["rows"]:
            groups[(x["advertisers"], x["budget_share"], x["pacer"])].append(x)
        rows = []
        for (n, share, pacer), xs in groups.items():
            base = groups[(n, share, "none")]
            rel = [x["value"] / b["value"] for x, b in zip(xs, base, strict=True)]
            rows.append(
                [
                    n,
                    f"{share:.0%}",
                    pacer,
                    ms(rel, 3),
                    ms([x["utilization"] for x in xs], 3),
                    ms([x["mean_exhaustion_time"] for x in xs], 3),
                    ms([x["max_tracking_error"] for x in xs], 3),
                    ms([x["roi"] / b["roi"] for x, b in zip(xs, base, strict=True)], 3),
                ]
            )
        table(
            [
                "Advertisers",
                "Budget",
                "Pacer",
                "Value vs none",
                "Utilization",
                "Exhaustion time",
                "Max tracking error",
                "ROI vs none",
            ],
            rows,
        )
        sg: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
        for x in p["single_advertiser"]:
            sg[(x["budget_share"], x["pacer"])].append(x)
        rows = [
            [
                f"{k[0]:.0%}",
                k[1],
                ms([x["competitive_ratio"] for x in xs], 3),
                ms([x["utilization"] for x in xs], 3),
            ]
            for k, xs in sg.items()
        ]
        print("Single advertiser against the hindsight LP optimum:\n")
        table(["Budget", "Pacer", "Value / hindsight optimum", "Utilization"], rows)
    if have("pacing_speed"):
        s = read("pacing_speed")
        rows = [
            [
                x["advertisers"],
                x["pacer"],
                f"{x['rust_auctions_per_s']:,.0f}",
                f"{x['python_auctions_per_s']:,.0f}",
                f"{x['speedup']:.0f}x",
                f"{x['max_abs_value_difference']:.1e}",
            ]
            for x in s["rows"]
        ]
        print("### Pacing simulator speed\n")
        table(
            ["Advertisers", "Pacer", "Rust auctions/s", "Python auctions/s", "Speedup", "Max diff"],
            rows,
        )
        print(
            "10M auctions x 10 advertisers: "
            f"{s['rust_10m_auctions_10_advertisers_seconds']:.2f} s\n"
        )


def main() -> None:
    sections = {"ctr": ctr, "ope": ope, "markets": markets}
    chosen = sys.argv[1:] or list(sections)
    for name in chosen:
        sections[name]()


if __name__ == "__main__":
    main()
