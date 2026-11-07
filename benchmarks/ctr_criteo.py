"""CTR models on Criteo: tuning, final multi-seed runs, calibration, ablations and scaling.

Training passes run over the training rows in file order, which is time order in the Criteo
log, as an online learner would see them (a shuffled pass over the 1.35 GB memory map is about
eight times slower on this machine because every row is a random page read).

Data: every 4th row of the Criteo Display Advertising Challenge ``train.txt`` (11,460,155 of
45,840,617 rows), split at random 80/10/10 with seed 0 (``split_rows``). Hyperparameters are
chosen on the validation split; the test split is touched only by ``final`` and ``ablate``.

    uv run python benchmarks/ctr_criteo.py sweep      # FTRL and FM grids on validation
    uv run python benchmarks/ctr_criteo.py sweep-fm   # finer FM grid (rates, factor L2, k)
    uv run python benchmarks/ctr_criteo.py final      # chosen configs, 3 random splits
    uv run python benchmarks/ctr_criteo.py ablate     # bits, feature groups, train size, k
    uv run python benchmarks/ctr_criteo.py reference  # scikit-learn SGD and a PyTorch FM
    uv run python benchmarks/ctr_criteo.py speed      # Rust core against the Python reference
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, peak_rss_mb, write

from ad_auction_lab.calibration import IsotonicCalibration, PlattScaling
from ad_auction_lab.ctr import FactorizationMachine, FTRLProximal
from ad_auction_lab.data import Table, load_criteo, split_rows
from ad_auction_lab.metrics import calibration, delong, paired_loss_test, summary

# Chosen on the validation split from ctr_sweep.json and ctr_sweep_fm.json. FTRL: alpha 0.05 with
# L1 1.0 is within 0.0002 logloss of L1 0 after 2 passes and keeps fewer weights. FM: k 32 at rate
# 0.01 after 1 pass is the best of both grids; every FM overfits on its second pass.
FTRL_BEST = {"alpha": 0.05, "l1": 1.0, "bits": 22}
FTRL_EPOCHS = 2
FM_BEST = {"k": 32, "lr": 0.01, "l2_factor": 1e-5, "bits": 22}
FM_EPOCHS = 1


def load(seed: int = 0) -> tuple[Table, Any, Any, Any, float]:
    t = load_criteo(DATA / "criteo")
    train, valid, test = split_rows(len(t), seed=seed)
    base = float(t.labels[train].mean())
    return t, train, valid, test, base


def run(
    model: FTRLProximal | FactorizationMachine,
    t: Table,
    train: Any,
    evals: dict[str, Any],
    base: float,
    epochs: int = 1,
) -> dict[str, Any]:
    out: dict[str, Any] = {"params": model.params(), "epochs": []}
    for e in range(1, epochs + 1):
        t0 = time.perf_counter()
        prog = model.partial_fit(t, train)
        sec = time.perf_counter() - t0
        row: dict[str, Any] = {
            "epoch": e,
            "seconds": sec,
            "rows_per_s": len(train) / sec,
            "progressive_logloss": prog,
        }
        for name, rows in evals.items():
            row[name] = summary(t.labels[rows], model.predict_proba(t, rows), base)
        out["epochs"].append(row)
        print(model.kind, model.params(), row, flush=True)
    return out


def cmd_sweep(a: argparse.Namespace) -> None:
    t, train, valid, _, base = load()
    if a.rows:
        train = train[: a.rows]
    runs = []
    for alpha in (0.02, 0.05, 0.1, 0.2):
        for l1 in (0.0, 1.0, 4.0):
            m = FTRLProximal(alpha=alpha, l1=l1, bits=22)
            runs.append(run(m, t, train, {"valid": valid}, base, epochs=2))
    for k in (8, 16):
        for lr in (0.02, 0.05, 0.1):
            m = FactorizationMachine(k=k, lr=lr, bits=22)
            runs.append(run(m, t, train, {"valid": valid}, base, epochs=3))
    write(
        "ctr_sweep",
        {"train_rows": len(train), "valid_rows": len(valid), "base_rate": base, "runs": runs},
    )


def cmd_sweep_fm(a: argparse.Namespace) -> None:
    """Second FM grid: smaller learning rates and stronger factor L2 (the first grid overfits)."""

    t, train, valid, _, base = load()
    runs = []
    for k in (8, 16, 32):
        for lr in (0.005, 0.01, 0.02):
            for l2f in (1e-5, 1e-4):
                m = FactorizationMachine(k=k, lr=lr, l2_factor=l2f, bits=22)
                runs.append(run(m, t, train, {"valid": valid}, base, epochs=2))
    write(
        "ctr_sweep_fm",
        {"train_rows": len(train), "valid_rows": len(valid), "base_rate": base, "runs": runs},
    )


def _calibrated(p_valid: Any, y_valid: Any, p_test: Any, y_test: Any, base: float) -> dict:
    out = {}
    for name, cal in (("platt", PlattScaling()), ("isotonic", IsotonicCalibration())):
        c = cal.fit(p_valid, y_valid)
        q = np.clip(c.transform(p_test), 1e-7, 1 - 1e-7)
        out[name] = summary(y_test, q, base)
    return out


def cmd_final(a: argparse.Namespace) -> None:
    results: dict[str, Any] = {"ftrl": [], "fm": []}
    keep: dict[str, Any] = {}
    sizes = {}
    for seed in range(a.seeds):
        t, train, valid, test, base = load(seed)
        yv, yt = t.labels[valid], t.labels[test]
        if seed == 0:
            sizes = {"train": len(train), "valid": len(valid), "test": len(test)}
            y0, base0 = yt, base
        for kind in ("ftrl", "fm"):
            if kind == "ftrl":
                m: Any = FTRLProximal(**FTRL_BEST)
                epochs = FTRL_EPOCHS
            else:
                m = FactorizationMachine(**FM_BEST, seed=seed)
                epochs = FM_EPOCHS
            r = run(m, t, train, {}, base, epochs=epochs)
            t0 = time.perf_counter()
            pt = m.predict_proba(t, test)
            pred_s = time.perf_counter() - t0
            pv = m.predict_proba(t, valid)
            r["test"] = summary(yt, pt, base)
            r["valid"] = summary(yv, pv, base)
            r["predict_rows_per_s"] = len(test) / pred_s
            r["calibrated"] = _calibrated(pv, yv, pt, yt, base)
            r["seed"] = seed
            if kind == "ftrl":
                w = m.weights()
                r["nonzero_weights"] = int(np.count_nonzero(w))
            if seed == 0:
                r["reliability"] = calibration(yt, pt, n_bins=20).to_dict()
                keep[kind] = pt
            results[kind].append(r)
            print(kind, seed, r["test"], flush=True)
    results["fm_vs_ftrl_auc"] = delong(y0, keep["fm"], keep["ftrl"]).to_dict()
    results["fm_vs_ftrl_logloss"] = paired_loss_test(y0, keep["fm"], keep["ftrl"]).to_dict()
    np.save(DATA / "criteo_test_ctr.npy", keep["fm"].astype(np.float32))
    write(
        "ctr_final",
        {
            "rows": sizes,
            "base_rate": base0,
            "seeds": "split seeds 0..n-1 (random 80/10/10 partitions); FM init seed = split seed",
            "ftrl_config": {**FTRL_BEST, "epochs": FTRL_EPOCHS},
            "fm_config": {**FM_BEST, "epochs": FM_EPOCHS},
            **results,
        },
    )


def _epochs(m: FTRLProximal | FactorizationMachine) -> int:
    return FTRL_EPOCHS if isinstance(m, FTRLProximal) else FM_EPOCHS


def cmd_ablate(a: argparse.Namespace) -> None:
    t, train, _valid, test, base = load()
    ev = {"test": test}
    out: dict[str, Any] = {"bits": [], "fields": [], "train_size": [], "fm_k": [], "l1": []}

    def save() -> None:  # after every group, so a stopped run keeps what it finished
        write("ctr_ablation", {"base_rate": base, "test_rows": len(test), **out})

    for l1 in (0.0, 1.0, 4.0, 16.0):
        m = FTRLProximal(**{**FTRL_BEST, "l1": l1})
        r = run(m, t, train, ev, base, epochs=_epochs(m))
        r["nonzero_weights"] = int(np.count_nonzero(m.weights()))
        out["l1"].append({"model": "ftrl", "l1": l1, **r})
    save()
    # The FM stops at 2^22 buckets: at 2^24 its factors and AdaGrad state need 4 GB at k = 32.
    for bits in (16, 18, 20, 22, 24):
        for kind in ("ftrl", "fm") if bits <= 22 else ("ftrl",):
            m: Any = (
                FTRLProximal(**{**FTRL_BEST, "bits": bits})
                if kind == "ftrl"
                else FactorizationMachine(**{**FM_BEST, "bits": bits})
            )
            r = run(m, t, train, ev, base, epochs=_epochs(m))
            out["bits"].append({"model": kind, "bits": bits, **r})
    save()
    n = len(t)
    groups = {
        "integer features only (13)": Table(t.small, np.zeros((n, 0), np.uint32), t.labels),
        "categorical features only (26)": Table(np.zeros((n, 0), np.uint8), t.wide, t.labels),
    }
    for name, tab in groups.items():
        for kind in ("ftrl", "fm"):
            m = FTRLProximal(**FTRL_BEST) if kind == "ftrl" else FactorizationMachine(**FM_BEST)
            r = run(m, tab, train, ev, base, epochs=_epochs(m))
            out["fields"].append({"model": kind, "fields": name, **r})
    save()
    for size in (100_000, 300_000, 1_000_000, 3_000_000, len(train)):
        for kind in ("ftrl", "fm"):
            m = FTRLProximal(**FTRL_BEST) if kind == "ftrl" else FactorizationMachine(**FM_BEST)
            r = run(m, t, train[:size], ev, base, epochs=_epochs(m))
            out["train_size"].append({"model": kind, "train_rows": size, **r})
    save()
    for k in (1, 2, 4, 8, 16, 32):
        m = FactorizationMachine(**{**FM_BEST, "k": k})
        r = run(m, t, train, ev, base, epochs=_epochs(m))
        out["fm_k"].append({"k": k, **r})
    save()


def cmd_speed(a: argparse.Namespace) -> None:
    t, train, _, test, _ = load()
    sub = t.take(np.sort(train[: a.rows]))
    held = t.take(np.sort(test[:20_000]))
    out = []
    # 2^20 buckets keeps both backends' state under 1 GB at k = 32; speed per row does not
    # depend on the table size, only the memory does.
    bits = a.bits
    for kind in ("ftrl", "fm"):
        row: dict[str, Any] = {"model": kind, "rows": len(sub), "bits": bits}
        preds = {}
        for backend in ("rust", "python"):
            m: Any = (
                FTRLProximal(**{**FTRL_BEST, "bits": bits}, backend=backend)
                if kind == "ftrl"
                else FactorizationMachine(**{**FM_BEST, "bits": bits}, backend=backend)
            )
            t0 = time.perf_counter()
            m.partial_fit(sub)
            row[f"{backend}_train_rows_per_s"] = len(sub) / (time.perf_counter() - t0)
            t0 = time.perf_counter()
            preds[backend] = m.predict_proba(held)
            row[f"{backend}_predict_rows_per_s"] = len(held) / (time.perf_counter() - t0)
            del m
        row["speedup_train"] = row["rust_train_rows_per_s"] / row["python_train_rows_per_s"]
        row["max_abs_prediction_difference"] = float(
            np.max(np.abs(preds["rust"].astype(np.float64) - preds["python"]))
        )
        out.append(row)
        print(row, flush=True)
    write("ctr_speed", {"results": out, "peak_rss_mb_at_end": peak_rss_mb()})


def _hashed_csr(t: Table, rows: Any, bits: int) -> Any:
    from scipy.sparse import csr_matrix

    from ad_auction_lab.hashing import hash_rows

    sub = t.take(np.asarray(rows))
    idx = hash_rows(sub.small, sub.wide, bits)
    n, f = idx.shape
    data = np.ones(n * f, dtype=np.float32)
    return csr_matrix((data, idx.reshape(-1), np.arange(0, n * f + 1, f)), shape=(n, 2**bits))


def _chunks(rows: Any, size: int = 500_000) -> Any:
    for s in range(0, len(rows), size):
        yield rows[s : s + size]


def ref_sklearn(t: Table, train: Any, valid: Any, test: Any, base: float, bits: int) -> dict:
    """Logistic regression with scikit-learn's SGDClassifier on the same hashed features.

    The default "optimal" step size diverges at small L2 on these features (validation logloss
    5.1 at alpha 1e-7, 0.70 at 1e-6), so the grid also covers constant step sizes.
    """

    from sklearn.linear_model import SGDClassifier

    grid: list[dict[str, Any]] = [{"alpha": 1e-5, "learning_rate": "optimal"}]
    grid += [
        {"alpha": l2, "learning_rate": "constant", "eta0": eta}
        for l2 in (1e-7, 1e-6)
        for eta in (0.005, 0.02, 0.05)
    ]
    runs = []
    best: tuple[float, Any, dict] | None = None
    for cfg in grid:
        clf = SGDClassifier(loss="log_loss", random_state=0, **cfg)
        t0 = time.perf_counter()
        for c in _chunks(train):
            clf.partial_fit(_hashed_csr(t, c, bits), t.labels[c], classes=[0, 1])
        sec = time.perf_counter() - t0
        pv = np.concatenate(
            [clf.predict_proba(_hashed_csr(t, c, bits))[:, 1] for c in _chunks(valid)]
        )
        s = summary(t.labels[valid], pv, base)
        runs.append({**cfg, "seconds": sec, "rows_per_s": len(train) / sec, "valid": s})
        print("sklearn", cfg, s, flush=True)
        if best is None or s["logloss"] < best[0]:
            best = (s["logloss"], clf, cfg)
    assert best is not None
    clf = best[1]
    pt = np.concatenate([clf.predict_proba(_hashed_csr(t, c, bits))[:, 1] for c in _chunks(test)])
    return {"runs": runs, "chosen": best[2], "test": summary(t.labels[test], pt, base)}


def ref_torch_fm(
    t: Table, train: Any, valid: Any, test: Any, base: float, bits: int, k: int, epochs: int
) -> dict:
    """The same second-order FM in PyTorch (sparse embeddings, Adagrad, mini-batches of 1024)."""

    import torch

    from ad_auction_lab.hashing import hash_rows

    torch.manual_seed(0)
    torch.set_num_threads(4)

    class FM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Embedding(2**bits, 1, sparse=True)
            self.v = torch.nn.Embedding(2**bits, k, sparse=True)
            self.b = torch.nn.Parameter(torch.zeros(1))
            torch.nn.init.zeros_(self.w.weight)
            torch.nn.init.normal_(self.v.weight, std=0.01)

        def forward(self, idx: Any) -> Any:
            v = self.v(idx)
            inter = 0.5 * (v.sum(1).pow(2) - v.pow(2).sum(1)).sum(1)
            return self.b + self.w(idx).sum((1, 2)) + inter

    def predict(model: Any, rows: Any) -> np.ndarray:
        out = []
        with torch.no_grad():
            for c in _chunks(rows, 200_000):
                sub = t.take(np.asarray(c))
                idx = torch.from_numpy(hash_rows(sub.small, sub.wide, bits))
                out.append(torch.sigmoid(model(idx)).numpy())
        return np.concatenate(out)

    runs = []
    best: tuple[float, dict] | None = None
    for lr in (0.02, 0.05, 0.1):
        model = FM()
        opt = torch.optim.Adagrad(list(model.parameters()), lr=lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        for e in range(1, epochs + 1):
            t0 = time.perf_counter()
            for c in _chunks(train, 200_000):
                sub = t.take(np.asarray(c))
                idx = torch.from_numpy(hash_rows(sub.small, sub.wide, bits))
                y = torch.from_numpy(sub.labels.astype(np.float32))
                for s in range(0, len(c), 1024):
                    opt.zero_grad()
                    loss = loss_fn(model(idx[s : s + 1024]), y[s : s + 1024])
                    loss.backward()
                    opt.step()
            sec = time.perf_counter() - t0
            sv = summary(t.labels[valid], predict(model, valid), base)
            row = {
                "lr": lr,
                "epoch": e,
                "seconds": sec,
                "rows_per_s": len(train) / sec,
                "valid": sv,
            }
            runs.append(row)
            print("torch fm", row, flush=True)
            if best is None or sv["logloss"] < best[0]:
                best = (sv["logloss"], {"lr": lr, "epoch": e})
                best_test = summary(t.labels[test], predict(model, test), base)
    assert best is not None
    return {"runs": runs, "chosen": best[1], "test": best_test, "batch": 1024, "k": k}


def cmd_reference(a: argparse.Namespace) -> None:
    t, train, valid, test, base = load()
    out: dict[str, Any] = {"bits": 22, "train_rows": len(train), "test_rows": len(test)}
    p_const = np.full(len(test), base)
    out["constant_base_rate"] = summary(t.labels[test], p_const, base)
    if a.which in ("all", "sklearn"):
        out["sklearn_sgd_logistic"] = ref_sklearn(t, train, valid, test, base, 22)
    if a.which in ("all", "torch"):
        out["torch_fm"] = ref_torch_fm(t, train, valid, test, base, 22, FM_BEST["k"], FM_EPOCHS + 1)
    write(f"ctr_reference_{a.which}", out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sweep")
    s.add_argument("--rows", type=int, default=0)
    s.set_defaults(func=cmd_sweep)
    sub.add_parser("sweep-fm").set_defaults(func=cmd_sweep_fm)
    s = sub.add_parser("final")
    s.add_argument("--seeds", type=int, default=3)
    s.set_defaults(func=cmd_final)
    sub.add_parser("ablate").set_defaults(func=cmd_ablate)
    s = sub.add_parser("speed")
    s.add_argument("--rows", type=int, default=20_000)
    s.add_argument("--bits", type=int, default=20)
    s.set_defaults(func=cmd_speed)
    s = sub.add_parser("reference")
    s.add_argument("--which", choices=["all", "sklearn", "torch"], default="all")
    s.set_defaults(func=cmd_reference)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
