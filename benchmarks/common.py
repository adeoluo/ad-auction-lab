"""Shared helpers for the benchmark scripts: data paths, machine description, result files."""

from __future__ import annotations

import contextlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
DATA = Path(os.environ.get("AAL_DATA", ROOT.parent / ".bench-data" / "ad-auction-lab"))


def peak_rss_mb() -> float:
    """Peak resident set size of this process in MB (bytes on macOS, KB on Linux)."""

    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 1024


def machine() -> dict[str, Any]:
    cpu = platform.processor()
    if sys.platform == "darwin":
        with contextlib.suppress(OSError):
            cpu = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            ).stdout.strip()
    return {
        "platform": platform.platform(),
        "cpu": cpu,
        "cores": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "load_average_1m": os.getloadavg()[0],
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _default(o: object) -> object:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def write(name: str, payload: dict[str, Any]) -> Path:
    """Write ``benchmarks/results/<name>.json`` with the machine description added."""

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{name}.json"
    body = {"machine": machine(), "peak_rss_mb": peak_rss_mb(), **payload}
    out.write_text(json.dumps(body, indent=2, default=_default) + "\n")
    print(f"wrote {out.relative_to(ROOT)}", flush=True)
    return out


def read(name: str) -> dict[str, Any]:
    out: dict[str, Any] = json.loads((RESULTS / f"{name}.json").read_text())
    return out
