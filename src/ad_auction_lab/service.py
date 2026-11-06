"""A standard-library HTTP service for auctions, bid shading, CTR prediction and OPE.

A pure :meth:`AuctionService.handle` maps ``(method, path, body)`` to a response so the routes are
tested without sockets; :func:`serve` binds it to a ``ThreadingHTTPServer``.

Endpoints
---------
``GET  /health``   liveness.
``GET  /ready``    readiness (and whether a CTR model is loaded).
``GET  /metrics``  request counts, errors and latency per route (Prometheus text format).
``POST /auction``  ``{"bids", "quality", "slot_ctr", "mechanism", "reserve", "squash"}``.
``POST /shade``    ``{"values", "market"}``: surplus-maximizing first-price bids.
``POST /predict``  ``{"small": [[int]], "wide": [[int]]}``: click probabilities (needs a model).
``POST /ope``      ``{"reward", "action", "pscore", "position", "action_dist", "q_hat",
                   "estimators", "tau", "clip"}``: off-policy estimates.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np

from ad_auction_lab import __version__
from ad_auction_lab.auctions import MECHANISMS, run_auctions
from ad_auction_lab.bidding import EmpiricalLandscape
from ad_auction_lab.ctr import FactorizationMachine, FTRLProximal
from ad_auction_lab.data import Table
from ad_auction_lab.ope import ESTIMATORS, BanditLog, estimate

JsonObject = dict[str, Any]
MAX_BODY = 2_000_000
MAX_ROWS = 10_000
MAX_MARKET = 200_000
BUCKETS = (1.0, 5.0, 25.0, 100.0, 500.0, float("inf"))


class BadRequestError(ValueError):
    pass


@dataclass
class Metrics:
    """Thread-safe per-route counters and a latency histogram in milliseconds."""

    requests: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    errors: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    latency_sum: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    buckets: dict[str, list[int]] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def observe(self, route: str, ms: float, error: bool) -> None:
        with self.lock:
            self.requests[route] += 1
            self.latency_sum[route] += ms
            if error:
                self.errors[route] += 1
            counts = self.buckets.setdefault(route, [0] * len(BUCKETS))
            for i, edge in enumerate(BUCKETS):
                if ms <= edge:
                    counts[i] += 1

    def render(self) -> str:
        lines = [
            "# HELP aal_requests_total Requests per route.",
            "# TYPE aal_requests_total counter",
        ]
        with self.lock:
            for r, n in sorted(self.requests.items()):
                lines.append(f'aal_requests_total{{route="{r}"}} {n}')
            lines += [
                "# HELP aal_errors_total Error responses per route.",
                "# TYPE aal_errors_total counter",
            ]
            for r in sorted(self.requests):
                lines.append(f'aal_errors_total{{route="{r}"}} {self.errors.get(r, 0)}')
            lines += [
                "# HELP aal_request_ms Request latency in milliseconds.",
                "# TYPE aal_request_ms histogram",
            ]
            for r, counts in sorted(self.buckets.items()):
                for edge, c in zip(BUCKETS, counts, strict=True):
                    le = "+Inf" if math.isinf(edge) else f"{edge:g}"
                    lines.append(f'aal_request_ms_bucket{{route="{r}",le="{le}"}} {c}')
                lines.append(f'aal_request_ms_sum{{route="{r}"}} {self.latency_sum[r]:.3f}')
                lines.append(f'aal_request_ms_count{{route="{r}"}} {self.requests[r]}')
        return "\n".join(lines) + "\n"


def _floats(body: JsonObject, key: str, required: bool = True) -> np.ndarray | None:
    if key not in body:
        if required:
            raise BadRequestError(f"missing field {key!r}")
        return None
    try:
        arr = np.asarray(body[key], dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise BadRequestError(f"{key!r} must be numeric") from e
    if arr.size > MAX_ROWS * 100:
        raise BadRequestError(f"{key!r} is too large")
    return arr


@dataclass
class AuctionService:
    model: FTRLProximal | FactorizationMachine | None = None
    metrics: Metrics = field(default_factory=Metrics)

    def handle(self, method: str, path: str, raw: bytes = b"") -> tuple[int, str, bytes]:
        route = path.split("?", 1)[0]
        start = time.perf_counter()
        status, ctype, payload = 500, "application/json", b"{}"
        try:
            status, ctype, payload = self._dispatch(method, route, raw)
        except BadRequestError as e:
            status, payload = 400, json.dumps({"error": str(e)}).encode()
        except ValueError as e:
            status, payload = 422, json.dumps({"error": str(e)}).encode()
        finally:
            known = route if route in ROUTES else "other"
            self.metrics.observe(known, (time.perf_counter() - start) * 1e3, status >= 400)
        return status, ctype, payload

    def _dispatch(self, method: str, route: str, raw: bytes) -> tuple[int, str, bytes]:
        if route not in ROUTES:
            return 404, "application/json", b'{"error": "not found"}'
        want = ROUTES[route]
        if method != want:
            return 405, "application/json", json.dumps({"error": f"use {want}"}).encode()
        if route == "/metrics":
            return 200, "text/plain; version=0.0.4", self.metrics.render().encode()
        if method == "GET":
            out: JsonObject = {"status": "ok", "version": __version__}
            if route == "/ready":
                out = {"ready": True, "ctr_model": self.model is not None}
            return 200, "application/json", json.dumps(out).encode()
        if len(raw) > MAX_BODY:
            raise BadRequestError("body too large")
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError as e:
            raise BadRequestError("body must be JSON") from e
        if not isinstance(body, dict):
            raise BadRequestError("body must be a JSON object")
        handler = {
            "/auction": self.auction,
            "/shade": self.shade,
            "/predict": self.predict,
            "/ope": self.ope,
        }[route]
        return 200, "application/json", json.dumps(handler(body)).encode()

    def auction(self, body: JsonObject) -> JsonObject:
        bids = _floats(body, "bids")
        assert bids is not None
        quality = _floats(body, "quality", required=False)
        slots = _floats(body, "slot_ctr", required=False)
        mech = body.get("mechanism", "gsp")
        if mech not in MECHANISMS:
            raise BadRequestError(f"mechanism must be one of {MECHANISMS}")
        if bids.ndim != 1 or bids.size == 0 or bids.size > 1000:
            raise BadRequestError("bids must be a list of 1 to 1000 numbers")
        q = np.ones_like(bids) if quality is None else quality
        alpha = np.array([1.0]) if slots is None else slots
        r = run_auctions(
            bids,
            q,
            alpha,
            mech,
            reserve=float(body.get("reserve", 0.0)),
            squash=float(body.get("squash", 1.0)),
        )
        return {
            "mechanism": mech,
            "slots": [
                {"slot": j, "bidder": int(w), "price_per_click": float(p)}
                for j, (w, p) in enumerate(zip(r.winner[0], r.price[0], strict=True))
            ],
            "expected_revenue": float(r.revenue()[0]),
        }

    def shade(self, body: JsonObject) -> JsonObject:
        values = _floats(body, "values")
        market = _floats(body, "market")
        assert values is not None and market is not None
        if values.size > MAX_ROWS or market.size > MAX_MARKET:
            raise BadRequestError("too many values or market prices")
        land = EmpiricalLandscape.fit(market)
        bids = land.shade(values)
        return {
            "bids": bids.reshape(-1).tolist(),
            "win_rate": land.win_rate(bids).reshape(-1).tolist(),
        }

    def predict(self, body: JsonObject) -> JsonObject:
        if self.model is None:
            raise BadRequestError("no CTR model loaded (start the service with --model)")
        try:
            small = np.asarray(body.get("small", []), dtype=np.int64)
            wide = np.asarray(body.get("wide", []), dtype=np.int64)
        except (TypeError, ValueError) as e:
            raise BadRequestError("small and wide must be integer matrices") from e
        n = max(len(small), len(wide))
        if n == 0 or n > MAX_ROWS:
            raise BadRequestError(f"send 1 to {MAX_ROWS} rows")
        small = small.reshape(n, -1) if small.size else np.zeros((n, 0), np.int64)
        wide = wide.reshape(n, -1) if wide.size else np.zeros((n, 0), np.int64)
        if small.size and (np.min(small) < 0 or np.max(small) > 255):
            raise BadRequestError("small features must be in [0, 255]")
        if wide.size and (np.min(wide) < 0 or np.max(wide) >= 2**32):
            raise BadRequestError("wide features must be 32-bit unsigned")
        t = Table(small.astype(np.uint8), wide.astype(np.uint32), np.zeros(n, np.uint8))
        return {"ctr": self.model.predict_proba(t).astype(float).tolist()}

    def ope(self, body: JsonObject) -> JsonObject:
        reward = _floats(body, "reward")
        pscore = _floats(body, "pscore")
        pi = _floats(body, "action_dist")
        q = _floats(body, "q_hat", required=False)
        assert reward is not None and pscore is not None and pi is not None
        try:
            action = np.asarray(body["action"], dtype=np.int64)
            position = np.asarray(body.get("position", np.zeros_like(action)), dtype=np.int64)
        except (KeyError, TypeError, ValueError) as e:
            raise BadRequestError("action (and position) must be integer lists") from e
        if pi.ndim != 2:
            raise BadRequestError("action_dist must be an (n_actions, n_positions) matrix")
        log = BanditLog.from_arrays(reward, action, pscore, position, pi.shape[0], pi.shape[1])
        names = body.get("estimators", list(ESTIMATORS))
        if not isinstance(names, list) or any(n not in ESTIMATORS for n in names):
            raise BadRequestError(f"estimators must be a subset of {ESTIMATORS}")
        tau = float(body.get("tau", math.inf))
        clip = float(body.get("clip", math.inf))
        out = {}
        for name in names:
            if q is None and name not in ("ips", "snips"):
                continue
            out[name] = estimate(name, log, pi, q, tau=tau, clip=clip)
        return {"rounds": len(log), "estimates": out}


ROUTES = {
    "/health": "GET",
    "/ready": "GET",
    "/metrics": "GET",
    "/auction": "POST",
    "/shade": "POST",
    "/predict": "POST",
    "/ope": "POST",
}


def make_handler(service: AuctionService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = f"ad-auction-lab/{__version__}"

        def _respond(self, method: str) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                status, ctype, payload = 413, "application/json", b'{"error": "body too large"}'
                self.close_connection = True
            else:
                raw = self.rfile.read(length) if length else b""
                status, ctype, payload = service.handle(method, self.path, raw)
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            self._respond("GET")

        def do_POST(self) -> None:
            self._respond("POST")

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def serve(
    service: AuctionService, host: str = "127.0.0.1", port: int = 8000
) -> ThreadingHTTPServer:
    """Create (but do not start) the server; call ``serve_forever()`` on the result."""

    return ThreadingHTTPServer((host, port), make_handler(service))
