"""HTTP service: pure handler routes, validation, metrics, and one real socket round trip."""

from __future__ import annotations

import http.client
import json
import threading
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from ad_auction_lab.ctr import FTRLProximal
from ad_auction_lab.data import Table
from ad_auction_lab.service import AuctionService, serve


def call(svc: AuctionService, method: str, path: str, body: object = None) -> tuple[int, dict]:  # type: ignore[type-arg]
    raw = b"" if body is None else json.dumps(body).encode()
    status, ctype, payload = svc.handle(method, path, raw)
    if ctype.startswith("application/json"):
        return status, json.loads(payload)
    return status, {"text": payload.decode()}


@pytest.fixture
def svc(small_clicks: Table) -> AuctionService:
    m = FTRLProximal(bits=10)
    m.fit(small_clicks)
    return AuctionService(model=m)


def test_health_ready(svc: AuctionService) -> None:
    assert call(svc, "GET", "/health") == (200, {"status": "ok", "version": "0.1.0"})
    assert call(svc, "GET", "/ready")[1] == {"ready": True, "ctr_model": True}
    assert call(AuctionService(), "GET", "/ready")[1]["ctr_model"] is False


def test_auction_route(svc: AuctionService) -> None:
    status, out = call(
        svc, "POST", "/auction", {"bids": [4, 2, 1], "slot_ctr": [0.2, 0.1], "mechanism": "vcg"}
    )
    assert status == 200
    assert [s["bidder"] for s in out["slots"]] == [0, 1]
    assert out["slots"][0]["price_per_click"] == pytest.approx(1.5)
    assert out["expected_revenue"] == pytest.approx(0.4)
    status, out = call(svc, "POST", "/auction", {"bids": [3, 1]})
    assert status == 200 and out["slots"][0]["price_per_click"] == 1.0


@pytest.mark.parametrize(
    ("body", "status"),
    [
        ({}, 400),
        ({"bids": "abc"}, 400),
        ({"bids": [1], "mechanism": "english"}, 400),
        ({"bids": []}, 400),
        ({"bids": [1, 2], "quality": [1, 2, 3]}, 422),
        ({"bids": [-1]}, 422),
        ([1, 2], 400),
    ],
)
def test_auction_validation(svc: AuctionService, body: object, status: int) -> None:
    assert call(svc, "POST", "/auction", body)[0] == status


def test_bad_json_and_routes(svc: AuctionService) -> None:
    assert svc.handle("POST", "/auction", b"{not json")[0] == 400
    assert svc.handle("GET", "/nope")[0] == 404
    assert svc.handle("GET", "/auction")[0] == 405
    assert svc.handle("POST", "/health")[0] == 405
    assert svc.handle("POST", "/auction", b"x" * 3_000_000)[0] == 400


def test_shade_route(svc: AuctionService) -> None:
    market = np.random.default_rng(0).lognormal(0, 1, 1000).tolist()
    status, out = call(svc, "POST", "/shade", {"values": [0.5, 2.0, 10.0], "market": market})
    assert status == 200
    assert all(0 <= b <= v for b, v in zip(out["bids"], [0.5, 2.0, 10.0], strict=True))
    assert out["win_rate"][2] > out["win_rate"][0]
    assert call(svc, "POST", "/shade", {"values": [1.0], "market": []})[0] == 422
    assert call(svc, "POST", "/shade", {"values": [1.0] * 20_000, "market": [1.0]})[0] == 400


def test_predict_route(svc: AuctionService, small_clicks: Table) -> None:
    body = {"small": small_clicks.small[:5].tolist(), "wide": small_clicks.wide[:5].tolist()}
    status, out = call(svc, "POST", "/predict", body)
    assert status == 200
    expected = svc.model.predict_proba(small_clicks, np.arange(5))  # type: ignore[union-attr]
    assert np.allclose(out["ctr"], expected, atol=1e-6)
    assert call(svc, "POST", "/predict", {"small": [], "wide": []})[0] == 400
    assert call(svc, "POST", "/predict", {"small": [[300]]})[0] == 400
    assert call(svc, "POST", "/predict", {"wide": [[-1]]})[0] == 400
    assert call(svc, "POST", "/predict", {"wide": [["a"]]})[0] == 400
    assert call(AuctionService(), "POST", "/predict", body)[0] == 400
    status, out = call(svc, "POST", "/predict", {"wide": [[1, 2, 3]]})
    assert status == 200 and len(out["ctr"]) == 1


def test_ope_route(svc: AuctionService) -> None:
    body = {
        "reward": [1, 0, 1, 0],
        "action": [0, 1, 1, 0],
        "pscore": [0.5] * 4,
        "action_dist": [[0.8], [0.2]],
    }
    status, out = call(svc, "POST", "/ope", body)
    assert status == 200 and set(out["estimates"]) == {"ips", "snips"}
    assert out["estimates"]["ips"] == pytest.approx(0.5)
    status, out = call(svc, "POST", "/ope", {**body, "q_hat": [[0.6], [0.3]], "tau": 1.0})
    assert set(out["estimates"]) == {"ips", "snips", "dm", "dr", "sndr", "switch-dr"}
    assert call(svc, "POST", "/ope", {**body, "estimators": ["magic"]})[0] == 400
    assert call(svc, "POST", "/ope", {**body, "action_dist": [0.5, 0.5]})[0] == 400
    assert call(svc, "POST", "/ope", {**body, "action": "x"})[0] == 400
    assert call(svc, "POST", "/ope", {**body, "pscore": [0, 0, 0, 0]})[0] == 422


def test_metrics_count_requests_and_errors(svc: AuctionService) -> None:
    call(svc, "GET", "/health")
    call(svc, "POST", "/auction", {})
    svc.handle("GET", "/elsewhere")
    status, out = call(svc, "GET", "/metrics")
    text = out["text"]
    assert status == 200
    assert 'aal_requests_total{route="/health"} 1' in text
    assert 'aal_errors_total{route="/auction"} 1' in text
    assert 'aal_requests_total{route="other"} 1' in text
    assert 'aal_request_ms_bucket{route="/health",le="+Inf"} 1' in text


def test_real_socket_round_trip(svc: AuctionService, tmp_path: Path) -> None:
    server = serve(svc, "127.0.0.1", 0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            assert json.loads(r.read())["status"] == "ok"
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/auction",
            data=json.dumps({"bids": [2, 1]}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            assert json.loads(r.read())["slots"][0]["bidder"] == 0
        # Only the headers are sent: the server must refuse before reading a large body.
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.putrequest("POST", "/auction")
        conn.putheader("Content-Length", "2100000")
        conn.endheaders()
        assert conn.getresponse().status == 413
        conn.close()
    finally:
        server.shutdown()
        server.server_close()
