"""Command line interface, end to end on small on-disk fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ad_auction_lab.cli import build_parser, main
from tests.fixtures import write_all


@pytest.fixture(scope="module")
def data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("data")
    write_all(root)
    return root


def test_auction_table(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["auction", "--bids", "4,2,1", "--slots", "0.2,0.1"]) == 0
    out = capsys.readouterr().out
    assert "gsp" in out and "vcg" in out and "first_price" in out
    assert "0.4000" in out  # VCG revenue


def test_auction_json(capsys: pytest.CaptureFixture[str]) -> None:
    main(["auction", "--bids", "4,2,1", "--mechanism", "vcg", "--slots", "0.2,0.1", "--json"])
    rows = json.loads(capsys.readouterr().out)
    assert rows == [
        {"mechanism": "vcg", "winners": [0, 1], "price_per_click": [1.5, 1.0], "revenue": 0.4}
    ]


def test_auction_quality_mismatch(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["auction", "--bids", "1,2", "--quality", "1"]) == 2
    assert "one value per bid" in capsys.readouterr().err


def test_bad_number_list() -> None:
    with pytest.raises(SystemExit):
        main(["auction", "--bids", "1,x"])


def test_incentives(capsys: pytest.CaptureFixture[str]) -> None:
    main(["incentives", "--auctions", "200", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert out["vcg"]["share_with_profitable_deviation"] == 0.0
    assert out["first_price"]["share_with_profitable_deviation"] > 0.3
    main(["incentives", "--auctions", "100"])
    assert "gain/welfare" in capsys.readouterr().out


def test_shade(data: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["shade", "--data", str(data), "--sample", "20000"]) == 0
    out = capsys.readouterr().out
    assert "bid (empirical)" in out and len(out.splitlines()) == 7
    main(["shade", "--data", str(data), "--values", "0.001"])
    assert "1.000e-03" in capsys.readouterr().out


def test_missing_data(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="make data"):
        main(["shade", "--data", str(tmp_path)])


def test_pace(data: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["pace", "--data", str(data), "--horizon", "20000", "--advertisers", "3"]) == 0
    out = capsys.readouterr().out
    for name in ("none", "pid", "dual"):
        assert f"\n{name} " in out


def test_train_ctr_and_serve_model(
    data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = tmp_path / "m.npz"
    args = ["train-ctr", "--data", str(data), "--bits", "12", "--out", str(model)]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "logloss" in out and model.exists()
    args = ["train-ctr", "--data", str(data), "--model", "fm", "--k", "4", "--bits", "10"]
    assert main([*args, "--train-rows", "500", "--test-rows", "200", "--epochs", "2"]) == 0
    assert "test rows 200" in capsys.readouterr().out


def test_ope(data: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ope", "--data", str(data), "--boot", "20"]) == 0
    out = capsys.readouterr().out
    assert "switch-dr" in out and "on-policy truth" in out
    assert main(["ope", "--data", str(data), "--boot", "20", "--target", "random"]) == 0


def test_serve_starts_and_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    import ad_auction_lab.service as service

    calls: list[str] = []

    class Fake:
        server_address = ("127.0.0.1", 1234)

        def serve_forever(self) -> None:
            calls.append("serve")

        def server_close(self) -> None:
            calls.append("close")

    monkeypatch.setattr(service, "serve", lambda *a, **k: Fake())
    assert main(["serve", "--port", "0"]) == 0
    assert calls == ["serve", "close"]


def test_parser_lists_every_command() -> None:
    text = build_parser().format_help()
    for cmd in ("auction", "incentives", "shade", "pace", "train-ctr", "ope", "serve"):
        assert cmd in text


def test_module_entrypoint() -> None:
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-m", "ad_auction_lab", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.startswith("ad-auction-lab ")
