"""Entrypoint: `python -m ad_auction_lab`."""

from __future__ import annotations

from ad_auction_lab import __version__
from ad_auction_lab.configuration import settings


def main() -> None:
    header = f"Ad Auction Lab v{__version__}"
    status = f"environment={settings.environment}, seed={settings.seed}"
    print(f"{header} - {status}")


if __name__ == "__main__":
    main()
