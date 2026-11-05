"""Entrypoint: `python -m ad_auction_lab`."""

from __future__ import annotations

from ad_auction_lab import __version__


def main() -> None:
    print(f"ad-auction-lab {__version__}")


if __name__ == "__main__":
    main()
