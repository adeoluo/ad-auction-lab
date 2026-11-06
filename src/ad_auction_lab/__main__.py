"""Entrypoint: `python -m ad_auction_lab`."""

from __future__ import annotations

import sys

from ad_auction_lab.cli import main

if __name__ == "__main__":
    sys.exit(main())
