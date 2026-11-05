from __future__ import annotations

import pytest

from ad_auction_lab.data import Table, synthetic_clicks


@pytest.fixture(scope="session")
def clicks() -> Table:
    return synthetic_clicks(6000, seed=0)


@pytest.fixture(scope="session")
def small_clicks() -> Table:
    return synthetic_clicks(300, n_small=3, n_wide=4, seed=1)
