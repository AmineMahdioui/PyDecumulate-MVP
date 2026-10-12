from __future__ import annotations

import streamlit as st
import pytest


@pytest.fixture(autouse=True)
def clear_streamlit_cache() -> None:
    # Keep cached wrapper calls isolated between tests.
    st.cache_data.clear()


@pytest.fixture
def base_simulation_kwargs() -> dict[str, float | int | str]:
    return {
        "initial_wealth": 100_000.0,
        "time_horizon": 3,
        "cppi_multiplier": 3.0,
        "floor_pct": 80.0,
        "expected_return": 6.0,
        "market_volatility": 12.0,
        "risk_free_rate": 2.0,
        "n_simulations": 32,
        "rebalance_freq": "monthly",
        "annual_withdrawal": 0.0,
        "annual_contribution": 12_000.0,
        "lambda_pct": 60.0,
        "simulation_method": "GBM (Parametric)",
        "block_size": 1,
    }
