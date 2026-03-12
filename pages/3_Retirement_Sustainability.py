"""
Page 3 — Retirement Sustainability
====================================
Answers one question: does the plan hold up under retirement withdrawals?
Probability of Success is the hero metric.
"""

import numpy as np
import streamlit as st

st.set_page_config(layout="wide")

from _shared import (
    AMUNDI_GREEN,
    build_fan_chart,
    build_histogram,
    build_model_caveats_panel,
    build_pos_hero,
    build_survival_comparison,
    build_survival_curve,
    run_simulation,
    shared_market_sidebar,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Sustainability Setup")

st.sidebar.header("Portfolio & Horizon")
initial_wealth = st.sidebar.number_input(
    "Initial Retirement Wealth (€)", min_value=10_000, max_value=10_000_000,
    value=1_000_000, step=50_000, format="%d",
)
annual_withdrawal = st.sidebar.number_input(
    "Annual Withdrawal (€)", min_value=0, max_value=10_000_000,
    value=40_000, step=5_000, format="%d",
)
time_horizon = st.sidebar.slider(
    "Withdrawal Horizon (Years)", min_value=1, max_value=40, value=20,
)
floor_pct = st.sidebar.slider(
    "Guaranteed Floor (%)", min_value=50, max_value=100, value=80, step=5,
    help="Minimum percentage of remaining withdrawal commitments to protect.",
)

mkt = shared_market_sidebar()

# ---------------------------------------------------------------------------
# Simulations
# ---------------------------------------------------------------------------
sim = run_simulation(
    initial_wealth=float(initial_wealth),
    time_horizon=time_horizon,
    cppi_multiplier=mkt["cppi_multiplier"],
    floor_pct=float(floor_pct),
    expected_return=mkt["expected_return"],
    market_volatility=mkt["market_volatility"],
    risk_free_rate=mkt["risk_free_rate"],
    n_simulations=mkt["n_simulations"],
    rebalance_freq=mkt["rebalance_freq"],
    annual_withdrawal=float(annual_withdrawal),
    annual_contribution=0.0,
    simulation_method=mkt["simulation_method"],
    block_length=mkt["block_length"],
    strategy_type="CPPI",
)

# Constant Mix baseline for survival comparison
sim_cm = run_simulation(
    initial_wealth=float(initial_wealth),
    time_horizon=time_horizon,
    cppi_multiplier=mkt["cppi_multiplier"],
    floor_pct=float(floor_pct),
    expected_return=mkt["expected_return"],
    market_volatility=mkt["market_volatility"],
    risk_free_rate=mkt["risk_free_rate"],
    n_simulations=mkt["n_simulations"],
    rebalance_freq=mkt["rebalance_freq"],
    annual_withdrawal=float(annual_withdrawal),
    annual_contribution=0.0,
    simulation_method=mkt["simulation_method"],
    block_length=mkt["block_length"],
    strategy_type="CM",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Retirement Sustainability")
st.caption(
    "Does the plan hold up under retirement withdrawals? "
    f"{mkt['n_simulations']:,} Monte-Carlo paths · "
    f"{time_horizon}-year withdrawal horizon"
)

# ---------------------------------------------------------------------------
# KPIs — PoS hero + survival-focused metrics
# ---------------------------------------------------------------------------
hero_col, k2, k3, k4, k5 = st.columns([1.4, 1, 1, 1, 1])

with hero_col:
    build_pos_hero(sim["prob_success"])

survival_times = sim["survival_times"]
avg_survival = float(np.mean(survival_times))
p10_survival = float(np.percentile(survival_times, 10))

k2.metric("Avg Survival Time", f"{avg_survival:.1f} yrs")
k3.metric("10th Pctl Survival", f"{p10_survival:.1f} yrs")
k4.metric("Expected Shortfall", f"€ {sim['expected_shortfall']:,.0f}")
k5.metric("Median Ending Wealth", f"€ {sim['median_ending']:,.0f}")

st.divider()

# ---------------------------------------------------------------------------
# Decumulation Fan Chart
# ---------------------------------------------------------------------------
st.plotly_chart(
    build_fan_chart(
        sim,
        title="Decumulation — Monte-Carlo Fan Chart",
        floor_label="Guaranteed Floor",
        band_color="0,122,51",
        median_color=AMUNDI_GREEN,
    ),
    width='stretch',
)

# ---------------------------------------------------------------------------
# Survival Curve + Survival Comparison
# ---------------------------------------------------------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.plotly_chart(
        build_survival_curve(
            sim, sim_cm,
            time_horizon=time_horizon,
            title="Survival Probability by Retirement Year",
        ),
        width='stretch',
    )

with col_right:
    st.plotly_chart(
        build_survival_comparison(
            sim, sim_cm,
            title="Portfolio Survival — CPPI vs Constant Mix",
            time_horizon=time_horizon,
        ),
        width='stretch',
    )

st.plotly_chart(
    build_histogram(sim, title="Ending Wealth Distribution", color=AMUNDI_GREEN),
    width='stretch',
)

# ---------------------------------------------------------------------------
# Caveats
# ---------------------------------------------------------------------------
build_model_caveats_panel()
