"""
Page 2 — Accumulation Strategy Design
=====================================
Compare how accumulation strategies invest before retirement.
"""

import streamlit as st

st.set_page_config(layout="wide")

from _shared import (
    build_allocation_comparison_by_age,
    build_glidepath_schedule_chart,
    build_model_caveats_panel,
    build_terminal_wealth_boxplot,
    run_simulation,
    shared_market_sidebar,
)

st.sidebar.title("Accumulation Setup")
st.sidebar.header("Portfolio & Horizon")
start_age = st.sidebar.number_input("Current Age", min_value=20, max_value=70, value=35, step=1)
initial_wealth = st.sidebar.number_input(
    "Starting Pot (€)", min_value=0, max_value=10_000_000,
    value=50_000, step=5_000, format="%d",
)
annual_contribution = st.sidebar.number_input(
    "Annual Contribution (€)", min_value=0, max_value=10_000_000,
    value=12_000, step=1_000, format="%d",
)
time_horizon = st.sidebar.slider("Savings Horizon (Years)", min_value=1, max_value=40, value=30)
floor_pct = st.sidebar.slider("CPPI Floor (%)", min_value=50, max_value=100, value=80, step=5)

mkt = shared_market_sidebar()

st.sidebar.header("Glidepath Parameters")
gp_initial = st.sidebar.slider("Initial Equity (%)", min_value=40, max_value=100, value=80, step=5) / 100.0
gp_final = st.sidebar.slider("Final Equity (%)", min_value=0, max_value=60, value=20, step=5) / 100.0
gp_shape = st.sidebar.radio("Glidepath Shape", options=["linear", "convex", "concave"], index=0)

common = dict(
    initial_wealth=float(initial_wealth),
    time_horizon=time_horizon,
    cppi_multiplier=mkt["cppi_multiplier"],
    floor_pct=float(floor_pct),
    expected_return=mkt["expected_return"],
    market_volatility=mkt["market_volatility"],
    risk_free_rate=mkt["risk_free_rate"],
    n_simulations=mkt["n_simulations"],
    rebalance_freq=mkt["rebalance_freq"],
    annual_withdrawal=0.0,
    annual_contribution=float(annual_contribution),
    simulation_method=mkt["simulation_method"],
    block_length=mkt["block_length"],
)

sim_cppi = run_simulation(strategy_type="CPPI", **common)
sim_glide = run_simulation(
    strategy_type="Glidepath",
    glidepath_initial=gp_initial,
    glidepath_final=gp_final,
    glidepath_shape=gp_shape,
    **common,
)
sim_cm = run_simulation(strategy_type="CM", **common)

st.title("Accumulation Strategy Design")
st.caption(
    "This page only covers the saving phase. It compares how different accumulation strategies "
    "allocate risk before retirement; it does not describe retirement-income behavior."
)

st.plotly_chart(
    build_allocation_comparison_by_age(
        {
            "CPPI": sim_cppi,
            "Glidepath": sim_glide,
            "Constant Mix": sim_cm,
        },
        start_age=int(start_age),
        rebalance_freq=mkt["rebalance_freq"],
        title="Risky Allocation by Investor Age",
    ),
    width='stretch',
)

st.plotly_chart(
    build_glidepath_schedule_chart(
        initial_equity=gp_initial,
        final_equity=gp_final,
        time_horizon=time_horizon,
        shape=gp_shape,
    ),
    width='stretch',
)

st.plotly_chart(
    build_terminal_wealth_boxplot(
        {
            "CPPI": sim_cppi["ending_wealth"],
            "Glidepath": sim_glide["ending_wealth"],
            "Constant Mix": sim_cm["ending_wealth"],
        }
    ),
    width='stretch',
)

st.caption(
    "The contribution schedule is identical across strategies on this page. The meaningful differences are "
    "allocation path and terminal wealth distribution, not how much cash is contributed. Values are displayed in nominal (€)."
)

build_model_caveats_panel()
