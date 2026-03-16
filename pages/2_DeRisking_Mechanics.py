"""
Page 2; Accumulation & De-Risking Mechanics
==============================================
Illustrates how two accumulation strategies change risky exposure over time.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(layout="wide")

from _shared import (
    AMUNDI_CYAN,
    AMUNDI_GREEN,
    AMUNDI_NAVY,
    build_glidepath_schedule_chart,
    build_model_caveats_panel,
    build_stacked_allocation_chart,
    run_simulation,
    shared_market_sidebar,
)

st.sidebar.title("Accumulation Setup")

show_cppi = st.sidebar.checkbox("Show CPPI controls", value=False)

st.sidebar.header("Portfolio & Horizon")
start_age = st.sidebar.number_input("Current Age", min_value=20, max_value=60, value=25, step=1)
initial_wealth = st.sidebar.number_input(
    "Starting Pot (€)", min_value=0, max_value=10_000_000,
    value=1_000_000, step=50_000, format="%d",
)
annual_contribution = st.sidebar.number_input(
    "Annual Contribution (€)", min_value=0, max_value=10_000_000,
    value=50_000, step=5_000, format="%d",
)
time_horizon = st.sidebar.slider("Savings Horizon (Years)", min_value=1, max_value=40, value=40)

mkt = shared_market_sidebar(context="accumulation", include_cppi=show_cppi)

st.sidebar.header("Constant Mix Parameters")
cm_lambda = st.sidebar.slider(
    "Risky Allocation (%)", min_value=10, max_value=100, value=60, step=5,
    help="Constant Mix: fixed percentage of portfolio held in risky assets at all times.",
)

st.sidebar.header("Glidepath Parameters")
gp_initial = st.sidebar.slider("Initial Equity (%)", min_value=40, max_value=100, value=80, step=5) / 100.0
gp_final = st.sidebar.slider("Final Equity (%)", min_value=0, max_value=60, value=20, step=5) / 100.0
gp_shape = st.sidebar.radio(
    "Glidepath Shape",
    options=["linear", "convex", "concave"],
    index=0,
    help=(
        "Linear: constant-rate transition. "
        "Convex: slow de-risking early, sharp near retirement. "
        "Concave: aggressive de-risking early, flattens later."
    ),
)

_sim_kw = dict(
    initial_wealth=float(initial_wealth),
    time_horizon=time_horizon,
    cppi_multiplier=mkt["cppi_multiplier"],
    floor_pct=0.0,
    expected_return=mkt["expected_return"],
    market_volatility=mkt["market_volatility"],
    risk_free_rate=mkt["risk_free_rate"],
    n_simulations=mkt["n_simulations"],
    rebalance_freq=mkt["rebalance_freq"],
    annual_withdrawal=0.0,
    annual_contribution=float(annual_contribution),
    lambda_pct=float(cm_lambda),
    simulation_method=mkt["simulation_method"],
    block_length=mkt["block_length"],
)

sim_cm = run_simulation(**_sim_kw, strategy_type="CM")
sim_gp = run_simulation(
    **_sim_kw,
    strategy_type="Glidepath",
    glidepath_initial=gp_initial,
    glidepath_final=gp_final,
    glidepath_shape=gp_shape,
)

n_steps = len(sim_cm["dates"])
retirement_age = start_age + time_horizon
age_axis = np.linspace(start_age, retirement_age, n_steps)

st.title("Accumulation & De-Risking Mechanics")
st.caption(
    "How different strategies manage the transition from risky to safe assets "
    f"over the accumulation horizon. Investor age {start_age} → {retirement_age}."
)

st.subheader("Constant Mix: Fixed Risky Allocation")
st.plotly_chart(
    build_stacked_allocation_chart(
        sim_cm,
        age_axis,
        title=f"Constant Mix Accumulation; Risky vs Safe Over Time ({cm_lambda:.0f}% Risky)",
    ),
    width='stretch',
)
st.caption(
    f"Constant Mix maintains a fixed {cm_lambda:.0f}% allocation to risky assets at every rebalance. "
    "There is no mechanical de-risking in a drawdown, so the strategy is simple but market-indifferent."
)

st.divider()
st.subheader("Glidepath: Deterministic De-Risking")
st.plotly_chart(
    build_glidepath_schedule_chart(
        initial_equity=gp_initial,
        final_equity=gp_final,
        time_horizon=time_horizon,
        shape=gp_shape,
    ),
    width='stretch',
)
st.caption(
    "The glidepath schedule is fixed at construction time. It provides a transparent de-risking rule, "
    "but does not react to realized market stress or improvements in funding status."
)

# Replace low-value cumulative-contribution chart with a summary table
st.divider()
st.subheader("Comparison Table — What Actually Changes")


def _summary(label: str, sim_data: dict) -> dict[str, float | str]:
    alloc = np.asarray(sim_data["allocation_percentiles"]["P50"], dtype=float)
    ending = np.asarray(sim_data["ending_wealth"], dtype=float)
    return {
        "Strategy": label,
        "Avg risky allocation (%)": float(np.mean(alloc)),
        "Final risky allocation (%)": float(alloc[-1]),
        "Median terminal wealth (€)": float(np.median(ending)),
        "P5 terminal wealth (€)": float(np.percentile(ending, 5)),
    }


summary_df = pd.DataFrame([
    _summary("Constant Mix", sim_cm),
    _summary("Glidepath", sim_gp),
]).set_index("Strategy")

st.dataframe(
    summary_df.style.format({
        "Avg risky allocation (%)": "{:.1f}",
        "Final risky allocation (%)": "{:.1f}",
        "Median terminal wealth (€)": "€ {:,.0f}",
        "P5 terminal wealth (€)": "€ {:,.0f}",
    }),
    width='stretch',
)

st.caption(
    "This replaces the cumulative-contributions chart because contributions are deterministic and identical across strategies. "
    "The decision-relevant differences are exposure path and downside distribution."
)

build_model_caveats_panel()
