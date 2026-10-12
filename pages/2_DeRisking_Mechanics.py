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

from _shared import run_simulation
from ui.charts import (
    AMUNDI_CYAN,
    AMUNDI_GREEN,
    AMUNDI_NAVY,
    build_glidepath_schedule_chart,
    build_stacked_allocation_chart,
)
from ui.components import build_model_caveats_panel, build_top_controls_header, shared_market_sidebar

st.title("Accumulation & De-Risking Mechanics")
st.caption(
    "How different strategies manage the transition from risky to safe assets "
    "over the accumulation horizon."
)

st.sidebar.title("Global Market & Simulation")

build_top_controls_header(
    "Scenario Inputs",
    "Shared market and simulation assumptions are in the sidebar.",
)
# show_cppi = st.checkbox("Show CPPI controls", value=False)
show_cppi = False

c1, c2, c3 = st.columns(3)
with c1:
    start_age = st.number_input("Starting Age", min_value=18, max_value=60, value=25, step=1)
    retirement_age_default = 63
    retirement_age_min = max(19, int(start_age) + 1)
    retirement_age = st.number_input(
        "Retirement Age",
        min_value=retirement_age_min,
        max_value=95,
        value=min(max(retirement_age_default, retirement_age_min), 95),
        step=1,
    )
with c2:
    initial_wealth = st.number_input(
        "Starting Pot (€)", min_value=0, max_value=10_000_000,
        value=1_000_000, step=50_000, format="%d",
    )
    annual_contribution = st.number_input(
        "Annual Contribution (€)", min_value=0, max_value=10_000_000,
        value=50_000, step=5_000, format="%d",
    )
with c3:
    cm_lambda = st.slider(
    "Risky Allocation (%)", min_value=10, max_value=100, value=60, step=5,
    help="Constant Mix: fixed percentage of portfolio held in risky assets at all times.",
    )
    gp_shape = st.radio(
    "Glidepath Shape",
    options=["linear", "convex", "concave"],
    index=0,
    help=(
        "Linear: constant-rate transition. "
        "Convex: slow de-risking early, sharp near retirement. "
        "Concave: aggressive de-risking early, flattens later."
    ),)
g1, g2, g3 = st.columns(3)
with g1:
    gp_initial = st.slider("Initial Equity (%)", min_value=40, max_value=100, value=80, step=5) / 100.0
with g2:
    gp_final = st.slider("Final Equity (%)", min_value=0, max_value=60, value=20, step=5) / 100.0
with g3:
    pass

time_horizon = int(retirement_age - start_age)

mkt = shared_market_sidebar(context="accumulation", include_cppi=show_cppi)
run_sims = st.button("Run Simulation", type="primary", use_container_width=True, key="dr_run")

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
    block_size=mkt["block_size"],
)

if ("derisking_sims" not in st.session_state) or run_sims:
    st.session_state["derisking_sims"] = (
        run_simulation(**_sim_kw, strategy_type="CM"),
        run_simulation(
            **_sim_kw,
            strategy_type="Glidepath",
            glidepath_initial=gp_initial,
            glidepath_final=gp_final,
            glidepath_shape=gp_shape,
        ),
    )

if "derisking_sims" not in st.session_state:
    st.info("Configure parameters and click **Run Simulation** to begin.")
    st.stop()

sim_cm, sim_gp = st.session_state["derisking_sims"]

n_steps = len(sim_cm["dates"])
age_axis = np.linspace(start_age, retirement_age, n_steps)

st.caption(f"Investor age {start_age} → {retirement_age}.")

st.subheader("De-Risking Visual Comparison")
left_col, right_col = st.columns(2)

with left_col:
    cm_alloc_fig = build_stacked_allocation_chart(
        sim_cm,
        age_axis,
        title=f"Constant Mix; Risky vs Safe ({cm_lambda:.0f}% Risky)",
        height=350,
    )
    cm_alloc_fig.update_xaxes(range=[18, int(retirement_age)])
    st.plotly_chart(
        cm_alloc_fig,
        width='stretch',
    )
    st.caption(
        f"Constant Mix maintains a fixed {cm_lambda:.0f}% allocation to risky assets at every rebalance. "
        "There is no mechanical de-risking in a drawdown, so the strategy is simple but market-indifferent."
    )

with right_col:
    gp_alloc_fig = build_glidepath_schedule_chart(
        initial_equity=gp_initial,
        final_equity=gp_final,
        time_horizon=time_horizon,
        shape=gp_shape,
        height=350,
        start_age=int(start_age),
    )
    gp_alloc_fig.update_xaxes(range=[18, int(retirement_age)])
    st.plotly_chart(gp_alloc_fig, width='stretch')

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

# build_model_caveats_panel()
