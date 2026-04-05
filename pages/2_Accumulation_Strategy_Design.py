"""
Page 2 — Accumulation Strategy Design
=====================================
Compare how accumulation strategies invest before retirement.
"""

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")

from _shared import run_simulation
from ui.charts import (
    build_allocation_comparison_by_age,
    build_glidepath_schedule_chart,
    build_terminal_wealth_boxplot,
)
from ui.components import (
    build_model_caveats_panel,
    format_delta_metric,
    shared_market_sidebar,
)

st.sidebar.title("Accumulation Setup")

show_cppi = st.sidebar.checkbox("Show CPPI controls", value=False)

st.sidebar.header("Portfolio & Horizon")
start_age = st.sidebar.number_input("Current Age", min_value=20, max_value=70, value=25, step=1)
initial_wealth = st.sidebar.number_input(
    "Starting Pot (€)", min_value=0, max_value=10_000_000,
    value=5_000, step=50_000, format="%d",
)
annual_contribution = st.sidebar.number_input(
    "Annual Contribution (€)", min_value=0, max_value=1_000_000,
    value=12_000, step=5_000, format="%d",
)
time_horizon = st.sidebar.slider("Savings Horizon (Years)", min_value=1, max_value=40, value=40)
floor_pct = st.sidebar.slider("CPPI Floor (%)", min_value=50, max_value=100, value=80, step=5)

mkt = shared_market_sidebar(context="accumulation", include_cppi=show_cppi)

st.sidebar.header("Glidepath Parameters")
gp_initial = st.sidebar.slider("Initial Equity (%)", min_value=40, max_value=100, value=80, step=5) / 100.0
gp_final = st.sidebar.slider("Final Equity (%)", min_value=0, max_value=60, value=20, step=5) / 100.0
gp_shape = st.sidebar.radio("Glidepath Shape", options=["linear", "convex", "concave"], index=1)
run_sims = st.sidebar.button("Run Strategies", type="primary", use_container_width=True, key="acc_run")

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

if ("acc_sims" not in st.session_state) or run_sims:
    st.session_state["acc_sims"] = (
        run_simulation(strategy_type="CPPI", **common),
        run_simulation(
            strategy_type="Glidepath",
            glidepath_initial=gp_initial,
            glidepath_final=gp_final,
            glidepath_shape=gp_shape,
            **common,
        ),
        run_simulation(strategy_type="CM", **common),
    )

if "acc_sims" not in st.session_state:
    st.title("Accumulation Strategy Design")
    st.info("Configure parameters in the sidebar, then click **Run Strategies** to begin.")
    st.stop()

sim_cppi, sim_glide, sim_cm = st.session_state["acc_sims"]

st.title("Accumulation Strategy Design")
st.caption(
    "This page only covers the saving phase. It compares how different accumulation strategies "
    "allocate risk before retirement; it does not describe retirement-income behavior."
)

baseline = st.session_state.get("baseline_cm_result")
baseline_med = float(baseline["median_ending"]) if baseline is not None else float(np.median(sim_cm["ending_wealth"]))

k1, k2, k3 = st.columns(3)
cp_med = float(np.median(sim_cppi["ending_wealth"]))
gp_med = float(np.median(sim_glide["ending_wealth"]))
cm_med = float(np.median(sim_cm["ending_wealth"]))

v1, d1 = format_delta_metric(cp_med, baseline_med, currency=True)
v2, d2 = format_delta_metric(gp_med, baseline_med, currency=True)
v3, d3 = format_delta_metric(cm_med, baseline_med, currency=True)

k1.metric("CPPI Median Terminal Wealth", v1, delta=d1)
k2.metric("Glidepath Median Terminal Wealth", v2, delta=d2)
k3.metric("Constant Mix Median Terminal Wealth", v3, delta=d3)

st.divider()

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

contributed_capital = float(initial_wealth + annual_contribution * time_horizon)


def _acc_summary(label: str, sim_data: dict) -> dict[str, float | str]:
    ending = np.asarray(sim_data["ending_wealth"], dtype=float)
    return {
        "Strategy": label,
        "Median terminal wealth (€)": float(np.median(ending)),
        "P5 terminal wealth (€)": float(np.percentile(ending, 5)),
        "P95 terminal wealth (€)": float(np.percentile(ending, 95)),
        "Prob below contributions (%)": float(np.mean(ending < contributed_capital) * 100.0),
    }


comparison_df = pd.DataFrame([
    _acc_summary("CPPI", sim_cppi),
    _acc_summary("Glidepath", sim_glide),
    _acc_summary("Constant Mix", sim_cm),
]).set_index("Strategy")

st.subheader("Comparison Table — Upside and Downside")
st.caption(
    "This is the missing context for the box plot. A strategy is not better only because it has higher outliers; "
    "it also needs acceptable downside and a reasonable chance of ending above contributed capital."
)
st.dataframe(
    comparison_df.style.format({
        "Median terminal wealth (€)": "€ {:,.0f}",
        "P5 terminal wealth (€)": "€ {:,.0f}",
        "P95 terminal wealth (€)": "€ {:,.0f}",
        "Prob below contributions (%)": "{:.1f}",
    }),
    width='stretch',
)

st.caption(
    "The contribution schedule is identical across strategies on this page. The meaningful differences are "
    "allocation path and terminal wealth distribution, not how much cash is contributed. Values are displayed in nominal (€)."
)

build_model_caveats_panel()
