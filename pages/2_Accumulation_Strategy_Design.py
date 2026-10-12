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
    build_age_in_bonds_schedule_chart,
    build_allocation_comparison_by_age,
    build_glidepath_schedule_chart,
    build_terminal_wealth_boxplot,
)
from ui.components import (
    build_top_controls_header,
    build_model_caveats_panel,
    format_delta_metric,
    shared_market_sidebar,
)

st.title("Accumulation Strategy Design")
st.caption(
    "This page only covers the saving phase. It compares how different accumulation strategies "
    "allocate risk before retirement; it does not describe retirement-income behavior."
)

st.sidebar.title("Global Market & Simulation")

build_top_controls_header(
    "Strategy Inputs",
    "",
)
show_cppi = st.checkbox("Show CPPI controls", value=False)

c1, c2, c3 = st.columns(3)
with c1:
    start_age = st.number_input("Starting Age", min_value=18, max_value=70, value=25, step=1)
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
        value=5_000, step=50_000, format="%d",
    )
    annual_contribution = st.number_input(
        "Annual Contribution (€)", min_value=0, max_value=1_000_000,
        value=12_000, step=5_000, format="%d",
    )
with c3:
    floor_pct = st.slider("CPPI Floor (%)", min_value=50, max_value=100, value=80, step=5)
    gp_shape = st.radio("Glidepath Shape", options=["linear", "convex", "concave"], index=1)

time_horizon = int(retirement_age - start_age)

g1, g2 = st.columns(2)
with g1:
    gp_initial = st.slider("Initial Equity (%)", min_value=40, max_value=100, value=80, step=5) / 100.0
with g2:
    gp_final = st.slider("Final Equity (%)", min_value=0, max_value=60, value=20, step=5) / 100.0

mkt = shared_market_sidebar(context="accumulation", include_cppi=show_cppi)
run_sims = st.button("Run Strategies", type="primary", use_container_width=True, key="acc_run")

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
    block_size=mkt["block_size"],
)

if (
    ("acc_sims" not in st.session_state)
    or run_sims
    or len(st.session_state.get("acc_sims", ())) != 4
):
    st.session_state["acc_sims"] = (
        run_simulation(strategy_type="CPPI", **common),
        run_simulation(
            strategy_type="Glidepath",
            glidepath_initial=gp_initial,
            glidepath_final=gp_final,
            glidepath_shape=gp_shape,
            **common,
        ),
        run_simulation(
            strategy_type="AgeInBonds",
            accumulation_start_age=int(start_age),
            **common,
        ),
        run_simulation(strategy_type="CM", **common),
    )

if "acc_sims" not in st.session_state:
    st.info("Configure parameters and click **Run Strategies** to begin.")
    st.stop()

sim_cppi, sim_glide, sim_aib, sim_cm = st.session_state["acc_sims"]

baseline_med = float(np.median(sim_aib["ending_wealth"]))

k1, k2, k3, k4 = st.columns(4)
cp_med = float(np.median(sim_cppi["ending_wealth"]))
gp_med = float(np.median(sim_glide["ending_wealth"]))
aib_med = float(np.median(sim_aib["ending_wealth"]))
cm_med = float(np.median(sim_cm["ending_wealth"]))

v1, d1, c1_delta = format_delta_metric(cp_med, baseline_med, currency=True)
v2, d2, c2_delta = format_delta_metric(gp_med, baseline_med, currency=True)
v3, d3, c3_delta = format_delta_metric(aib_med, baseline_med, currency=True)
v4, d4, c4_delta = format_delta_metric(cm_med, baseline_med, currency=True)

k1.metric("CPPI Median Terminal Wealth", v1, delta=d1, delta_color=c1_delta)
k2.metric("Glidepath Median Terminal Wealth", v2, delta=d2, delta_color=c2_delta)
k3.metric("Age-in-Bonds Median Terminal Wealth", v3, delta=d3, delta_color=c3_delta)
k4.metric("Constant Mix Median Terminal Wealth", v4, delta=d4, delta_color=c4_delta)

st.divider()

alloc_fig = build_allocation_comparison_by_age(
    {
        "CPPI": sim_cppi,
        "Glidepath": sim_glide,
        "Age-in-Bonds": sim_aib,
        "Constant Mix": sim_cm,
    },
    start_age=int(start_age),
    rebalance_freq=mkt["rebalance_freq"],
    title="Risky Allocation by Investor Age",
)
alloc_fig.update_xaxes(range=[18, int(retirement_age)])
st.plotly_chart(alloc_fig, width='stretch')


st.plotly_chart(
    build_terminal_wealth_boxplot(
        {
            "CPPI": sim_cppi["ending_wealth"],
            "Glidepath": sim_glide["ending_wealth"],
            "Age-in-Bonds": sim_aib["ending_wealth"],
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
    _acc_summary("Age-in-Bonds", sim_aib),
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

# build_model_caveats_panel()
