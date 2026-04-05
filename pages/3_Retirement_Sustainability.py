"""
Page 3 — Retirement Sustainability
====================================
Answers one question: does the plan hold up under retirement withdrawals?
Probability of Success is the hero metric.
"""

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")

from _shared import run_simulation
from ui.charts import (
    AMUNDI_GREEN,
    build_fan_chart,
    build_survival_comparison,
    build_survival_curve,
)
from ui.components import (
    build_model_caveats_panel,
    build_pos_hero,
    format_delta_metric,
    shared_market_sidebar,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Sustainability Setup")

show_cppi = st.sidebar.checkbox("Show CPPI controls", value=False)

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
    "Withdrawal Horizon (Years)", min_value=1, max_value=40, value=30,
)

st.sidebar.header("Retirement Strategy")
lambda_pct = st.sidebar.slider(
    "Constant Mix Risky Allocation (%)", min_value=30, max_value=90, value=60, step=5,
)
gp_initial = st.sidebar.slider(
    "Glidepath Initial Equity (%)", min_value=20, max_value=90, value=60, step=5,
) / 100.0
gp_final = st.sidebar.slider(
    "Glidepath Final Equity (%)", min_value=0, max_value=80, value=30, step=5,
) / 100.0
gp_shape = st.sidebar.radio("Glidepath Shape", options=["linear", "convex", "concave"], index=0)

mkt = shared_market_sidebar(context="retirement", include_cppi=show_cppi)
run_sims = st.sidebar.button("Run Simulation", type="primary", use_container_width=True, key="ret_run")

# ---------------------------------------------------------------------------
# Simulations
# ---------------------------------------------------------------------------
common = dict(
    initial_wealth=float(initial_wealth),
    time_horizon=time_horizon,
    cppi_multiplier=mkt["cppi_multiplier"],
    floor_pct=0.0,
    expected_return=mkt["expected_return"],
    market_volatility=mkt["market_volatility"],
    risk_free_rate=mkt["risk_free_rate"],
    n_simulations=mkt["n_simulations"],
    rebalance_freq=mkt["rebalance_freq"],
    annual_withdrawal=float(annual_withdrawal),
    annual_contribution=0.0,
    simulation_method=mkt["simulation_method"],
    block_length=mkt["block_length"],
)

if ("retirement_sims" not in st.session_state) or run_sims:
    st.session_state["retirement_sims"] = (
        run_simulation(
            lambda_pct=float(lambda_pct),
            strategy_type="Glidepath",
            glidepath_initial=gp_initial,
            glidepath_final=gp_final,
            glidepath_shape=gp_shape,
            **common,
        ),
        run_simulation(
            lambda_pct=float(lambda_pct),
            strategy_type="CM",
            **common,
        ),
    )

if "retirement_sims" not in st.session_state:
    st.title("Retirement Sustainability")
    st.info("Configure parameters in the sidebar, then click **Run Simulation** to begin.")
    st.stop()

sim, sim_cm = st.session_state["retirement_sims"]


def _strategy_summary(label: str, sim_data: dict) -> dict[str, float | str]:
    survival_times = np.asarray(sim_data["survival_times"], dtype=float)
    return {
        "Strategy": label,
        "PoS (%)": round(float(sim_data["prob_success"]), 1),
        "Avg survival (yrs)": round(float(np.mean(survival_times)), 1),
        "10th pct survival (yrs)": round(float(np.percentile(survival_times, 10)), 1),
        "Expected shortfall (€)": round(float(sim_data["expected_shortfall"]), 0),
        "Median ending wealth (€)": round(float(sim_data["median_ending"]), 0),
        "P5 ending wealth (€)": round(float(np.percentile(sim_data["ending_wealth"], 5)), 0),
    }


comparison_df = pd.DataFrame([
    _strategy_summary("Glidepath", sim),
    _strategy_summary("Constant Mix", sim_cm),
]).set_index("Strategy")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Retirement Sustainability")
st.caption(
    "Does the plan hold up under retirement withdrawals under Glidepath vs Constant Mix? "
    f"{mkt['n_simulations']:,} Monte-Carlo paths · "
    f"{time_horizon}-year withdrawal horizon"
)

st.caption(
    "Success = terminal wealth remains above the model floor at horizon. "
    "Expected shortfall = average cumulative missed withdrawals among failing paths. "
    "Survival = years funded before depletion or floor breach."
)

# ---------------------------------------------------------------------------
# KPIs — PoS hero + survival-focused metrics
# ---------------------------------------------------------------------------
hero_col, k2, k3, k4, k5 = st.columns([1.4, 1, 1, 1, 1])

with hero_col:
    build_pos_hero(sim["prob_success"])

survival_times = np.asarray(sim["survival_times"], dtype=float)
avg_survival = float(np.mean(survival_times))
p10_survival = float(np.percentile(survival_times, 10))

baseline = st.session_state.get("baseline_cm_result")
baseline_pos = float(baseline["prob_success"]) if baseline is not None else float(sim_cm["prob_success"])
baseline_shortfall = float(baseline["expected_shortfall"]) if baseline is not None else float(sim_cm["expected_shortfall"])
baseline_median = float(baseline["median_ending"]) if baseline is not None else float(sim_cm["median_ending"])

surv_delta = avg_survival - float(np.mean(np.asarray(sim_cm["survival_times"], dtype=float)))
p10_delta = p10_survival - float(np.percentile(np.asarray(sim_cm["survival_times"], dtype=float), 10))
k2.metric("Avg Survival Time", f"{avg_survival:.1f} yrs", delta=f"{surv_delta:+.1f} yrs vs CM")
k3.metric("10th Pctl Survival", f"{p10_survival:.1f} yrs", delta=f"{p10_delta:+.1f} yrs vs CM")

k4_value, k4_delta = format_delta_metric(float(sim['expected_shortfall']), baseline_shortfall, currency=True, inverse=True)
k4.metric("Expected Shortfall", k4_value, delta=k4_delta, delta_color="inverse")

k5_value, k5_delta = format_delta_metric(float(sim['median_ending']), baseline_median, currency=True)
k5.metric("Median Ending Wealth", k5_value, delta=k5_delta)

st.divider()

# ---------------------------------------------------------------------------
# Decumulation Fan Chart
# ---------------------------------------------------------------------------
st.plotly_chart(
    build_fan_chart(
        sim,
        title="Decumulation — Monte-Carlo Fan Chart",
        floor_label="Reference Floor",
        band_color="0,122,51",
        median_color=AMUNDI_GREEN,
        median_label="Median Glidepath Portfolio",
        show_risky_mean=False,
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
            primary_label="Glidepath",
            secondary_label="Constant Mix",
        ),
        width='stretch',
    )

with col_right:
    st.plotly_chart(
        build_survival_comparison(
            sim, sim_cm,
            title="Portfolio Survival — Glidepath vs Constant Mix",
            time_horizon=time_horizon,
            primary_label="Glidepath",
            secondary_label="Constant Mix",
        ),
        width='stretch',
    )

st.subheader("Comparison Table — Outcome Quality")
st.dataframe(
    comparison_df.style.format({
        "PoS (%)": "{:.1f}",
        "Avg survival (yrs)": "{:.1f}",
        "10th pct survival (yrs)": "{:.1f}",
        "Expected shortfall (€)": "€ {:,.0f}",
        "Median ending wealth (€)": "€ {:,.0f}",
        "P5 ending wealth (€)": "€ {:,.0f}",
    }),
    width='stretch',
)

# ---------------------------------------------------------------------------
# Caveats
# ---------------------------------------------------------------------------
build_model_caveats_panel()
