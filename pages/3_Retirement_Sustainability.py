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
    build_age_in_bonds_schedule_chart,
    build_fan_chart,
    build_survival_comparison,
    build_survival_curve,
)
from ui.components import (
    build_model_caveats_panel,
    build_pos_hero,
    build_top_controls_header,
    format_delta_metric,
    shared_market_sidebar,
)

st.title("Retirement Sustainability")

st.sidebar.title("Global Market & Simulation")

build_top_controls_header(
    "Scenario Inputs",
    "",
)
show_cppi = st.checkbox("Show CPPI controls", value=False)

c1, c2, c3 = st.columns(3)
with c1:
    initial_wealth = st.number_input(
        "Initial Retirement Wealth (€)", min_value=10_000, max_value=10_000_000,
        value=1_000_000, step=50_000, format="%d",
    )
    annual_withdrawal = st.number_input(
        "Annual Withdrawal (€)", min_value=0, max_value=10_000_000,
        value=40_000, step=5_000, format="%d",
    )
with c2:
    time_horizon = st.slider(
        "Withdrawal Horizon (Years)", min_value=1, max_value=40, value=30,
    )
    primary_strategy = st.radio(
        "Primary Strategy",
        options=["Glidepath", "Age-in-Bonds"],
        index=1,
    )
with c3:
    lambda_pct = st.slider(
        "Constant Mix Risky Allocation (%)", min_value=30, max_value=90, value=60, step=5,
    )
    retirement_start_age = st.number_input(
        "Retirement Start Age",
        min_value=40,
        max_value=90,
        value=64,
        step=1,
    )

g1, g2, g3 = st.columns(3)
with g1:
    gp_initial = st.slider(
        "Glidepath Initial Equity (%)", min_value=20, max_value=90, value=60, step=5,
    ) / 100.0
with g2:
    gp_final = st.slider(
        "Glidepath Final Equity (%)", min_value=0, max_value=80, value=30, step=5,
    ) / 100.0
with g3:
    gp_shape = st.radio("Glidepath Shape", options=["linear", "convex", "concave"], index=0)

mkt = shared_market_sidebar(context="retirement", include_cppi=show_cppi)
run_sims = st.button("Run Simulation", type="primary", use_container_width=True, key="ret_run")

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
    block_size=mkt["block_size"],
)

if (
    ("retirement_sims" not in st.session_state)
    or run_sims
    or len(st.session_state.get("retirement_sims", ())) != 3
):
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
            strategy_type="AgeInBonds",
            decumulation_start_age=int(retirement_start_age),
            **common,
        ),
        run_simulation(
            lambda_pct=float(lambda_pct),
            strategy_type="CM",
            **common,
        ),
    )

if "retirement_sims" not in st.session_state:
    st.info("Configure parameters and click **Run Simulation** to begin.")
    st.stop()

sim_gp, sim_aib, sim_cm = st.session_state["retirement_sims"]
sim = sim_gp if primary_strategy == "Glidepath" else sim_aib
primary_label = primary_strategy


def _strategy_summary(label: str, sim_data: dict) -> dict[str, float | str]:
    survival_times = np.asarray(sim_data["survival_times"], dtype=float)
    return {
        "Strategy": label,
        "Income PoS (%)": round(float(sim_data["prob_success_income"]), 1),
        "Median Funded Years": round(float(np.median(survival_times)), 1),
        "Downside Funded Yrs (P10)": round(float(np.percentile(survival_times, 10)), 1),
        "Median Ending Wealth (€)": round(float(sim_data["median_ending"]), 0),
    }


comparison_df = pd.DataFrame([
    _strategy_summary("Glidepath", sim_gp),
    _strategy_summary("Age-in-Bonds", sim_aib),
    _strategy_summary("Constant Mix", sim_cm),
]).set_index("Strategy")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.caption(
    f"Does the plan sustain your retirement income under {primary_strategy} vs Constant Mix? "
    f"{mkt['n_simulations']:,} Monte-Carlo paths · "
    f"{time_horizon}-year withdrawal horizon"
)

st.caption(
    "Income PoS = probability of withdrawals never falling short. "
    "Funded Years = how long the portfolio supports the target withdrawal. "
    "Downside Funded Years (P10) = 10th-percentile outcome, for downside protection."
)

# ---------------------------------------------------------------------------
# KPIs — Income PoS hero + funded-years metrics
# ---------------------------------------------------------------------------
hero_col, k2, k3, k4, k5 = st.columns([1.4, 1, 1, 1, 1])

with hero_col:
    build_pos_hero(sim["prob_success_income"])

survival_times = np.asarray(sim["survival_times"], dtype=float)
avg_survival = float(np.mean(survival_times))
p10_survival = float(np.percentile(survival_times, 10))

# Age-in-Bonds is the retirement base case for KPI deltas.
baseline = sim_aib
baseline_income_pos = float(baseline["prob_success_income"])
baseline_median_funded = float(np.median(np.asarray(baseline["survival_times"], dtype=float)))
baseline_median = float(baseline["median_ending"])

median_funded_years = float(np.median(survival_times))
income_pos_delta = float(sim["prob_success_income"]) - baseline_income_pos
median_funded_delta = median_funded_years - baseline_median_funded
k2.metric("Income PoS", f"{float(sim['prob_success_income']):.1f} %", delta=f"{income_pos_delta:+.1f} % vs Age-in-Bonds")
k3.metric("Median Funded Years", f"{median_funded_years:.1f} yrs", delta=f"{median_funded_delta:+.1f} yrs vs Age-in-Bonds")

k4.metric("Downside Funded Years (P10)", f"{p10_survival:.1f} yrs")

k5_value, k5_delta, k5_delta_color = format_delta_metric(float(sim['median_ending']), baseline_median, currency=True)
k5.metric("Median Ending Wealth", k5_value, delta=k5_delta, delta_color=k5_delta_color)

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
        median_label=f"Median {primary_label} Portfolio",
        show_risky_mean=False,
        x_values=np.linspace(
            float(retirement_start_age),
            float(retirement_start_age) + float(time_horizon),
            len(sim["dates"]),
        ),
        xaxis_title="Investor Age (Retirement Onward)",
    ),
    width='stretch',
)

if primary_strategy == "Age-in-Bonds":
    st.plotly_chart(
        build_age_in_bonds_schedule_chart(
            starting_age=int(retirement_start_age),
            time_horizon=time_horizon,
            rebalance_freq=mkt["rebalance_freq"],
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
            primary_label=primary_label,
            secondary_label="Constant Mix",
            x_start_age=int(retirement_start_age),
        ),
        width='stretch',
    )

with col_right:
    st.plotly_chart(
        build_survival_comparison(
            sim, sim_cm,
            title=f"Portfolio Survival — {primary_label} vs Constant Mix",
            time_horizon=time_horizon,
            primary_label=primary_label,
            secondary_label="Constant Mix",
        ),
        width='stretch',
    )

st.subheader("Strategy Comparison")
st.dataframe(
    comparison_df.style.format({
        "Income PoS (%)": "{:.1f}",
        "Median Funded Years": "{:.1f}",
        "Downside Funded Yrs (P10)": "{:.1f}",
        "Median Ending Wealth (€)": "€ {:,.0f}",
    }),
    width='stretch',
)

# ---------------------------------------------------------------------------
# Caveats
# ---------------------------------------------------------------------------
# build_model_caveats_panel()
