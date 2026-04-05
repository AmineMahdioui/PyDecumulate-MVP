"""
Page 1 — Lifecycle Wealth Path (Modernized UX)
"""

import numpy as np
import streamlit as st

st.set_page_config(layout="wide")

from _shared import run_lifecycle_simulation
from ui.charts import build_mountain_chart_age
from ui.components import (
    build_model_caveats_panel,
    format_delta_metric,
    shared_market_sidebar,
)

st.title("Lifecycle Wealth Path")
st.caption("Plan your financial journey from accumulation to retirement drawdown.")

# ---------------------------------------------------------------------------
# UX Improvement 1: Progressive Disclosure (Hide the Math)
# Move the market assumptions and simulation controls entirely to the sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Market Assumptions")
mkt = shared_market_sidebar(context="lifecycle", include_cppi=False)

# ---------------------------------------------------------------------------
# UX Improvement 2: Clean Top-Level Inputs (Focus on the User, not the Algorithm)
# ---------------------------------------------------------------------------
st.subheader("Your Financial Profile")
col_acc, col_dec = st.columns(2)

with col_acc:
    start_age = st.number_input("Current Age", min_value=20, max_value=70, value=25, step=1)
    lc_acc_horizon = st.slider("Years until Retirement", min_value=1, max_value=40, value=40)
    lc_acc_wealth = st.number_input("Current Savings (€)", min_value=0, value=5_000, step=5_000)
    lc_acc_contribution = st.number_input("Annual Contribution (€)", min_value=0, value=12_000, step=1_000)

with col_dec:
    lc_dec_horizon = st.slider("Expected Years in Retirement", min_value=1, max_value=40, value=30)
    # UX Shift: Ask for desired MONTHLY income, convert to annual in backend
    monthly_withdrawal = st.number_input("Target Monthly Retirement Income (€)", min_value=0, value=3_333, step=100)
    lc_dec_withdrawal = monthly_withdrawal * 12

# ---------------------------------------------------------------------------
# UX Improvement 3: Advanced Strategy Settings (Hidden by default)
# Group CPPI vs Glidepath into simple "Risk Profiles"
# ---------------------------------------------------------------------------
with st.expander("⚙️ Advanced Strategy & Risk Settings"):
    st.write("Customize how the algorithm protects your capital.")

    lifecycle_mode_label = st.radio(
        "Backend Strategy Model",
        options=["Dynamic Protection (CPPI → Constant Mix)", "Pre-set Lifecycle (Glidepath → Glidepath)"],
        horizontal=True,
    )
    is_cppi_to_cm = "CPPI" in lifecycle_mode_label
    lifecycle_mode = "CPPI_TO_CM" if is_cppi_to_cm else "GLIDEPATH_TO_GLIDEPATH"

    if is_cppi_to_cm:
        lc_acc_floor = st.slider("Capital Protection Floor (%)", min_value=50, max_value=100, value=90, step=5)
        lc_dec_lambda = st.slider("Retirement Equity Allocation (%)", min_value=20, max_value=90, value=60, step=5)
        mkt_cppi_multiplier = st.slider("CPPI Multiplier (m)", min_value=1.0, max_value=10.0, value=3.0, step=0.5)

        # Default unused parameters
        gp_acc_initial, gp_acc_final, gp_dec_initial, gp_dec_final = 0.8, 0.2, 0.6, 0.3
        gp_acc_shape = gp_dec_shape = "linear"
    else:
        # Simplified Glidepath UI
        gp_acc_initial = st.slider("Initial Equity Allocation (%)", min_value=40, max_value=100, value=80) / 100.0
        gp_dec_final = st.slider("Final Retirement Equity Allocation (%)", min_value=0, max_value=80, value=30) / 100.0
        gp_acc_shape = st.selectbox("Glidepath Shape", options=["linear", "convex", "concave"])

        # Default unused parameters
        lc_acc_floor, lc_dec_lambda, mkt_cppi_multiplier = 80, 60, 1.0
        gp_acc_final, gp_dec_initial, gp_dec_shape = gp_dec_final, gp_dec_final, gp_acc_shape

st.divider()

# ---------------------------------------------------------------------------
# Run Simulation
# ---------------------------------------------------------------------------
if st.button("Calculate Retirement Trajectory", type="primary", use_container_width=True):
    st.session_state["lc_result"] = run_lifecycle_simulation(
        acc_initial_wealth=float(lc_acc_wealth),
        acc_time_horizon=lc_acc_horizon,
        acc_contribution=float(lc_acc_contribution),
        acc_floor_pct=float(lc_acc_floor),
        acc_cppi_multiplier=mkt_cppi_multiplier,
        dec_time_horizon=lc_dec_horizon,
        dec_withdrawal=float(lc_dec_withdrawal),
        expected_return=mkt["expected_return"],
        market_volatility=mkt["market_volatility"],
        risk_free_rate=mkt["risk_free_rate"],
        n_simulations=mkt["n_simulations"],
        rebalance_freq=mkt["rebalance_freq"],
        simulation_method=mkt["simulation_method"],
        block_length=mkt["block_length"],
        Lambda=float(lc_dec_lambda),
        lifecycle_mode=lifecycle_mode,
        acc_glidepath_initial=float(gp_acc_initial),
        acc_glidepath_final=float(gp_acc_final),
        acc_glidepath_shape=gp_acc_shape,
        dec_glidepath_initial=float(gp_dec_initial),
        dec_glidepath_final=float(gp_dec_final),
        dec_glidepath_shape=gp_dec_shape,
    )

if "lc_result" not in st.session_state:
    st.info("Adjust your parameters and click calculate to view your wealth path.")
    st.stop()

sim_acc, sim_dec, retirement_pot = st.session_state["lc_result"]

# ---------------------------------------------------------------------------
# UX Improvement 4: Focus on the "So What?" Metrics
# ---------------------------------------------------------------------------
st.subheader("Your Retirement Outlook")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Median Retirement Pot", f"€ {retirement_pot:,.0f}")
k2.metric("Probability of Success", f"{float(sim_dec['prob_success']):.1f} %")
# Translate Expected Shortfall into terms they understand (Years of Income Lost)
shortfall_euros = float(sim_dec['expected_shortfall'])
months_lost = (shortfall_euros / monthly_withdrawal) if monthly_withdrawal > 0 else 0
k3.metric("Expected Shortfall", f"€ {shortfall_euros:,.0f}", delta=f"-{months_lost:.1f} months income", delta_color="inverse")
k4.metric("Median Residual Wealth (End of Life)", f"€ {float(sim_dec['median_ending']):,.0f}")

st.divider()

# Display the main simplified chart
steps_per_year = {"daily": 252, "weekly": 52, "monthly": 12, "quarterly": 4, "yearly": 1}[mkt["rebalance_freq"]]
acc_age_axis = start_age + np.arange(1, len(sim_acc["dates"]) + 1) / steps_per_year
dec_age_axis = start_age + lc_acc_horizon + np.arange(1, len(sim_dec["dates"]) + 1) / steps_per_year

st.plotly_chart(
    build_mountain_chart_age(
        sim_acc,
        sim_dec,
        acc_age_axis,
        dec_age_axis,
        show_inner_bands=False,  # Hidden by default to reduce visual noise
    ),
    width='stretch',
)

build_model_caveats_panel()
