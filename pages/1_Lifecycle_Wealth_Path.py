"""
Page 1 — Lifecycle Wealth Path (Modernized UX)
"""

import numpy as np
import streamlit as st

st.set_page_config(layout="wide")

from _shared import run_lifecycle_simulation
from ui.charts import build_mountain_chart_age
from ui.components import (
    build_top_controls_header,
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
st.sidebar.title("Global Market & Simulation")
mkt = shared_market_sidebar(context="lifecycle", include_cppi=False)

# ---------------------------------------------------------------------------
# UX Improvement 2: Clean Top-Level Inputs (Focus on the User, not the Algorithm)
# ---------------------------------------------------------------------------
build_top_controls_header(
    "Your Financial Profile",
    "Set your personal lifecycle assumptions.",
)
col_acc, col_dec = st.columns(2)

with col_acc:
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
    lc_acc_horizon = int(retirement_age - start_age)
    lc_acc_wealth = st.number_input("Current Savings (€)", min_value=0, value=5_000, step=5_000)
    lc_acc_contribution = st.number_input("Annual Contribution (€)", min_value=0, value=12_000, step=1_000)

with col_dec:
    lc_dec_horizon = st.slider("Expected Years in Retirement", min_value=1, max_value=80, value=30)
    withdrawal_input_mode = st.radio("Retirement Income Input", options=["Monthly", "Annual"], horizontal=True)
    if withdrawal_input_mode == "Monthly":
        monthly_withdrawal = st.number_input("Target Monthly Retirement Income (€)", min_value=0, value=3_333, step=100)
        lc_dec_withdrawal = monthly_withdrawal * 12
    else:
        lc_dec_withdrawal = st.number_input("Target Annual Retirement Income (€)", min_value=0, value=40_000, step=1_000)

viz_left, viz_right = st.columns(2)
with viz_left:
    show_inner_bands = st.checkbox("Show P25-P75 Inner Bands", value=False)
with viz_right:
    include_contribution_layer = st.checkbox("Show Cumulative Contributions Layer", value=False)

with st.expander("Phase Timing Options", expanded=False):
    use_split_rebalance = st.checkbox("Use separate rebalance frequencies for accumulation and decumulation", value=False)
    if use_split_rebalance:
        freq_col_1, freq_col_2 = st.columns(2)
        with freq_col_1:
            acc_rebalance_freq = st.selectbox(
                "Accumulation Rebalancing Frequency",
                options=["daily", "weekly", "monthly", "quarterly", "yearly"],
                index=["daily", "weekly", "monthly", "quarterly", "yearly"].index(mkt["rebalance_freq"]),
            )
        with freq_col_2:
            dec_rebalance_freq = st.selectbox(
                "Decumulation Rebalancing Frequency",
                options=["daily", "weekly", "monthly", "quarterly", "yearly"],
                index=["daily", "weekly", "monthly", "quarterly", "yearly"].index(mkt["rebalance_freq"]),
            )
    else:
        acc_rebalance_freq = mkt["rebalance_freq"]
        dec_rebalance_freq = mkt["rebalance_freq"]

# ---------------------------------------------------------------------------
# UX Improvement 3: Advanced Strategy Settings (Hidden by default)
# Group CPPI vs Glidepath into simple "Risk Profiles"
# ---------------------------------------------------------------------------
with st.expander("⚙️ Advanced Strategy & Risk Settings"):
    st.write("Customize how the algorithm protects your capital.")

    lifecycle_mode_label = st.radio(
        "Backend Strategy Model",
        options=[
            "Dynamic Protection (CPPI → Constant Mix)",
            "Pre-set Lifecycle (Glidepath → Glidepath)",
            "Age Rule Lifecycle (Age-in-Bonds → Age-in-Bonds)",
        ],
        horizontal=True,
    )
    is_cppi_to_cm = "CPPI" in lifecycle_mode_label
    is_age_in_bonds = "Age-in-Bonds" in lifecycle_mode_label
    if is_cppi_to_cm:
        lifecycle_mode = "CPPI_TO_CM"
    elif is_age_in_bonds:
        lifecycle_mode = "AGEINBONDS_TO_AGEINBONDS"
    else:
        lifecycle_mode = "GLIDEPATH_TO_GLIDEPATH"

    st.write("Handoff Age Controls")
    retirement_age_default = int(retirement_age)
    use_custom_retirement_age = st.checkbox("Override retirement age for handoff", value=False)
    retirement_age_input = (
        st.number_input(
            "Retirement Age Override",
            min_value=max(20, int(start_age) + 1),
            max_value=95,
            value=min(max(retirement_age_default, int(start_age) + 1), 95),
            step=1,
        )
        if use_custom_retirement_age
        else retirement_age_default
    )

    if is_cppi_to_cm:
        lc_acc_floor = st.slider("Capital Protection Floor (%)", min_value=50, max_value=100, value=90, step=5)
        lc_dec_lambda = st.slider("Retirement Equity Allocation (%)", min_value=20, max_value=90, value=60, step=5)
        mkt_cppi_multiplier = st.slider("CPPI Multiplier (m)", min_value=1.0, max_value=10.0, value=3.0, step=0.5)

        # Default unused parameters
        gp_acc_initial, gp_acc_final, gp_dec_initial, gp_dec_final = 0.8, 0.2, 0.6, 0.3
        gp_acc_shape = gp_dec_shape = "linear"
    elif not is_age_in_bonds:
        # Simplified Glidepath UI
        gp_acc_initial = st.slider("Initial Equity Allocation (%)", min_value=40, max_value=100, value=80) / 100.0
        gp_dec_final = st.slider("Final Retirement Equity Allocation (%)", min_value=0, max_value=80, value=30) / 100.0
        gp_acc_shape = st.selectbox("Glidepath Shape", options=["linear", "convex", "concave"])

        # Default unused parameters
        lc_acc_floor, lc_dec_lambda, mkt_cppi_multiplier = 80, 60, 1.0
        gp_acc_final, gp_dec_initial, gp_dec_shape = gp_dec_final, gp_dec_final, gp_acc_shape
    else:
        # Age-in-bonds mode does not use glidepath controls
        lc_acc_floor, lc_dec_lambda, mkt_cppi_multiplier = 80, 60, 1.0
        gp_acc_initial, gp_acc_final, gp_dec_initial, gp_dec_final = 0.8, 0.2, 0.6, 0.3
        gp_acc_shape = gp_dec_shape = "linear"

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
        acc_rebalance_freq=acc_rebalance_freq,
        dec_rebalance_freq=dec_rebalance_freq,
        simulation_method=mkt["simulation_method"],
        block_size=mkt["block_size"],
        Lambda=float(lc_dec_lambda),
        lifecycle_mode=lifecycle_mode,
        acc_glidepath_initial=float(gp_acc_initial),
        acc_glidepath_final=float(gp_acc_final),
        acc_glidepath_shape=gp_acc_shape,
        dec_glidepath_initial=float(gp_dec_initial),
        dec_glidepath_final=float(gp_dec_final),
        dec_glidepath_shape=gp_dec_shape,
        current_age=int(start_age),
        retirement_age=int(retirement_age_input),
    )

if "lc_result" not in st.session_state:
    st.info("Adjust your parameters and click calculate to view your wealth path.")
    st.stop()

sim_acc, sim_dec, retirement_pot = st.session_state["lc_result"]

# ---------------------------------------------------------------------------
# UX Improvement 4: Focus on the "So What?" Metrics
# ---------------------------------------------------------------------------
st.subheader("Your Retirement Outlook")

k1, k2, k3, k4, k5 = st.columns(5)
dec_survival = np.asarray(sim_dec["survival_times"], dtype=float)
median_funded_years = float(np.median(dec_survival))
p10_funded_years = float(np.percentile(dec_survival, 10))

k1.metric("Median Retirement Pot", f"€ {retirement_pot:,.0f}")
k2.metric("Income PoS", f"{float(sim_dec['prob_success_income']):.1f} %")
k3.metric("Median Funded Years", f"{median_funded_years:.1f} yrs")
k4.metric("Downside Funded Years (P10)", f"{p10_funded_years:.1f} yrs")
k5.metric("Median Residual Wealth (End of Life)", f"€ {float(sim_dec['median_ending']):,.0f}")

st.divider()

# Display the main simplified chart
retirement_age_axis = float(retirement_age_input)
acc_age_axis = np.linspace(18.0, retirement_age_axis, len(sim_acc["dates"]))
dec_age_axis = np.linspace(retirement_age_axis, retirement_age_axis + float(lc_dec_horizon), len(sim_dec["dates"]))

st.plotly_chart(
    build_mountain_chart_age(
        sim_acc,
        sim_dec,
        acc_age_axis,
        dec_age_axis,
        include_contributions=include_contribution_layer,
        show_inner_bands=show_inner_bands,
    ),
    width='stretch',
)

build_model_caveats_panel()
