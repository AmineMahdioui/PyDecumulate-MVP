"""
Page 1; Lifecycle Wealth Path
================================
Full lifecycle simulation: accumulation → retirement → decumulation,
presented as a single mountain chart with the median retirement pot
as the hero KPI.
"""

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")

from _shared import (
    AMUNDI_GREEN,
    FREQ_TO_PD_OFFSET,
    build_cash_flow_timeline,
    build_mountain_chart_age,
    build_model_caveats_panel,
    run_lifecycle_simulation,
    shared_market_sidebar,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Lifecycle Setup")
mkt = shared_market_sidebar()

# ---------------------------------------------------------------------------
# Main area; per-phase parameters
# ---------------------------------------------------------------------------
st.title("Lifecycle Wealth Path")
st.caption(
    "Retirement Solutions Research Prototype; Simulate the complete journey "
    "from accumulation through retirement, stitched into a single timeline."
)

col_acc, col_dec = st.columns(2)

with col_acc:
    st.subheader("Accumulation Phase")
    lc_start_age = st.number_input(
        "Current Age", min_value=20, max_value=60,
        value=35, step=1, key="lc_start_age",
    )
    lc_acc_wealth = st.number_input(
        "Starting Pot (€)", min_value=0, max_value=10_000_000,
        value=50_000, step=5_000, format="%d", key="lc_acc_wealth",
    )
    lc_acc_contribution = st.number_input(
        "Annual Contribution (€)", min_value=0, max_value=10_000_000,
        value=12_000, step=1_000, format="%d", key="lc_acc_contrib",
    )
    lc_acc_horizon = st.slider(
        "Years to Retirement", min_value=1, max_value=40,
        value=30, key="lc_acc_horizon",
    )
    lc_acc_floor = st.slider(
        "Capital Protection Floor (%)", min_value=50, max_value=100,
        value=80, step=5, key="lc_acc_floor",
    )

with col_dec:
    st.subheader("Decumulation Phase")
    st.info(
        "The decumulation starting wealth is automatically set to the "
        "**median ending wealth** of the accumulation phase (pathwise handoff)."
    )
    lc_dec_withdrawal = st.number_input(
        "Annual Withdrawal (€)", min_value=0, max_value=10_000_000,
        value=40_000, step=5_000, format="%d", key="lc_dec_withdrawal",
    )
    lc_dec_horizon = st.slider(
        "Years in Retirement", min_value=1, max_value=40,
        value=25, key="lc_dec_horizon",
    )
    lc_dec_floor = st.slider(
        "Guaranteed Floor (%)", min_value=50, max_value=100,
        value=80, step=5, key="lc_dec_floor",
    )

# ---------------------------------------------------------------------------
# Run CPPI lifecycle simulation (pathwise handoff)
# ---------------------------------------------------------------------------
sim_acc, sim_dec, retirement_pot = run_lifecycle_simulation(
    acc_initial_wealth=float(lc_acc_wealth),
    acc_time_horizon=lc_acc_horizon,
    acc_contribution=float(lc_acc_contribution),
    acc_floor_pct=float(lc_acc_floor),
    acc_cppi_multiplier=mkt["cppi_multiplier"],
    dec_time_horizon=lc_dec_horizon,
    dec_withdrawal=float(lc_dec_withdrawal),
    dec_floor_pct=float(lc_dec_floor),
    expected_return=mkt["expected_return"],
    market_volatility=mkt["market_volatility"],
    risk_free_rate=mkt["risk_free_rate"],
    n_simulations=mkt["n_simulations"],
    rebalance_freq=mkt["rebalance_freq"],
)

# ---------------------------------------------------------------------------
# KPIs; Median Retirement Pot is the hero; PoS is secondary
# ---------------------------------------------------------------------------
st.divider()
total_in = lc_acc_wealth + lc_acc_contribution * lc_acc_horizon
withdrawal_rate = (
    (lc_dec_withdrawal / retirement_pot * 100) if retirement_pot > 0 else 0.0
)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric(
    "Median Retirement Pot",
    f"€ {retirement_pot:,.0f}",
    delta=f"{retirement_pot - total_in:+,.0f} vs contributed",
)
k2.metric("Retirement Phase PoS *", f"{sim_dec['prob_success']:.1f} %")
k3.metric("Median Residual Wealth", f"€ {sim_dec['median_ending']:,.0f}")
k4.metric("Total Contributions", f"€ {total_in:,.0f}")
k5.metric("Withdrawal Rate", f"{withdrawal_rate:.1f} %")
k6.metric("Floor Level", f"{lc_dec_floor} %")

st.caption(
    "\\* Retirement Phase PoS is conditional on the median retirement pot. "
    "It is **not** a joint full-lifecycle probability."
)

st.divider()

# ---------------------------------------------------------------------------
# Mountain Chart; flagship visualisation (Investor Age x-axis)
# ---------------------------------------------------------------------------
n_acc_steps = len(sim_acc["dates"])
n_dec_steps = len(sim_dec["dates"])
retirement_age = lc_start_age + lc_acc_horizon
age_axis_acc = np.linspace(lc_start_age, retirement_age, n_acc_steps)
age_axis_dec = np.linspace(retirement_age, retirement_age + lc_dec_horizon, n_dec_steps)

st.plotly_chart(
    build_mountain_chart_age(sim_acc, sim_dec, age_axis_acc, age_axis_dec),
    width='stretch',
)

# ---------------------------------------------------------------------------
# Cash Flow Timeline (aligned to Investor Age)
# ---------------------------------------------------------------------------
st.plotly_chart(
    build_cash_flow_timeline(
        acc_contribution=float(lc_acc_contribution),
        dec_withdrawal=float(lc_dec_withdrawal),
        acc_horizon=lc_acc_horizon,
        dec_horizon=lc_dec_horizon,
        start_age=lc_start_age,
    ),
    width='stretch',
)
st.caption(
    "Cash in (green) vs cash out (red) at each age. "
    "All values are fixed annual amounts in Nominal (€); no inflation adjustment."
)

# ---------------------------------------------------------------------------
# Assumptions & Caveats
# ---------------------------------------------------------------------------
st.caption(
    f"Assumptions; {mkt['n_simulations']:,} Monte-Carlo paths · "
    f"Expected return {mkt['expected_return']:.1f} % · "
    f"Volatility {mkt['market_volatility']:.1f} % · "
    f"Risk-free rate {mkt['risk_free_rate']:.1f} % · "
    f"Rebalanced {mkt['rebalance_freq']}"
)

build_model_caveats_panel()
