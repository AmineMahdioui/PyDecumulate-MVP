"""
Page 1 — Lifecycle Wealth Path
==============================
Full lifecycle simulation: accumulation → retirement → decumulation.
"""

import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")

from _shared import (
    FREQ_TO_PD_OFFSET,
    build_cash_flow_timeline,
    build_model_caveats_panel,
    build_mountain_chart,
    run_lifecycle_simulation,
    shared_market_sidebar,
)

st.sidebar.title("Lifecycle Setup")
mkt = shared_market_sidebar(context="lifecycle", include_cppi=True)

st.title("Lifecycle Wealth Path")
st.caption(
    "A single-page view of saving, retirement transition, and drawdown. "
    "Current lifecycle engine uses CPPI in accumulation and Constant Mix in retirement."
)
st.info(
    "On this page, the sidebar **CPPI multiplier** affects the accumulation phase only. "
    "The retirement phase currently uses Constant Mix, so the retirement floor input is shown "
    "for transparency but is not enforced as a hard CPPI-style floor."
)

col_acc, col_dec = st.columns(2)

with col_acc:
    st.subheader("Accumulation Phase")
    start_age = st.number_input("Current Age", min_value=20, max_value=70, value=25, step=1)
    lc_acc_wealth = st.number_input(
        "Starting Pot (€)", min_value=0, max_value=10_000_000,
        value=1_000_000, step=50_000, format="%d", key="lc_acc_wealth",
    )
    lc_acc_contribution = st.number_input(
        "Annual Contribution (€)", min_value=0, max_value=10_000_000,
        value=50_000, step=5_000, format="%d", key="lc_acc_contrib",
    )
    lc_acc_horizon = st.slider(
        "Years to Retirement", min_value=1, max_value=40,
        value=40, key="lc_acc_horizon",
    )
    lc_acc_floor = st.slider(
        "Capital Protection Floor (%)", min_value=50, max_value=100,
        value=80, step=5, key="lc_acc_floor",
    )

with col_dec:
    st.subheader("Retirement Phase")
    st.info(
        "Lifecycle handoff is **pathwise**: each simulated accumulation path feeds its own "
        "retirement starting wealth."
    )
    lc_dec_withdrawal = st.number_input(
        "Annual Withdrawal (€)", min_value=0, max_value=10_000_000,
        value=40_000, step=5_000, format="%d", key="lc_dec_withdrawal",
    )
    lc_dec_horizon = st.slider(
        "Years in Retirement", min_value=1, max_value=40,
        value=30, key="lc_dec_horizon",
    )
    lc_dec_floor = st.slider(
        "Retirement Floor Input (%)", min_value=50, max_value=100,
        value=80, step=5, key="lc_dec_floor",
        help="Current lifecycle retirement uses Constant Mix. This value is retained for transparency and future multi-strategy lifecycle extensions, but it is not enforced as a hard floor in the current retirement path.",
    )

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
    simulation_method=mkt["simulation_method"],
    block_length=mkt["block_length"],
)

st.divider()
total_in = lc_acc_wealth + lc_acc_contribution * lc_acc_horizon
withdrawal_rate = (lc_dec_withdrawal / retirement_pot * 100) if retirement_pot > 0 else 0.0

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric(
    "Median Retirement Pot",
    f"€ {retirement_pot:,.0f}",
    delta=f"{retirement_pot - total_in:+,.0f} vs contributed",
)
k2.metric("Full-Lifecycle PoS (CPPI→CM)", f"{sim_dec['prob_success']:.1f} %")
k3.metric("Median Residual Wealth", f"€ {sim_dec['median_ending']:,.0f}")
k4.metric("Total Contributions", f"€ {total_in:,.0f}")
k5.metric("Withdrawal Rate", f"{withdrawal_rate:.1f} %")
k6.metric("Retirement Floor Input", f"{lc_dec_floor} %")

st.divider()

acc_dates = sim_acc["dates"]
freq_alias = FREQ_TO_PD_OFFSET[mkt["rebalance_freq"]]
dec_start = acc_dates[-1] + pd.tseries.frequencies.to_offset(freq_alias)
dec_dates = pd.date_range(
    start=dec_start,
    periods=len(sim_dec["dates"]),
    freq=freq_alias,
)

st.plotly_chart(
    build_mountain_chart(sim_acc, sim_dec, acc_dates, dec_dates),
    width='stretch',
)

st.plotly_chart(
    build_cash_flow_timeline(
        start_age=int(start_age),
        acc_horizon=int(lc_acc_horizon),
        dec_horizon=int(lc_dec_horizon),
        annual_contribution=float(lc_acc_contribution),
        annual_withdrawal=float(lc_dec_withdrawal),
    ),
    width='stretch',
)

st.caption(
    f"Assumptions — {mkt['n_simulations']:,} Monte-Carlo paths · "
    f"Expected return {mkt['expected_return']:.1f} % · "
    f"Volatility {mkt['market_volatility']:.1f} % · "
    f"Risk-free rate {mkt['risk_free_rate']:.1f} % · "
    f"Rebalanced {mkt['rebalance_freq']} · Displayed values are nominal (€)"
)

build_model_caveats_panel()
