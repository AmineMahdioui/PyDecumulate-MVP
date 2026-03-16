"""
Page 1 — Lifecycle Wealth Path
==============================
Full lifecycle simulation: accumulation → retirement → decumulation.
"""

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")

from _shared import (
    FREQ_TO_PD_OFFSET,
    build_model_caveats_panel,
    build_mountain_chart,
    run_lifecycle_simulation,
    shared_market_sidebar,
)

st.sidebar.title("Lifecycle Setup")
mkt = shared_market_sidebar(context="lifecycle", include_cppi=True)

st.title("Lifecycle Wealth Path")
st.caption("A single-page view of saving, retirement transition, and drawdown.")

lifecycle_mode_label = st.radio(
    "Lifecycle Strategy Mode",
    options=["CPPI→CM", "Glidepath→Glidepath"],
    horizontal=True,
)
is_cppi_to_cm = lifecycle_mode_label == "CPPI→CM"
lifecycle_mode = "CPPI_TO_CM" if is_cppi_to_cm else "GLIDEPATH_TO_GLIDEPATH"

with st.expander("How CPPI Works (Floor Mechanics)", expanded=False):
    st.markdown(
        "- CPPI sets a **floor** and invests risky assets from the **cushion**: `cushion = max(wealth - floor, 0)`  \n"
        "- Target risky allocation is `m × cushion`, capped at total wealth (no leverage).  \n"
        "- If wealth approaches the floor, cushion shrinks and risky exposure is mechanically reduced.  \n"
        "- In this app, that floor logic is active in **accumulation CPPI** only. Decumulation runs CM or Glidepath, so retirement floor is not an active control."
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
    if is_cppi_to_cm:
        lc_acc_floor = st.slider(
            "Capital Protection Floor (%)", min_value=50, max_value=100,
            value=80, step=5, key="lc_acc_floor",
        )
        gp_acc_initial = 0.80
        gp_acc_final = 0.20
        gp_acc_shape = "linear"
    else:
        gp_acc_initial = st.slider(
            "Accumulation Glidepath Initial Equity (%)", min_value=40, max_value=100,
            value=80, step=5, key="lc_gp_acc_initial",
        ) / 100.0
        gp_acc_final = st.slider(
            "Accumulation Glidepath Final Equity (%)", min_value=0, max_value=80,
            value=20, step=5, key="lc_gp_acc_final",
        ) / 100.0
        gp_acc_shape = st.radio(
            "Accumulation Glidepath Shape",
            options=["linear", "convex", "concave"],
            index=0,
            key="lc_gp_acc_shape",
        )
        lc_acc_floor = 80

with col_dec:
    st.subheader("Retirement Phase")
    with st.expander("Lifecycle Handoff", expanded=False):
        st.markdown(
            "Each simulated accumulation path feeds its own retirement starting wealth. "
            "This keeps the lifecycle transition pathwise rather than using a single pooled retirement pot."
        )
    lc_dec_withdrawal = st.number_input(
        "Annual Withdrawal (€)", min_value=0, max_value=10_000_000,
        value=40_000, step=5_000, format="%d", key="lc_dec_withdrawal",
    )
    lc_dec_horizon = st.slider(
        "Years in Retirement", min_value=1, max_value=40,
        value=30, key="lc_dec_horizon",
    )
    if is_cppi_to_cm:
        lc_dec_lambda = st.slider(
            "Retirement % Invested in Risky Assets (CM)",
            min_value=20,
            max_value=90,
            value=60,
            step=5,
            key="lc_dec_lambda",
        )
        gp_dec_initial = 0.60
        gp_dec_final = 0.30
        gp_dec_shape = "linear"
    else:
        gp_dec_initial = st.slider(
            "Retirement Glidepath Initial Equity (%)", min_value=20, max_value=90,
            value=60, step=5, key="lc_gp_dec_initial",
        ) / 100.0
        gp_dec_final = st.slider(
            "Retirement Glidepath Final Equity (%)", min_value=0, max_value=80,
            value=30, step=5, key="lc_gp_dec_final",
        ) / 100.0
        gp_dec_shape = st.radio(
            "Retirement Glidepath Shape",
            options=["linear", "convex", "concave"],
            index=0,
            key="lc_gp_dec_shape",
        )
        lc_dec_lambda = 60

sim_acc, sim_dec, retirement_pot = run_lifecycle_simulation(
    acc_initial_wealth=float(lc_acc_wealth),
    acc_time_horizon=lc_acc_horizon,
    acc_contribution=float(lc_acc_contribution),
    acc_floor_pct=float(lc_acc_floor),
    acc_cppi_multiplier=mkt["cppi_multiplier"],
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

st.divider()
retirement_p5 = float(sim_acc.get("retirement_p5_nominal", np.percentile(sim_acc["retirement_wealths_nominal"], 5)))
retirement_risky_alloc_median = float(sim_acc.get("retirement_risky_alloc_median", np.median(sim_acc["retirement_risky_allocation"]) * 100.0))
floor_touch_path_pct = sim_acc.get("floor_touch_path_pct")
floor_touch_time_pct = sim_acc.get("floor_touch_time_pct")

st.caption(
    "These are lifecycle risk metrics, not marketing metrics. Focus on retirement handoff quality, survival, and shortfall."
)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Median Retirement Pot", f"€ {retirement_pot:,.0f}")
k2.metric("P5 Retirement Pot", f"€ {retirement_p5:,.0f}")
k3.metric(f"Full-Lifecycle PoS ({lifecycle_mode_label})", f"{sim_dec['prob_success']:.1f} %")
k4.metric("Expected Shortfall", f"€ {sim_dec['expected_shortfall']:,.0f}")
k5.metric("Median Residual Wealth", f"€ {sim_dec['median_ending']:,.0f}")
if is_cppi_to_cm:
    k6.metric("Median Risky Allocation at Retirement", f"{retirement_risky_alloc_median:.1f} %")
else:
    k6.metric("Retirement Glidepath", f"{gp_dec_initial*100:.0f}% → {gp_dec_final*100:.0f}%")

if floor_touch_path_pct is not None and floor_touch_time_pct is not None:
    st.caption(
        f"Accumulation floor contact — {floor_touch_path_pct:.1f}% of paths touched the floor at least once; "
        f"{floor_touch_time_pct:.1f}% of simulated time-steps were at or below floor."
    )

st.divider()

acc_dates = sim_acc["dates"]
freq_alias = FREQ_TO_PD_OFFSET[mkt["rebalance_freq"]]
dec_start = acc_dates[-1] + pd.tseries.frequencies.to_offset(freq_alias)
dec_dates = pd.date_range(start=dec_start, periods=len(sim_dec["dates"]), freq=freq_alias)

st.plotly_chart(
    build_mountain_chart(sim_acc, sim_dec, acc_dates, dec_dates),
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
