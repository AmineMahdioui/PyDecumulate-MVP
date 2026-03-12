"""
Page 4 — Sensitivity & Model Risk
=================================
Button-triggered heatmap of Probability of Success across
withdrawal rates × retirement horizons.
"""

import streamlit as st

st.set_page_config(layout="wide")

from _shared import (
    build_model_caveats_panel,
    build_sensitivity_heatmap,
    run_sensitivity_sweep,
    shared_market_sidebar,
)

st.sidebar.title("Sensitivity Setup")
mkt = shared_market_sidebar()

st.title("Sensitivity & Model Risk")
st.caption(
    "Explore how Probability of Success changes across withdrawal rates and retirement horizons "
    "under the current market assumptions."
)

c1, c2, c3 = st.columns(3)
with c1:
    strategy_type = st.radio(
        "Strategy",
        options=["CPPI", "Constant Mix"],
        index=0,
        horizontal=True,
    )
with c2:
    initial_wealth = st.number_input(
        "Initial Retirement Wealth (€)",
        min_value=100_000, max_value=10_000_000,
        value=1_000_000, step=100_000, format="%d",
    )
with c3:
    floor_pct = st.slider("Floor (%)", min_value=50, max_value=100, value=80, step=5)

c4, c5 = st.columns(2)
with c4:
    withdrawal_rates = st.multiselect(
        "Withdrawal Rates (% of retirement pot)",
        options=[2, 3, 4, 5, 6, 7, 8],
        default=[2, 4, 6],
    )
with c5:
    horizons = st.multiselect(
        "Retirement Horizons (years)",
        options=[10, 15, 20, 25, 30, 35],
        default=[15, 20, 25, 30],
    )

_strategy_map = {"CPPI": "CPPI", "Constant Mix": "CM"}
_strat = _strategy_map[strategy_type]

st.caption(
    "Withdrawal rate = annual withdrawal ÷ initial retirement wealth. "
    f"Each cell shows Probability of Success (%) using **{mkt['n_simulations']:,} simulations**. "
    "Displayed values are nominal and the Glidepath strategy is intentionally excluded here because "
    "the current codebase does not yet have a dedicated decumulation glidepath engine."
)

if not withdrawal_rates or not horizons:
    st.warning("Select at least one withdrawal rate and one horizon.")
else:
    if st.button("Generate Sensitivity Matrix", type="primary"):
        with st.spinner("Running sweep …"):
            df = run_sensitivity_sweep(
                withdrawal_rates=[float(r) for r in withdrawal_rates],
                horizons=[int(h) for h in horizons],
                initial_wealth=float(initial_wealth),
                floor_pct=float(floor_pct),
                cppi_multiplier=mkt["cppi_multiplier"],
                expected_return=mkt["expected_return"],
                market_volatility=mkt["market_volatility"],
                risk_free_rate=mkt["risk_free_rate"],
                rebalance_freq=mkt["rebalance_freq"],
                simulation_method=mkt["simulation_method"],
                block_length=mkt["block_length"],
                strategy_type=_strat,
                n_sims_sweep=mkt["n_simulations"],
            )
            st.session_state["sensitivity_df"] = df

    if "sensitivity_df" in st.session_state:
        st.plotly_chart(build_sensitivity_heatmap(st.session_state["sensitivity_df"]), width='stretch')

build_model_caveats_panel()
