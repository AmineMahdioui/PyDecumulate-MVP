"""
Retirement Solutions Research Prototype — Multi-Page Streamlit Dashboard
========================================================================
Entrypoint.  Run with:
    streamlit run app.py
"""

import streamlit as st
from _shared import run_simulation

# ---------------------------------------------------------------------------
# Session-state defaults — run once per browser session
# ---------------------------------------------------------------------------
def _init_session_state() -> None:
    """Seed session state with app-wide defaults on first load."""
    if "baseline_cm_result" not in st.session_state:
        # Fixed 60/40 Constant Mix baseline — lightweight anchor for delta metrics.
        # Cached by @st.cache_data; cost is paid only once across all reruns.
        st.session_state["baseline_cm_result"] = run_simulation(
            initial_wealth=100_000,
            time_horizon=40,
            cppi_multiplier=3.0,
            floor_pct=80.0,
            expected_return=5.0,
            market_volatility=15.0,
            risk_free_rate=1.0,
            n_simulations=1_000,
            rebalance_freq="monthly",
            annual_withdrawal=0.0,
            annual_contribution=12_000,
            lambda_pct=60.0,
            strategy_type="CM",
        )


_init_session_state()


st.set_page_config(
    page_title="Retirement Solutions Research Prototype",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

st.title("Retirement Solutions Research Prototype")
st.markdown(
    """
    A Monte Carlo dashboard for **retirement-solution prototyping**.
    It currently focuses on three truthful building blocks:

    - **Lifecycle view:** accumulation followed by retirement drawdown
    - **Accumulation strategy design:** CPPI, Glidepath, and Constant Mix
    - **Retirement sustainability:** survival, shortfall, and ending-wealth outcomes

    Use the sidebar to navigate between pages.
    All displayed values are currently shown in **nominal euro terms** unless a chart or metric says otherwise.
    """
)

with st.expander("Concepts in plain English"):
    st.markdown(
        """
        - **Probability of Success (PoS):** the share of simulated paths that finish above the model floor.
        - **Expected Shortfall:** the average size of the bad outcomes when the strategy fails its target.
        - **Survival Time:** how long the portfolio lasts before it runs out or breaches the floor.
        - **Floor:** the minimum wealth level the model tries to defend. In CPPI this is a model mechanism, not a legal guarantee.
        - **Glidepath:** a pre-set plan that reduces risky allocation over time.
        - **CPPI:** a dynamic strategy that changes risky exposure based on the cushion above the floor.
        """
    )

with st.expander("Advanced parameters"):
    st.markdown(
        """
        - **Return model:** GBM (parametric) or historical bootstrap.
        - **Bootstrap settings:** block length controls how much short-term dependence is preserved.
        - **Rebalancing frequency:** daily to yearly.
        - **Market assumptions:** expected return, volatility, and risk-free rate.
        - **Inflation module status:** present in the codebase, but **not active in the displayed page metrics** by default.
        """
    )

st.info("Select a page from the sidebar to get started.")
