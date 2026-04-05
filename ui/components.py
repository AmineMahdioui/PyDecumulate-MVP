"""Shared Streamlit UI components and formatters."""

import streamlit as st

from ui.charts import AMUNDI_NAVY


# ---------------------------------------------------------------------------
# Sidebar: market & simulation settings (shared across pages)
# ---------------------------------------------------------------------------
def shared_market_sidebar(
    context: str = "retirement",
    include_cppi: bool = True,
    show_n_simulations: bool = True,
    fixed_n_simulations: int | None = None,
) -> dict:
    """Render sidebar widgets for market assumptions & simulation settings.

    Returns a dict with keys that match ``run_simulation`` keyword args:
    ``expected_return``, ``market_volatility``, ``risk_free_rate``,
    ``n_simulations``, ``rebalance_freq``, ``cppi_multiplier``.
    """
    context_defaults = {
        "accumulation": dict(expected_return=8.0, market_volatility=15.0, risk_free_rate=2.0, n_simulations=1000),
        "retirement": dict(expected_return=5.0, market_volatility=15.0, risk_free_rate=1.0, n_simulations=1000),
        "lifecycle": dict(expected_return=5.0, market_volatility=15.0, risk_free_rate=1.0, n_simulations=1000),
    }
    defaults = context_defaults.get(context, context_defaults["retirement"])

    cppi_multiplier = 1.0
    if include_cppi:
        with st.sidebar.expander("CPPI (Advanced)", expanded=False):
            cppi_multiplier = st.number_input(
                "CPPI Multiplier (m)",
                min_value=1.0,
                max_value=10.0,
                value=1.0,
                step=0.1,
                format="%.2f",
                help="Multiplier m used by CPPI: risky allocation = m × cushion.",
            )

    st.sidebar.header("Market Assumptions")

    expected_return = st.sidebar.slider(
        "Expected Market Return (%)",
        min_value=0.0,
        max_value=20.0,
        value=float(defaults["expected_return"]),
        step=0.5,
    )
    market_volatility = st.sidebar.slider(
        "Market Volatility (%)",
        min_value=1.0,
        max_value=50.0,
        value=float(defaults["market_volatility"]),
        step=0.5,
    )
    risk_free_rate = st.sidebar.slider(
        "Risk-Free Rate (%)",
        min_value=0.0,
        max_value=10.0,
        value=float(defaults["risk_free_rate"]),
        step=0.25,
    )

    st.sidebar.header("Simulation")

    simulation_method = st.sidebar.radio(
        "Return Model",
        options=["GBM (Parametric)", "Historical Bootstrap"],
        index=0,
        help="GBM uses the Expected Return & Volatility sliders to simulate returns. Historical Bootstrap resamples real market data (requires internet).",
    )

    if simulation_method == "Historical Bootstrap":
        st.sidebar.warning(
            "⚠️ Historical Bootstrap is not mathematically validated yet."
            "Market assumptions sliders (return & volatility) are ignored.",
            icon=None,
        )

    block_size = 1
    if simulation_method == "Historical Bootstrap":
        block_size = st.sidebar.slider(
            "Block Size (Bootstrap)",
            min_value=1,
            max_value=24,
            value=1,
            step=1,
            help="Number of consecutive periods per bootstrap block. 1 = i.i.d. resample; larger values preserve short-term autocorrelation.",
        )

    if show_n_simulations:
        n_simulations = st.sidebar.slider(
            "Monte-Carlo Simulations",
            min_value=100,
            max_value=5000,
            value=int(defaults["n_simulations"]),
            step=100,
        )
    else:
        n_simulations = fixed_n_simulations if fixed_n_simulations is not None else int(defaults["n_simulations"])
    rebalance_freq = st.sidebar.selectbox(
        "Rebalancing Frequency",
        options=["daily", "weekly", "monthly", "quarterly", "yearly"],
        index=2,
    )

    return dict(
        cppi_multiplier=cppi_multiplier,
        expected_return=expected_return,
        market_volatility=market_volatility,
        risk_free_rate=risk_free_rate,
        n_simulations=n_simulations,
        rebalance_freq=rebalance_freq,
        simulation_method=simulation_method,
        block_size=block_size,
    )


# ---------------------------------------------------------------------------
# Model Caveats — shared across all pages
# ---------------------------------------------------------------------------
CAVEATS_TEXT = """\
**Model Caveats**

- GBM assumes constant expected return and volatility — no regime switches, mean-reversion, or fat tails.
- Lifecycle charts and KPIs are shown in **nominal euro terms** unless explicitly labelled otherwise; the inflation module exists in the codebase but is not yet active in the displayed page outputs.
- The lifecycle page currently uses **CPPI in accumulation** and a **Constant Mix decumulation handoff**. It is pathwise, but it is not yet a configurable multi-strategy lifecycle engine.
- CPPI includes a model floor mechanism (soft cushion constraint); Glidepath and Constant Mix do not provide equivalent floor protection.
- Retirement comparisons currently support **Constant Mix** and **Glidepath** decumulation.
- Withdrawals are fixed in nominal terms on the sustainability page unless a future inflation-aware mode is explicitly enabled.
- No taxes, transaction costs, liquidity constraints, or stochastic longevity are modelled in the displayed dashboard outputs.
"""

ROADMAP_TEXT = """\
- **Stochastic Inflation:** Vasicek / CIR process to stress-test inflation spikes.
- **Longevity Module:** Gompertz-Makeham mortality table for stochastic death dates per path.
- **RDUM:** Ruin-Date Utility Maximisation — dynamically adjusting the risky multiplier *m* based on remaining wealth cushion.
- **Private Asset Integration:** Illiquid Real Assets (PE / Infra) in early accumulation to capture the illiquidity premium.
- **Turnover / Cost Overlay:** Transaction cost modelling per rebalance.
"""


def build_model_caveats_panel() -> None:
    """Render a visible caveats box + expandable technical roadmap."""
    st.info(CAVEATS_TEXT)
    with st.expander("Technical Roadmap"):
        st.markdown(ROADMAP_TEXT)


def build_pos_hero(prob_success: float, label: str = "Probability of Success") -> None:
    """Render a large metric tile with conditional colour and probability bar.

    Colour thresholds:
      ≥ 70 % → green | 50–70 % → amber | < 50 % → red
    """
    if prob_success >= 70:
        colour = "#007A33"
    elif prob_success >= 50:
        colour = "#D4A017"
    else:
        colour = "#C62828"

    st.markdown(
        f"""
        <div style="
            background: {colour}15;
            border-left: 5px solid {colour};
            padding: 16px 20px 10px 20px;
            border-radius: 6px;
            margin-bottom: 4px;
        ">
            <div style="font-size: 0.85rem; color: {AMUNDI_NAVY}; font-weight: 600;">
                {label}
            </div>
            <div style="font-size: 2.4rem; color: {colour}; font-weight: 700; line-height: 1.15;">
                {prob_success:.1f} %
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(int(prob_success), 100))


def format_delta_metric(
    value: float,
    baseline: float,
    currency: bool = True,
    inverse: bool = False,
) -> tuple[str, str]:
    """Return (formatted_value, delta_str) for use with st.metric(..., delta=...)."""
    delta_abs = value - baseline
    delta_pct = (delta_abs / baseline * 100.0) if baseline != 0 else 0.0
    sign = "+" if delta_abs >= 0 else ""
    if currency:
        value_str = f"€ {value:,.0f}"
        delta_str = f"{sign}€ {delta_abs:,.0f} ({delta_pct:+.1f}% vs baseline)"
    else:
        value_str = f"{value:.1f}%"
        delta_str = f"{delta_pct:+.1f}% vs baseline"
    return value_str, delta_str
