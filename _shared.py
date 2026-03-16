"""
Shared helpers, constants and simulation runner used across all pages.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import plotly.express as px

from simulator import (
    AccCPPIEngine,
    AccConstantMixEngine,
    AccGlidepathEngine,
    AccLinearGlidepath,
    DecConstantMixEngine,
    DecGlidepathEngine,
    MarketSimulator,
    MonteCarloAnalyzer,
    StrategyParameters,
    HistoricalBootstrapSimulator,
    LifecycleParameters,
    LifecycleResult,
    LifecycleSimulator,
)

# ---------------------------------------------------------------------------
# Amundi-inspired colour palette
# ---------------------------------------------------------------------------
AMUNDI_CYAN = "#009FE3"
AMUNDI_NAVY = "#001C4B"
AMUNDI_WHITE = "#FFFFFF"
AMUNDI_LIGHT_BLUE = "#C7D9E3"
AMUNDI_GREY = "#58595B"
AMUNDI_GREEN = "#007A33"

# ---------------------------------------------------------------------------
# Frequency mapping (rebalance label → pandas offset alias)
# ---------------------------------------------------------------------------
FREQ_TO_PD_OFFSET: dict[str, str] = {
    "daily": "B",
    "weekly": "W",
    "monthly": "ME",
    "quarterly": "QE",
    "yearly": "YE",
}

# ---------------------------------------------------------------------------
# Cached simulation runner
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Running Monte-Carlo simulation …")
def run_simulation(
    initial_wealth: float,
    time_horizon: int,
    cppi_multiplier: float,
    floor_pct: float,
    expected_return: float,
    market_volatility: float,
    risk_free_rate: float,
    n_simulations: int,
    rebalance_freq: str,
    annual_withdrawal: float,
    annual_contribution: float,
    lambda_pct: float = 60.0,
    simulation_method: str = "GBM (Parametric)",
    block_length: int = 1,
    strategy_type: str = "CPPI",    # Glidepath-specific kwargs (ignored for other strategy types)
    glidepath_initial: float = 0.80,
    glidepath_final: float = 0.20,
    glidepath_shape: str = "linear",) -> dict:
    """Execute the full MC pipeline and return serialisable results."""
    params = StrategyParameters(
        initial_wealth=initial_wealth,
        time_horizon=time_horizon,
        cppi_multiplier=cppi_multiplier,
        floor_pct=floor_pct,
        expected_return=expected_return,
        market_volatility=market_volatility,
        risk_free_rate=risk_free_rate,
        n_simulations=n_simulations,
        rebalance_freq=rebalance_freq,
        annual_withdrawal=annual_withdrawal,
        annual_contribution=annual_contribution,
        Lambda=lambda_pct,
    )
    p = LifecycleParameters(
        n_simulations=n_simulations,
        time_horizon=time_horizon,
        rebalance_freq=rebalance_freq,
        cppi_multiplier=cppi_multiplier,
        floor_pct=floor_pct,
        risk_free_rate=risk_free_rate,
        annual_withdrawal=annual_withdrawal,
    )

    if simulation_method == "GBM (Parametric)":
        market = MarketSimulator(params)
        sim_returns = {"equity": market.generate_returns(seed=42)}
    else:
        sim_returns = HistoricalBootstrapSimulator(p).generate_returns(
            seed=42,
            block_size=block_length,
        )

    if params.is_decumulation:
        if strategy_type == "CPPI":
            raise ValueError(
                "Decumulation CPPI is deprecated and unsupported in the app. "
                "Use 'CM' or 'Glidepath' for retirement comparisons."
            )
        if strategy_type in {"Glidepath", "GP"}:
            engine = DecGlidepathEngine(
                params,
                initial_equity=glidepath_initial,
                final_equity=glidepath_final,
                shape=glidepath_shape,
            )
        else:
            engine = DecConstantMixEngine(params)
    else:
        if strategy_type == "CPPI":
            engine = AccCPPIEngine(params)
        elif strategy_type == "Glidepath":
            engine = AccGlidepathEngine(
                params,
                initial_equity=glidepath_initial,
                final_equity=glidepath_final,
                shape=glidepath_shape,
            )
        else:
            engine = AccConstantMixEngine(params)
    result = engine.run(sim_returns["equity"])

    analyzer = MonteCarloAnalyzer(result, params)

    dates = pd.date_range(
        start="2025-01-01",
        periods=params.n_steps,
        freq=FREQ_TO_PD_OFFSET[params.rebalance_freq],
    )

    # Cumulative contributions for accumulation; None for decumulation
    contribution_cumsum = (
        result.contributions_made.cumsum(axis=0).mean(axis=1)
        if hasattr(result, "contributions_made")
        else None
    )

    return {
        "prob_success": analyzer.probability_of_success(),
        "expected_shortfall": analyzer.expected_shortfall(),
        "median_ending": analyzer.median_ending_wealth(),
        "median_mdd": analyzer.median_max_drawdown(),
        "percentiles": analyzer.percentile_paths(),
        "ending_wealth": analyzer.ending_wealth_array(),
        "floor_values": result.floor_values,
        "risky_mean": result.risky_paths.mean(axis=1),
        "summary_df": analyzer.summary_dataframe(dates),
        "dates": dates,
        # new analytics
        "allocation_percentiles": analyzer.allocation_percentile_paths(),
        "survival_times": analyzer.survival_time_per_path(),
        "contribution_cumsum": contribution_cumsum,
        "withdrawal_percentiles": analyzer.withdrawal_percentile_paths(),
        "lambda": params.Lambda / 100.0,  # Convert percentage to decimal
    }


# ---------------------------------------------------------------------------
# Lifecycle simulation runner (pathwise handoff: acc terminal state → dec)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Running lifecycle simulation …")
def run_lifecycle_simulation(
    # Accumulation phase params
    acc_initial_wealth: float,
    acc_time_horizon: int,
    acc_contribution: float,
    acc_floor_pct: float,
    acc_cppi_multiplier: float,
    # Decumulation phase params
    dec_time_horizon: int,
    dec_withdrawal: float,
    dec_floor_pct: float = 0.0,
    # Shared market params
    expected_return: float = 8.0,
    market_volatility: float = 15.0,
    risk_free_rate: float = 2.0,
    n_simulations: int = 1000,
    rebalance_freq: str = "monthly",
    simulation_method: str = "GBM (Parametric)",
    block_length: int = 1,
    annual_inflation_rate: float = 0.0,
    Lambda: float = 60.0,
    lifecycle_mode: str = "CPPI_TO_CM",
    acc_glidepath_initial: float = 0.80,
    acc_glidepath_final: float = 0.20,
    acc_glidepath_shape: str = "linear",
    dec_glidepath_initial: float = 0.60,
    dec_glidepath_final: float = 0.30,
    dec_glidepath_shape: str = "linear",
) -> tuple[dict, dict, float]:
    """Run a full lifecycle simulation with pathwise retirement handoff.

    The function supports two lifecycle modes:
    - ``CPPI_TO_CM``
    - ``GLIDEPATH_TO_GLIDEPATH``

    It returns page-friendly dictionaries for the accumulation and decumulation
    phases, plus the median nominal wealth at retirement.
    """
    acc_params = StrategyParameters(
        initial_wealth=acc_initial_wealth,
        time_horizon=acc_time_horizon,
        cppi_multiplier=acc_cppi_multiplier,
        floor_pct=acc_floor_pct,
        expected_return=expected_return,
        market_volatility=market_volatility,
        risk_free_rate=risk_free_rate,
        n_simulations=n_simulations,
        rebalance_freq=rebalance_freq,
        annual_contribution=acc_contribution,
        annual_inflation_rate=annual_inflation_rate,
    )
    dec_params = StrategyParameters(
        initial_wealth=0.0,
        time_horizon=dec_time_horizon,
        cppi_multiplier=acc_cppi_multiplier,
        floor_pct=dec_floor_pct,
        expected_return=expected_return,
        market_volatility=market_volatility,
        risk_free_rate=risk_free_rate,
        n_simulations=n_simulations,
        rebalance_freq=rebalance_freq,
        annual_withdrawal=dec_withdrawal,
        annual_inflation_rate=annual_inflation_rate,
        Lambda=Lambda,
    )

    if simulation_method == "Historical Bootstrap":
        bootstrap_params = LifecycleParameters(
            n_simulations=n_simulations,
            time_horizon=max(acc_time_horizon, dec_time_horizon),
            rebalance_freq=rebalance_freq,
            cppi_multiplier=acc_cppi_multiplier,
            floor_pct=acc_floor_pct,
            risk_free_rate=risk_free_rate,
            annual_withdrawal=dec_withdrawal,
        )
        bootstrap = HistoricalBootstrapSimulator(bootstrap_params)
        acc_returns = bootstrap.generate_returns(
            n_steps=acc_params.n_steps,
            n_simulations=n_simulations,
            seed=42,
            block_size=block_length,
        )["equity"]
        dec_returns = bootstrap.generate_returns(
            n_steps=dec_params.n_steps,
            n_simulations=n_simulations,
            seed=123,
            block_size=block_length,
        )["equity"]
    else:
        acc_returns = MarketSimulator(acc_params).generate_returns(seed=42)
        dec_returns = MarketSimulator(dec_params).generate_returns(seed=123)

    lc: LifecycleResult = LifecycleSimulator(
        acc_params,
        dec_params,
        lifecycle_mode=lifecycle_mode,
        acc_glidepath_initial=acc_glidepath_initial,
        acc_glidepath_final=acc_glidepath_final,
        acc_glidepath_shape=acc_glidepath_shape,
        dec_glidepath_initial=dec_glidepath_initial,
        dec_glidepath_final=dec_glidepath_final,
        dec_glidepath_shape=dec_glidepath_shape,
    ).run(acc_returns, dec_returns)

    retirement_pot_nominal = float(np.median(lc.retirement_wealths_nominal))
    retirement_p5_nominal = float(np.percentile(lc.retirement_wealths_nominal, 5))
    retirement_risky_alloc_median = float(np.median(lc.retirement_risky_allocation) * 100.0)

    acc_analyzer = MonteCarloAnalyzer(lc.accumulation, acc_params)
    dec_analyzer = MonteCarloAnalyzer(lc.decumulation, dec_params)

    acc_dates = pd.date_range(
        start="2025-01-01",
        periods=acc_params.n_steps,
        freq=FREQ_TO_PD_OFFSET[acc_params.rebalance_freq],
    )
    dec_dates = pd.date_range(
        start="2025-01-01",
        periods=dec_params.n_steps,
        freq=FREQ_TO_PD_OFFSET[dec_params.rebalance_freq],
    )

    contribution_cumsum = lc.accumulation.contributions_made.cumsum(axis=0).mean(axis=1)

    acc_pv = lc.accumulation.portfolio_values
    acc_floor = lc.accumulation.floor_values[:, np.newaxis]
    floor_contact = acc_pv <= acc_floor
    floor_touch_path_pct = float(floor_contact.any(axis=0).mean() * 100.0)
    floor_touch_time_pct = float(floor_contact.mean() * 100.0)

    sim_acc = {
        "prob_success": acc_analyzer.probability_of_success(),
        "expected_shortfall": acc_analyzer.expected_shortfall(),
        "median_ending": acc_analyzer.median_ending_wealth(),
        "median_mdd": acc_analyzer.median_max_drawdown(),
        "percentiles": acc_analyzer.percentile_paths(),
        "ending_wealth": acc_analyzer.ending_wealth_array(),
        "floor_values": lc.accumulation.floor_values,
        "risky_mean": lc.accumulation.risky_paths.mean(axis=1),
        "summary_df": acc_analyzer.summary_dataframe(acc_dates),
        "dates": acc_dates,
        "allocation_percentiles": acc_analyzer.allocation_percentile_paths(),
        "survival_times": acc_analyzer.survival_time_per_path(),
        "contribution_cumsum": contribution_cumsum,
        "withdrawal_percentiles": acc_analyzer.withdrawal_percentile_paths(),
        "lambda": acc_params.Lambda / 100.0,
        "retirement_wealths_nominal": lc.retirement_wealths_nominal,
        "retirement_risky_allocation": lc.retirement_risky_allocation,
        "retirement_p5_nominal": retirement_p5_nominal,
        "retirement_risky_alloc_median": retirement_risky_alloc_median,
        "floor_touch_path_pct": floor_touch_path_pct,
        "floor_touch_time_pct": floor_touch_time_pct,
    }

    sim_dec = {
        "prob_success": dec_analyzer.probability_of_success(),
        "expected_shortfall": dec_analyzer.expected_shortfall(),
        "median_ending": dec_analyzer.median_ending_wealth(),
        "median_mdd": dec_analyzer.median_max_drawdown(),
        "percentiles": dec_analyzer.percentile_paths(),
        "ending_wealth": dec_analyzer.ending_wealth_array(),
        "floor_values": lc.decumulation.floor_values,
        "risky_mean": lc.decumulation.risky_paths.mean(axis=1),
        "summary_df": dec_analyzer.summary_dataframe(dec_dates),
        "dates": dec_dates,
        "allocation_percentiles": dec_analyzer.allocation_percentile_paths(),
        "survival_times": dec_analyzer.survival_time_per_path(),
        "contribution_cumsum": None,
        "withdrawal_percentiles": dec_analyzer.withdrawal_percentile_paths(),
        "lambda": dec_params.Lambda / 100.0,
    }

    return sim_acc, sim_dec, retirement_pot_nominal


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
        st.sidebar.header("CPPI Strategy")
        cppi_multiplier = st.sidebar.slider(
            "CPPI Multiplier (m)",
            min_value=1.0,
            max_value=10.0,
            value=1.0 if context == "lifecycle" else 3.0,
            step=0.5,
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

    block_length = 1
    if simulation_method == "Historical Bootstrap":
        block_length = st.sidebar.slider(
            "Block Length (Bootstrap)",
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
        block_length=block_length,
    )


# ---------------------------------------------------------------------------
# Reusable chart builders
# ---------------------------------------------------------------------------
def build_fan_chart(
    sim_data: dict,
    title: str,
    floor_label: str = "Floor",
    band_color: str = "0,122,51",
    median_color: str = AMUNDI_CYAN,
    median_label: str = "Median Portfolio",
    show_risky_mean: bool = True,
) -> go.Figure:
    """Return a Plotly fan chart for a single simulation result."""
    pcts = sim_data["percentiles"]
    dates = sim_data["dates"]
    fig = go.Figure()

    # P5-P95
    fig.add_trace(go.Scatter(
        x=dates, y=pcts["P95"], mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=pcts["P5"], mode="lines",
        line=dict(width=0), fill="tonexty",
        fillcolor=f"rgba({band_color},0.10)", name="P5 – P95",
    ))
    # P25-P75
    fig.add_trace(go.Scatter(
        x=dates, y=pcts["P75"], mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=pcts["P25"], mode="lines",
        line=dict(width=0), fill="tonexty",
        fillcolor=f"rgba({band_color},0.25)", name="P25 – P75",
    ))
    # Median
    fig.add_trace(go.Scatter(
        x=dates, y=pcts["P50"], mode="lines",
        name=median_label, line=dict(color=median_color, width=3),
    ))
    # Floor
    fig.add_trace(go.Scatter(
        x=dates, y=sim_data["floor_values"], mode="lines",
        name=floor_label, line=dict(color=median_color, width=2, dash="dash"),
    ))
    if show_risky_mean:
        fig.add_trace(go.Scatter(
            x=dates, y=sim_data["risky_mean"], mode="lines",
            name="Risky Asset (Mean)", line=dict(color=AMUNDI_GREY, width=2, dash="dot"),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value (€)",
        template="plotly_white",
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=60, b=40),
        font=dict(family="Arial, sans-serif", size=13, color=AMUNDI_NAVY),
    )
    return fig


def build_histogram(sim_data: dict, title: str = "", color: str = AMUNDI_CYAN) -> go.Figure:
    """Return a Plotly histogram of ending wealth."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=sim_data["ending_wealth"], nbinsx=60,
        marker_color=color, opacity=0.75, name="Ending Wealth",
    ))
    fig.add_vline(
        x=float(sim_data["floor_values"][-1]),
        line_dash="dash", line_color=color,
        annotation_text="Floor", annotation_position="top left",
    )
    fig.add_vline(
        x=float(sim_data["median_ending"]),
        line_dash="solid", line_color=AMUNDI_NAVY,
        annotation_text="Median", annotation_position="top right",
    )
    fig.update_layout(
        title=title,
        xaxis_title="Terminal Portfolio Value (€)",
        yaxis_title="Frequency",
        template="plotly_white",
        height=380,
        margin=dict(l=60, r=30, t=40, b=40),
        font=dict(family="Arial, sans-serif", size=13, color=AMUNDI_NAVY),
    )
    return fig


# ---------------------------------------------------------------------------
# Chart 3 — Mountain chart (lifecycle: accumulation → decumulation)
# Inspired by Research Figure 1 & 3: the complete investor lifecycle
# ---------------------------------------------------------------------------
def build_mountain_chart(
    sim_acc: dict,
    sim_dec: dict,
    acc_dates,
    dec_dates,
) -> go.Figure:
    """Lifecycle mountain chart.

    Shows the accumulation phase building up to a peak at retirement, then
    the decumulation drawdown — the complete wealth lifecycle in one figure.
    A "contributions base" layer visually separates invested capital from
    investment returns during accumulation.
    """
    acc_pcts = sim_acc["percentiles"]
    dec_pcts = sim_dec["percentiles"]
    retirement_date_str = str(acc_dates[-1])

    fig = go.Figure()

    # --- Accumulation P5–P95 outer band ---
    fig.add_trace(go.Scatter(
        x=acc_dates, y=acc_pcts["P95"], mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=acc_dates, y=acc_pcts["P5"], mode="lines",
        line=dict(width=0), fill="tonexty",
        fillcolor="rgba(0,159,227,0.08)", name="Acc P5–P95",
    ))
    # --- Accumulation P25–P75 inner band ---
    fig.add_trace(go.Scatter(
        x=acc_dates, y=acc_pcts["P75"], mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=acc_dates, y=acc_pcts["P25"], mode="lines",
        line=dict(width=0), fill="tonexty",
        fillcolor="rgba(0,159,227,0.18)", name="Acc P25–P75",
    ))
    # --- Contributions base layer (invested capital vs investment returns) ---
    contrib_cumsum = sim_acc.get("contribution_cumsum")
    if contrib_cumsum is not None:
        fig.add_trace(go.Scatter(
            x=acc_dates, y=contrib_cumsum, mode="lines",
            fill="tozeroy", fillcolor="rgba(0,122,51,0.20)",
            line=dict(color=AMUNDI_GREEN, width=1, dash="dot"),
            name="Cumulative Contributions",
        ))
    # --- Accumulation median ---
    fig.add_trace(go.Scatter(
        x=acc_dates, y=acc_pcts["P50"], mode="lines",
        name="Accumulation Median", line=dict(color=AMUNDI_CYAN, width=3),
        fill="tozeroy", fillcolor="rgba(0,159,227,0.12)",
    ))

    # --- Decumulation P5–P95 outer band ---
    fig.add_trace(go.Scatter(
        x=dec_dates, y=dec_pcts["P95"], mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=dec_dates, y=dec_pcts["P5"], mode="lines",
        line=dict(width=0), fill="tonexty",
        fillcolor="rgba(0,122,51,0.08)", name="Dec P5–P95",
    ))
    # --- Decumulation P25–P75 inner band ---
    fig.add_trace(go.Scatter(
        x=dec_dates, y=dec_pcts["P75"], mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=dec_dates, y=dec_pcts["P25"], mode="lines",
        line=dict(width=0), fill="tonexty",
        fillcolor="rgba(0,122,51,0.18)", name="Dec P25–P75",
    ))
    # --- Decumulation median ---
    fig.add_trace(go.Scatter(
        x=dec_dates, y=dec_pcts["P50"], mode="lines",
        name="Decumulation Median", line=dict(color=AMUNDI_GREEN, width=3),
        fill="tozeroy", fillcolor="rgba(0,122,51,0.12)",
    ))

    # --- Floor lines ---
    fig.add_trace(go.Scatter(
        x=acc_dates, y=sim_acc["floor_values"], mode="lines",
        name="Protection Floor", line=dict(color=AMUNDI_CYAN, width=1.5, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=dec_dates, y=sim_dec["floor_values"], mode="lines",
        name="Guaranteed Floor", line=dict(color=AMUNDI_GREEN, width=1.5, dash="dash"),
    ))

    # --- Retirement vertical line ---
    fig.add_vline(x=retirement_date_str, line_dash="solid",
                  line_color=AMUNDI_NAVY, line_width=2)
    fig.add_annotation(
        x=retirement_date_str, y=1.06, yref="paper",
        text="<b>Retirement</b>", showarrow=False,
        font=dict(size=13, color=AMUNDI_NAVY), xanchor="left",
    )
    fig.add_vrect(
        x0=str(acc_dates[0]), x1=retirement_date_str,
        fillcolor="rgba(0,159,227,0.03)", layer="below", line_width=0,
    )
    fig.add_vrect(
        x0=retirement_date_str, x1=str(dec_dates[-1]),
        fillcolor="rgba(0,122,51,0.03)", layer="below", line_width=0,
    )

    fig.update_layout(
        title="Full Lifecycle — The Wealth Mountain",
        xaxis_title="Date",
        yaxis_title="Portfolio Value (€)",
        template="plotly_white",
        height=560,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=80, b=40),
        font=dict(family="Arial, sans-serif", size=13, color=AMUNDI_NAVY),
    )
    return fig


# ---------------------------------------------------------------------------
# Chart 3b; Mountain chart with Investor Age x-axis (lifecycle page)
# ---------------------------------------------------------------------------
def build_mountain_chart_age(
    sim_acc: dict,
    sim_dec: dict,
    age_axis_acc: np.ndarray,
    age_axis_dec: np.ndarray,
    include_contributions: bool = False,
) -> go.Figure:
    """Lifecycle mountain chart with Investor Age on the x-axis.

    Identical visual logic to ``build_mountain_chart`` but uses numeric
    age arrays instead of ``DatetimeIndex``.
    """
    acc_pcts = sim_acc["percentiles"]
    dec_pcts = sim_dec["percentiles"]
    retirement_age = float(age_axis_acc[-1])

    fig = go.Figure()

    # --- Accumulation P5–P95 outer band ---
    fig.add_trace(go.Scatter(
        x=age_axis_acc, y=acc_pcts["P95"], mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=age_axis_acc, y=acc_pcts["P5"], mode="lines",
        line=dict(width=0), fill="tonexty",
        fillcolor="rgba(0,159,227,0.08)", name="Acc P5–P95",
    ))
    # --- Accumulation P25–P75 inner band ---
    fig.add_trace(go.Scatter(
        x=age_axis_acc, y=acc_pcts["P75"], mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=age_axis_acc, y=acc_pcts["P25"], mode="lines",
        line=dict(width=0), fill="tonexty",
        fillcolor="rgba(0,159,227,0.18)", name="Acc P25–P75",
    ))
    # --- Optional contributions base layer ---
    contrib_cumsum = sim_acc.get("contribution_cumsum")
    if include_contributions and contrib_cumsum is not None:
        fig.add_trace(go.Scatter(
            x=age_axis_acc, y=contrib_cumsum, mode="lines",
            fill="tozeroy", fillcolor="rgba(0,122,51,0.20)",
            line=dict(color=AMUNDI_GREEN, width=1, dash="dot"),
            name="Cumulative Contributions",
        ))
    # --- Accumulation median ---
    fig.add_trace(go.Scatter(
        x=age_axis_acc, y=acc_pcts["P50"], mode="lines",
        name="Accumulation Median", line=dict(color=AMUNDI_CYAN, width=3),
        fill="tozeroy", fillcolor="rgba(0,159,227,0.12)",
    ))

    # --- Decumulation P5–P95 outer band ---
    fig.add_trace(go.Scatter(
        x=age_axis_dec, y=dec_pcts["P95"], mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=age_axis_dec, y=dec_pcts["P5"], mode="lines",
        line=dict(width=0), fill="tonexty",
        fillcolor="rgba(0,122,51,0.08)", name="Dec P5–P95",
    ))
    # --- Decumulation P25–P75 inner band ---
    fig.add_trace(go.Scatter(
        x=age_axis_dec, y=dec_pcts["P75"], mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=age_axis_dec, y=dec_pcts["P25"], mode="lines",
        line=dict(width=0), fill="tonexty",
        fillcolor="rgba(0,122,51,0.18)", name="Dec P25–P75",
    ))
    # --- Decumulation median ---
    fig.add_trace(go.Scatter(
        x=age_axis_dec, y=dec_pcts["P50"], mode="lines",
        name="Decumulation Median", line=dict(color=AMUNDI_GREEN, width=3),
        fill="tozeroy", fillcolor="rgba(0,122,51,0.12)",
    ))

    # --- Floor lines ---
    fig.add_trace(go.Scatter(
        x=age_axis_acc, y=sim_acc["floor_values"], mode="lines",
        name="Protection Floor", line=dict(color=AMUNDI_CYAN, width=1.5, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=age_axis_dec, y=sim_dec["floor_values"], mode="lines",
        name="Guaranteed Floor", line=dict(color=AMUNDI_GREEN, width=1.5, dash="dash"),
    ))

    # --- Retirement vertical line ---
    fig.add_vline(x=retirement_age, line_dash="solid",
                  line_color=AMUNDI_NAVY, line_width=2)
    fig.add_annotation(
        x=retirement_age, y=1.06, yref="paper",
        text="<b>Retirement</b>", showarrow=False,
        font=dict(size=13, color=AMUNDI_NAVY), xanchor="left",
    )
    fig.add_vrect(
        x0=float(age_axis_acc[0]), x1=retirement_age,
        fillcolor="rgba(0,159,227,0.03)", layer="below", line_width=0,
    )
    fig.add_vrect(
        x0=retirement_age, x1=float(age_axis_dec[-1]),
        fillcolor="rgba(0,122,51,0.03)", layer="below", line_width=0,
    )

    fig.update_layout(
        title="Full Lifecycle — Wealth Path",
        xaxis_title="Investor Age",
        yaxis_title="Portfolio Value (€)",
        template="plotly_white",
        height=560,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=80, b=40),
        font=dict(family="Arial, sans-serif", size=13, color=AMUNDI_NAVY),
    )
    return fig


# ---------------------------------------------------------------------------
# Chart 3c; Cash Flow Timeline (lifecycle page)
# ---------------------------------------------------------------------------
def build_cash_flow_timeline(
    acc_contribution: float,
    dec_withdrawal: float,
    acc_horizon: int,
    dec_horizon: int,
    start_age: int,
) -> go.Figure:
    """Bar chart showing savings (positive) and withdrawals (negative) by age.

    X-axis = Investor Age (aligned with the mountain chart).
    Y-axis = Nominal (€) per year.
    """
    retirement_age = start_age + acc_horizon

    # Accumulation years (positive bars)
    acc_ages = list(range(start_age, retirement_age))
    acc_vals = [acc_contribution] * len(acc_ages)

    # Decumulation years (negative bars)
    dec_ages = list(range(retirement_age, retirement_age + dec_horizon))
    dec_vals = [-dec_withdrawal] * len(dec_ages)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=acc_ages, y=acc_vals,
        name="Annual Savings",
        marker_color=AMUNDI_GREEN,
        opacity=0.75,
    ))
    fig.add_trace(go.Bar(
        x=dec_ages, y=dec_vals,
        name="Annual Withdrawal",
        marker_color="#C62828",
        opacity=0.75,
    ))

    # Retirement line
    fig.add_vline(x=retirement_age - 0.5, line_dash="solid",
                  line_color=AMUNDI_NAVY, line_width=1.5)
    fig.add_annotation(
        x=retirement_age - 0.5, y=1.06, yref="paper",
        text="<b>Retirement</b>", showarrow=False,
        font=dict(size=11, color=AMUNDI_NAVY), xanchor="left",
    )

    fig.update_layout(
        title="Cash Flow Timeline; Money In vs Money Out",
        xaxis_title="Investor Age",
        yaxis_title="Annual Cash Flow; Nominal (€)",
        template="plotly_white",
        height=340,
        barmode="relative",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=60, b=40),
        font=dict(family="Arial, sans-serif", size=13, color=AMUNDI_NAVY),
    )
    return fig


# ---------------------------------------------------------------------------
# Chart 4; Glide path comparison (CPPI dynamic vs Constant Mix static)
# Inspired by Research Figure 18: constrained vs unconstrained glide path
# ---------------------------------------------------------------------------
def build_glide_path_comparison(
    sim_cppi: dict,
    sim_cm: dict,
    dates,
    title: str = "Dynamic CPPI vs Constant Mix — Risky Asset Allocation",
) -> go.Figure:
    """Overlay CPPI's dynamic allocation fan against a flat CM allocation.

    Illustrates the key insight from Figure 18: applying real-world constraints
    (CPPI floor, no leverage) transforms the theoretically optimal constant-mix
    into a concave dynamic glide path that actively de-risks near the floor.
    """
    cppi_alloc = sim_cppi["allocation_percentiles"]
    cm_alloc   = sim_cm["allocation_percentiles"]

    fig = go.Figure()

    # CPPI P5–P95 outer band
    fig.add_trace(go.Scatter(
        x=dates, y=cppi_alloc["P95"], mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=cppi_alloc["P5"], mode="lines",
        line=dict(width=0), fill="tonexty",
        fillcolor="rgba(0,159,227,0.10)", name="CPPI P5–P95",
    ))
    # CPPI P25–P75
    fig.add_trace(go.Scatter(
        x=dates, y=cppi_alloc["P75"], mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=cppi_alloc["P25"], mode="lines",
        line=dict(width=0), fill="tonexty",
        fillcolor="rgba(0,159,227,0.25)", name="CPPI P25–P75",
    ))
    # CPPI median
    fig.add_trace(go.Scatter(
        x=dates, y=cppi_alloc["P50"], mode="lines",
        name="CPPI — Dynamic (Constrained)", line=dict(color=AMUNDI_CYAN, width=3),
    ))
    # CM median — flat reference
    fig.add_trace(go.Scatter(
        x=dates, y=cm_alloc["P50"], mode="lines",
        name="Constant Mix — Fixed Allocation",
        line=dict(color=AMUNDI_GREY, width=2, dash="dash"),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Risky Asset Allocation (%)",
        yaxis=dict(range=[0, 105], ticksuffix="%"),
        template="plotly_white",
        height=440,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=60, b=40),
        font=dict(family="Arial, sans-serif", size=13, color=AMUNDI_NAVY),
    )
    return fig


# ---------------------------------------------------------------------------
# Chart 5 — Stacked allocation area chart
# Inspired by Research Figure 25 & 26: portfolio composition over time
# ---------------------------------------------------------------------------
def build_stacked_allocation_chart(
    sim_data: dict,
    dates,
    title: str = "Portfolio Composition — Risky vs Safe Assets",
) -> go.Figure:
    """Stacked area showing how CPPI dynamically splits the portfolio.

    The risky band (blue) shrinks toward the floor when markets are weak;
    the safe / money-market band (grey) expands to absorb the cushion.
    Uses the median allocation path.
    """
    alloc = sim_data["allocation_percentiles"]
    risky_med = alloc["P50"]
    safe_med  = 100.0 - risky_med

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=risky_med, mode="lines",
        stackgroup="alloc",
        fillcolor="rgba(0,159,227,0.55)",
        line=dict(color=AMUNDI_CYAN, width=1),
        name="Risky Asset (Median)",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=safe_med, mode="lines",
        stackgroup="alloc",
        fillcolor="rgba(88,89,91,0.25)",
        line=dict(color=AMUNDI_GREY, width=1),
        name="Safe / Money Market (Median)",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Allocation (%)",
        yaxis=dict(range=[0, 100], ticksuffix="%"),
        template="plotly_white",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=60, b=40),
        font=dict(family="Arial, sans-serif", size=13, color=AMUNDI_NAVY),
    )
    return fig


# ---------------------------------------------------------------------------
# Chart 6 — Survival time comparison (CPPI vs Constant Mix)
# Inspired by Research Figure 9 & 10: RDUM vs SWR survival quantiles
# ---------------------------------------------------------------------------
def build_survival_comparison(
    sim_primary: dict,
    sim_secondary: dict,
    title: str = "Strategy Comparison — Portfolio Survival Time",
    time_horizon: int | None = None,
    primary_label: str = "Primary Strategy",
    secondary_label: str = "Secondary Strategy",
) -> go.Figure:
    """Side-by-side box plots of survival time for two strategies.

    Directly mirrors the research finding (Figure 9 & 10) that dynamic
    floor-protection strategies improve worst-case survival quantiles
    compared to a static withdrawal rate.  Paths that survive the full
    horizon are right-censored and shown with a dashed annotation.
    """
    st_primary = sim_primary["survival_times"]
    st_secondary = sim_secondary["survival_times"]

    fig = go.Figure()
    fig.add_trace(go.Box(
        y=st_primary, name=primary_label,
        boxpoints="outliers",
        marker_color=AMUNDI_CYAN,
        line_color=AMUNDI_NAVY,
    ))
    fig.add_trace(go.Box(
        y=st_secondary, name=secondary_label,
        boxpoints="outliers",
        marker_color=AMUNDI_GREY,
        line_color=AMUNDI_NAVY,
    ))

    # Mark the censoring threshold (simulated horizon + 1 = "survived")
    censor_y = float(max(st_primary.max(), st_secondary.max()))
    fig.add_hline(
        y=censor_y,
        line_dash="dot", line_color=AMUNDI_NAVY, line_width=1,
        annotation_text="Full horizon survived ✓",
        annotation_position="top right",
    )

    fig.update_layout(
        title=title,
        yaxis_title="Years Survived",
        template="plotly_white",
        height=440,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=60, b=40),
        font=dict(family="Arial, sans-serif", size=13, color=AMUNDI_NAVY),
    )
    return fig


# ---------------------------------------------------------------------------
# Lifecycle cash-flow timeline
# ---------------------------------------------------------------------------
def build_cash_flow_timeline(
    start_age: int,
    acc_horizon: int,
    dec_horizon: int,
    annual_contribution: float,
    annual_withdrawal: float,
) -> go.Figure:
    """Simple age-based cash-flow timeline in nominal euro terms."""
    acc_ages = np.arange(start_age, start_age + acc_horizon)
    dec_start_age = start_age + acc_horizon
    dec_ages = np.arange(dec_start_age, dec_start_age + dec_horizon)

    fig = go.Figure()
    if len(acc_ages):
        fig.add_trace(go.Bar(
            x=acc_ages,
            y=np.full(len(acc_ages), annual_contribution, dtype=float),
            name="Annual Contribution",
            marker_color=AMUNDI_CYAN,
            opacity=0.85,
        ))
    if len(dec_ages):
        fig.add_trace(go.Bar(
            x=dec_ages,
            y=-np.full(len(dec_ages), annual_withdrawal, dtype=float),
            name="Annual Withdrawal",
            marker_color=AMUNDI_GREEN,
            opacity=0.75,
        ))
        fig.add_vline(x=dec_start_age - 0.5, line_color=AMUNDI_NAVY, line_width=2)

    fig.update_layout(
        title="Lifecycle Cash-Flow Timeline",
        xaxis_title="Investor Age",
        yaxis_title="Nominal (€)",
        template="plotly_white",
        height=360,
        barmode="relative",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=60, b=40),
        font=dict(family="Arial, sans-serif", size=13, color=AMUNDI_NAVY),
    )
    return fig


# ---------------------------------------------------------------------------
# Accumulation strategy charts (age-based)
# ---------------------------------------------------------------------------
def _age_axis(start_age: int, n_points: int, steps_per_year: int) -> np.ndarray:
    return start_age + np.arange(n_points) / max(steps_per_year, 1)


def build_allocation_comparison_by_age(
    sims: dict[str, dict],
    start_age: int,
    rebalance_freq: str,
    title: str = "Risky Allocation by Investor Age",
) -> go.Figure:
    """Compare median risky allocation paths across accumulation strategies."""
    steps_per_year = {"daily": 252, "weekly": 52, "monthly": 12, "quarterly": 4, "yearly": 1}[rebalance_freq]
    fig = go.Figure()
    colors = {
        "CPPI": AMUNDI_CYAN,
        "Glidepath": AMUNDI_GREEN,
        "Constant Mix": AMUNDI_GREY,
    }
    dashes = {"CPPI": "solid", "Glidepath": "dash", "Constant Mix": "dot"}
    for label, sim in sims.items():
        ages = _age_axis(start_age, len(sim["allocation_percentiles"]["P50"]), steps_per_year)
        fig.add_trace(go.Scatter(
            x=ages,
            y=sim["allocation_percentiles"]["P50"],
            mode="lines",
            name=label,
            line=dict(color=colors.get(label, AMUNDI_NAVY), width=3 if label == "CPPI" else 2, dash=dashes.get(label, "solid")),
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Investor Age",
        yaxis_title="Risky Allocation (%)",
        yaxis=dict(range=[0, 105], ticksuffix="%"),
        template="plotly_white",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=60, b=40),
        font=dict(family="Arial, sans-serif", size=13, color=AMUNDI_NAVY),
    )
    return fig


def build_terminal_wealth_boxplot(
    strategy_wealths: dict[str, np.ndarray],
    title: str = "Terminal Wealth Distribution — Accumulation Strategies",
) -> go.Figure:
    """Box plot comparing terminal nominal wealth across strategies."""
    fig = go.Figure()
    colors = {
        "CPPI": AMUNDI_CYAN,
        "Glidepath": AMUNDI_GREEN,
        "Constant Mix": AMUNDI_GREY,
    }
    for label, wealths in strategy_wealths.items():
        fig.add_trace(go.Box(
            y=wealths,
            name=label,
            boxpoints="outliers",
            marker_color=colors.get(label, AMUNDI_NAVY),
            line_color=AMUNDI_NAVY,
        ))
    fig.update_layout(
        title=title,
        yaxis_title="Nominal (€)",
        template="plotly_white",
        height=420,
        margin=dict(l=60, r=30, t=60, b=40),
        font=dict(family="Arial, sans-serif", size=13, color=AMUNDI_NAVY),
    )
    return fig


# ---------------------------------------------------------------------------
# Survival curve
# ---------------------------------------------------------------------------
def build_survival_curve(
    sim_primary: dict,
    sim_secondary: dict | None = None,
    time_horizon: int | None = None,
    title: str = "Probability the Portfolio Is Still Alive",
    primary_label: str = "Primary Strategy",
    secondary_label: str = "Secondary Strategy",
) -> go.Figure:
    """Survival curve built from per-path survival times, not ECDF."""
    primary_times = np.asarray(sim_primary["survival_times"], dtype=float)
    if time_horizon is None:
        time_horizon = int(np.ceil(primary_times.max()))
    grid = np.arange(0, time_horizon + 1)

    def _survivor(times: np.ndarray) -> np.ndarray:
        return np.array([100.0 * np.mean(times > g) for g in grid])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=grid,
        y=_survivor(primary_times),
        mode="lines",
        name=primary_label,
        line=dict(color=AMUNDI_CYAN, width=3),
    ))
    if sim_secondary is not None:
        secondary_times = np.asarray(sim_secondary["survival_times"], dtype=float)
        fig.add_trace(go.Scatter(
            x=grid,
            y=_survivor(secondary_times),
            mode="lines",
            name=secondary_label,
            line=dict(color=AMUNDI_GREY, width=2, dash="dash"),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Years in Retirement",
        yaxis_title="Survival Probability (%)",
        yaxis=dict(range=[0, 105], ticksuffix="%"),
        template="plotly_white",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=60, b=40),
        font=dict(family="Arial, sans-serif", size=13, color=AMUNDI_NAVY),
    )
    return fig


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


# ---------------------------------------------------------------------------
# PoS Hero Component
# ---------------------------------------------------------------------------
def build_pos_hero(prob_success: float, label: str = "Probability of Success") -> None:
    """Render a large metric tile with conditional colour and probability bar.

    Colour thresholds:
      ≥ 70 % → green | 50–70 % → amber | < 50 % → red
    """
    if prob_success >= 70:
        colour = "#007A33"  # AMUNDI_GREEN
    elif prob_success >= 50:
        colour = "#D4A017"  # amber
    else:
        colour = "#C62828"  # red

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


# ---------------------------------------------------------------------------
# Glidepath Deterministic Schedule Chart
# ---------------------------------------------------------------------------
def build_glidepath_schedule_chart(
    initial_equity: float,
    final_equity: float,
    time_horizon: int,
    shape: str = "linear",
) -> go.Figure:
    """Deterministic allocation schedule for a glidepath (no simulation needed).

    Parameters are in *fraction* space (0–1).  The chart displays percentages.
    """
    glidepath = AccLinearGlidepath(
        initial_equity_allocation=initial_equity,
        final_equity_allocation=final_equity,
        years=time_horizon,
        shape=shape,
    )

    years = np.arange(0, time_horizon + 1, dtype=float)
    equity = np.array([glidepath.get_equity_allocation(int(y)) for y in years])
    hundred = np.full_like(equity, 100.0)

    fig = go.Figure()
    # Bottom area: equity (fills from 0 up to the equity line)
    fig.add_trace(go.Scatter(
        x=years, y=equity * 100, mode="lines+markers",
        name="Equity Allocation",
        fill="tozeroy", fillcolor="rgba(0,159,227,0.35)",
        line=dict(color=AMUNDI_CYAN, width=3),
        marker=dict(size=5),
    ))
    # Top area: bonds fill from the equity boundary up to 100%
    fig.add_trace(go.Scatter(
        x=years, y=hundred, mode="none",
        name="Bond / Safe Allocation",
        fill="tonexty", fillcolor="rgba(88,89,91,0.20)",
        showlegend=True,
        line=dict(color="rgba(0,0,0,0)"),
    ))
    fig.update_layout(
        title=f"Glidepath Schedule ({shape.title()}) — Deterministic Allocation",
        xaxis_title="Year",
        yaxis_title="Allocation (%)",
        yaxis=dict(range=[0, 105], ticksuffix="%"),
        template="plotly_white",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=60, b=40),
        font=dict(family="Arial, sans-serif", size=13, color=AMUNDI_NAVY),
    )
    return fig


# ---------------------------------------------------------------------------
# Sensitivity Sweep — cached
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Computing sensitivity matrix …")
def run_sensitivity_sweep(
    withdrawal_rates: list[float],
    horizons: list[int],
    initial_wealth: float,
    floor_pct: float,
    cppi_multiplier: float,
    lambda_pct: float,
    expected_return: float,
    market_volatility: float,
    risk_free_rate: float,
    rebalance_freq: str,
    simulation_method: str = "GBM (Parametric)",
    block_length: int = 1,
    strategy_type: str = "Glidepath",
    glidepath_initial: float = 0.60,
    glidepath_final: float = 0.30,
    glidepath_shape: str = "linear",
    n_sims_sweep: int = 300,
) -> pd.DataFrame:
    """Compute a PoS matrix over withdrawal rates × retirement horizons.

    ``withdrawal_rates`` are expressed as *percentages* (e.g. 4.0 means 4 %).
    The annual withdrawal for each cell = ``rate / 100 * initial_wealth``.

    Uses a reduced simulation count (``n_sims_sweep``) for responsiveness.
    """
    rows: dict[int, dict[str, float]] = {}
    for horizon in sorted(horizons):
        row: dict[str, float] = {}
        for rate in sorted(withdrawal_rates):
            ann_wd = rate / 100.0 * initial_wealth
            sim = run_simulation(
                initial_wealth=initial_wealth,
                time_horizon=horizon,
                cppi_multiplier=cppi_multiplier,
                floor_pct=floor_pct,
                expected_return=expected_return,
                market_volatility=market_volatility,
                risk_free_rate=risk_free_rate,
                n_simulations=n_sims_sweep,
                rebalance_freq=rebalance_freq,
                annual_withdrawal=ann_wd,
                annual_contribution=0.0,
                lambda_pct=lambda_pct,
                simulation_method=simulation_method,
                block_length=block_length,
                strategy_type=strategy_type,
                glidepath_initial=glidepath_initial,
                glidepath_final=glidepath_final,
                glidepath_shape=glidepath_shape,
            )
            row[f"{rate:.0f} %"] = round(sim["prob_success"], 1)
        rows[horizon] = row

    df = pd.DataFrame(rows).T
    df.index.name = "Retirement Horizon (yrs)"
    return df


# ---------------------------------------------------------------------------
# Sensitivity Heatmap
# ---------------------------------------------------------------------------
def build_sensitivity_heatmap(df: pd.DataFrame) -> go.Figure:
    """``px.imshow`` heatmap of a PoS matrix (withdrawal rate × horizon)."""
    fig = px.imshow(
        df.values,
        x=list(df.columns),
        y=[str(y) for y in df.index],
        color_continuous_scale="RdYlGn",
        zmin=0,
        zmax=100,
        text_auto=True,
        aspect="auto",
        labels=dict(
            x="Annual Withdrawal (% of Retirement Pot)",
            y="Retirement Horizon (years)",
            color="PoS %",
        ),
    )
    fig.update_layout(
        title="Sensitivity Matrix — Probability of Success (%)",
        template="plotly_white",
        height=max(300, 80 * len(df)),
        margin=dict(l=60, r=30, t=60, b=40),
        font=dict(family="Arial, sans-serif", size=13, color=AMUNDI_NAVY),
    )
    return fig
