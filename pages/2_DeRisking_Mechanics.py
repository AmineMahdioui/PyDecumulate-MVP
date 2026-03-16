"""
Page 2; Accumulation & De-Risking Mechanics
==============================================
Three sections examining how different strategies allocate between risky and
safe assets over the accumulation horizon:
  Section A; Constant Mix: fixed risky allocation
  Section B; Glidepath: deterministic schedule (linear / convex / concave)
  Section C; Strategy comparison: terminal wealth box plot
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(layout="wide")

from _shared import (
    AMUNDI_CYAN,
    AMUNDI_GREEN,
    AMUNDI_GREY,
    AMUNDI_NAVY,
    build_glidepath_schedule_chart,
    build_model_caveats_panel,
    build_stacked_allocation_chart,
    run_simulation,
    shared_market_sidebar,
)

# ---------------------------------------------------------------------------
# Sidebar; shared market + accumulation + glidepath controls
# ---------------------------------------------------------------------------
st.sidebar.title("Accumulation Setup")

st.sidebar.header("Portfolio & Horizon")
start_age = st.sidebar.number_input(
    "Current Age", min_value=20, max_value=60, value=25, step=1,
)
initial_wealth = st.sidebar.number_input(
    "Starting Pot (€)", min_value=0, max_value=10_000_000,
    value=1_000_000, step=50_000, format="%d",
)
annual_contribution = st.sidebar.number_input(
    "Annual Contribution (€)", min_value=0, max_value=10_000_000,
    value=50_000, step=5_000, format="%d",
)
time_horizon = st.sidebar.slider(
    "Savings Horizon (Years)", min_value=1, max_value=40, value=40,
)
mkt = shared_market_sidebar(context="accumulation", include_cppi=False)

# Constant Mix controls
st.sidebar.header("Constant Mix Parameters")
cm_lambda = st.sidebar.slider(
    "Risky Allocation (%)", min_value=10, max_value=100, value=60, step=5,
    help="Constant Mix: fixed percentage of portfolio held in risky assets at all times.",
)

# Glidepath-specific controls
st.sidebar.header("Glidepath Parameters")
gp_initial = st.sidebar.slider(
    "Initial Equity (%)", min_value=40, max_value=100, value=80, step=5,
) / 100.0
gp_final = st.sidebar.slider(
    "Final Equity (%)", min_value=0, max_value=60, value=20, step=5,
) / 100.0
gp_shape = st.sidebar.radio(
    "Glidepath Shape",
    options=["linear", "convex", "concave"],
    index=0,
    help=(
        "Linear: constant-rate transition. "
        "Convex: slow de-risking early, sharp near retirement. "
        "Concave: aggressive de-risking early, flattens later."
    ),
)

# ---------------------------------------------------------------------------
# Common simulation kwargs
# ---------------------------------------------------------------------------
_sim_kw = dict(
    initial_wealth=float(initial_wealth),
    time_horizon=time_horizon,
    cppi_multiplier=mkt["cppi_multiplier"],
    floor_pct=0.0,
    expected_return=mkt["expected_return"],
    market_volatility=mkt["market_volatility"],
    risk_free_rate=mkt["risk_free_rate"],
    n_simulations=mkt["n_simulations"],
    rebalance_freq=mkt["rebalance_freq"],
    annual_withdrawal=0.0,
    annual_contribution=float(annual_contribution),
    lambda_pct=float(cm_lambda),
    simulation_method=mkt["simulation_method"],
    block_length=mkt["block_length"],
)

sim_cm = run_simulation(**_sim_kw, strategy_type="CM")
sim_gp = run_simulation(
    **_sim_kw,
    strategy_type="Glidepath",
    glidepath_initial=gp_initial,
    glidepath_final=gp_final,
    glidepath_shape=gp_shape,
)

# ---------------------------------------------------------------------------
# Age axis for charts
# ---------------------------------------------------------------------------
n_steps = len(sim_cm["dates"])
retirement_age = start_age + time_horizon
age_axis = np.linspace(start_age, retirement_age, n_steps)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Accumulation & De-Risking Mechanics")
st.caption(
    "How different strategies manage the transition from risky to safe assets "
    f"over the accumulation horizon. Investor age {start_age} → {retirement_age}."
)

# ═══════════════════════════════════════════════════════════════════════════
# Section A; Constant Mix Allocation
# ═══════════════════════════════════════════════════════════════════════════
st.subheader("Constant Mix: Fixed Risky Allocation")

st.plotly_chart(
    build_stacked_allocation_chart(
        sim_cm, age_axis,
        title=f"Constant Mix Accumulation; Risky vs Safe Over Time ({cm_lambda:.0f}% Risky)",
    ),
    width='stretch',
)

st.caption(
    f"Constant Mix maintains a fixed **{cm_lambda:.0f}%** allocation to risky assets at every rebalance. "
    "Unlike CPPI, there is no cushion mechanism — the allocation does not respond to drawdowns. "
    "This means risky exposure is never mechanically reduced in a falling market, "
    "but also that the strategy avoids the cash-lock risk of CPPI in prolonged downturns."
)

# ═══════════════════════════════════════════════════════════════════════════
# Section B; Glidepath
# ═══════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("Glidepath: Deterministic De-Risking")

st.plotly_chart(
    build_glidepath_schedule_chart(
        initial_equity=gp_initial,
        final_equity=gp_final,
        time_horizon=time_horizon,
        shape=gp_shape,
    ),
    width='stretch',
)

st.caption(
    "The glidepath schedule is fixed at construction time. Unlike CPPI, "
    "equity exposure follows the schedule regardless of market performance "
    "— there is no cushion mechanism that reduces risk in response to "
    "drawdowns. This makes glidepath allocation fully predictable but "
    "market-indifferent."
)

# ═══════════════════════════════════════════════════════════════════════════
# Section C; Cumulative Contributions Invested
# ═══════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("Cumulative Contributions Invested Over Time")

fig_contrib = go.Figure()
for label, sim, color in [
    ("Constant Mix", sim_cm, AMUNDI_CYAN),
    ("Glidepath", sim_gp, "#E67E22"),
]:
    cc = sim.get("contribution_cumsum")
    if cc is not None:
        fig_contrib.add_trace(go.Scatter(
            x=age_axis, y=cc, mode="lines",
            name=label, line=dict(color=color, width=2),
        ))
fig_contrib.update_layout(
    title="Cumulative Capital Invested; Constant Mix vs Glidepath",
    xaxis_title="Investor Age",
    yaxis_title="Cumulative Contributions; Nominal (€)",
    template="plotly_white",
    height=380,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=60, r=30, t=60, b=40),
    font=dict(family="Arial, sans-serif", size=13, color=AMUNDI_NAVY),
)
st.plotly_chart(fig_contrib, width='stretch')

st.caption(
    "Both strategies invest the same annual contribution. "
    "The cumulative lines are identical because contributions are a deterministic input; "
    "the real divergence shows up in terminal wealth."
)

# ═══════════════════════════════════════════════════════════════════════════
# Section D; Terminal Wealth Box Plot Comparison
# ═══════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("Terminal Accumulated Wealth; Strategy Comparison")

fig_box = go.Figure()
for label, sim, color in [
    ("Constant Mix", sim_cm, AMUNDI_CYAN),
    ("Glidepath", sim_gp, "#E67E22"),
]:
    fig_box.add_trace(go.Box(
        y=sim["ending_wealth"],
        name=label,
        marker_color=color,
        line_color=AMUNDI_NAVY,
        boxpoints="outliers",
    ))

fig_box.update_layout(
    title="Terminal Wealth Distribution; Constant Mix vs Glidepath",
    yaxis_title="Terminal Wealth; Nominal (€)",
    template="plotly_white",
    height=440,
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=60, r=30, t=60, b=40),
    font=dict(family="Arial, sans-serif", size=13, color=AMUNDI_NAVY),
)
st.plotly_chart(fig_box, width='stretch')

st.caption(
    f"Comparison across {mkt['n_simulations']:,} Monte-Carlo paths. "
    "Each box shows the interquartile range (P25–P75) with whiskers at 1.5×IQR. "
    "All values are Nominal (€); no inflation adjustment."
)

build_model_caveats_panel()
