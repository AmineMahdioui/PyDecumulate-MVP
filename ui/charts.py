"""Shared chart builders and visual constants."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from simulator import AccLinearGlidepath

# ---------------------------------------------------------------------------
# Amundi-inspired colour palette
# ---------------------------------------------------------------------------
AMUNDI_CYAN = "#009FE3"
AMUNDI_NAVY = "#001C4B"
AMUNDI_WHITE = "#FFFFFF"
AMUNDI_LIGHT_BLUE = "#C7D9E3"
AMUNDI_GREY = "#58595B"
AMUNDI_GREEN = "#007A33"


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


def build_mountain_chart_age(
    sim_acc: dict,
    sim_dec: dict,
    age_axis_acc: np.ndarray,
    age_axis_dec: np.ndarray,
    include_contributions: bool = False,
    show_inner_bands: bool = True,
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
    if show_inner_bands:
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
    if show_inner_bands:
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


# Historical note: this function is intentionally shadowed later by a second
# definition with the same name to preserve existing runtime behavior.
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


def build_stacked_allocation_chart(
    sim_data: dict,
    dates,
    title: str = "Portfolio Composition — Risky vs Safe Assets",
    height: int = 420,
) -> go.Figure:
    """Stacked area showing how CPPI dynamically splits the portfolio.

    The risky band (blue) shrinks toward the floor when markets are weak;
    the safe / money-market band (grey) expands to absorb the cushion.
    Uses the median allocation path.
    """
    alloc = sim_data["allocation_percentiles"]
    risky_med = alloc["P50"]
    safe_med = 100.0 - risky_med

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
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=60, b=40),
        font=dict(family="Arial, sans-serif", size=13, color=AMUNDI_NAVY),
    )
    return fig


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
    compared to a static withdrawal rate. Paths that survive the full
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


def build_glidepath_schedule_chart(
    initial_equity: float,
    final_equity: float,
    time_horizon: int,
    shape: str = "linear",
    height: int = 400,
) -> go.Figure:
    """Deterministic allocation schedule for a glidepath (no simulation needed).

    Parameters are in *fraction* space (0–1). The chart displays percentages.
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
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=60, b=40),
        font=dict(family="Arial, sans-serif", size=13, color=AMUNDI_NAVY),
    )
    return fig


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
