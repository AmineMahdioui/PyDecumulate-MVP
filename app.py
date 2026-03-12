"""
Retirement Solutions Research Prototype; Multi-Page Streamlit Dashboard
========================================================================
Entrypoint.  Run with:
    streamlit run app.py

Pages:
  1. Lifecycle Wealth Path      ; full accumulation → decumulation mountain chart
  2. Dynamic De-Risking Mechanics; CPPI cushion + glidepath schedule
  3. Retirement Sustainability  ; decumulation fan, survival, histogram
  4. Sensitivity & Model Risk   ; heatmap sweep + caveats
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Retirement Solutions Research Prototype",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Landing page
# ---------------------------------------------------------------------------
st.title("Retirement Solutions Research Prototype")
st.markdown(
    """
    A Monte Carlo simulation engine for **CPPI accumulation** and
    **decumulation lifecycle analysis**, designed to demonstrate dynamic
    de-risking mechanics, glidepath strategies, and withdrawal
    sustainability modelling.

    Use the **sidebar** to navigate between pages:

    | Page | Focus |
    |------|-------|
    | **Lifecycle Wealth Path** | Full accumulation → decumulation mountain chart with the median retirement pot as the anchor metric. |
    | **Dynamic De-Risking Mechanics** | CPPI cushion-based allocation and deterministic glidepath schedules (linear / convex / concave). |
    | **Retirement Sustainability** | Probability of Success, survival analysis, and ending-wealth distribution under withdrawals. |
    | **Sensitivity & Model Risk** | Heatmap of PoS across withdrawal rates × retirement horizons, with explicit model caveats. |

    > **Note:** This is a research prototype; not investment advice.
    > All results are model outputs under simplifying assumptions.
    """
)

# ---------------------------------------------------------------------------
# Educational expanders
# ---------------------------------------------------------------------------
with st.expander("Concepts in Plain English"):
    st.markdown(
        """
        | Concept | What it means |
        |---------|---------------|
        | **Probability of Success (PoS)** | Out of all simulated futures, the percentage where your money lasts the full retirement horizon. Higher is better; 100 % means every scenario survived. |
        | **Expected Shortfall** | When things go wrong, *how wrong* do they go? This is the average size of the gap between what you needed and what the portfolio could actually pay, measured only across the failing scenarios. |
        | **Survival Time** | How many years the portfolio stays funded before it runs out. The *average* tells you the typical outcome; the *10th percentile* tells you the near-worst case. |
        | **Floor** | A safety net built into the CPPI strategy. It represents the minimum portfolio value the model tries to protect; expressed as a percentage of your invested capital (accumulation) or of the present value of remaining withdrawals (decumulation). It is a *modelled* cushion, not a contractual guarantee. |
        | **Glidepath** | A pre-set schedule that gradually shifts the portfolio from equities to bonds as retirement approaches. Unlike CPPI, it follows the calendar; not market conditions. |
        | **CPPI (Constant Proportion Portfolio Insurance)** | A dynamic strategy that sizes equity exposure based on the *cushion*; the gap between your current wealth and the floor. When markets fall the cushion shrinks, so equity is automatically reduced. When markets rise the cushion expands and equity increases. |
        """
    )

with st.expander("Advanced Parameters & Model Status"):
    st.markdown(
        """
        **Return model**; Two options are available in the sidebar:
        - *GBM (Parametric)*: log-normal Geometric Brownian Motion using the expected return (μ) and volatility (σ) sliders.
        - *Historical Bootstrap*: block-resamples real monthly equity returns. The block length controls how much short-term autocorrelation is preserved (1 = i.i.d.).

        **Bootstrap settings**; Block length (1–24 months), number of Monte-Carlo paths (100–5 000).

        **Rebalance frequency**; How often the strategy re-targets its allocation: daily / weekly / monthly / quarterly / yearly. More frequent rebalancing tracks the target more closely but implies higher turnover.

        **Market assumptions**; Expected return, volatility, and risk-free rate are constant over the full horizon (no regime switches or mean-reversion). These are *inputs*, not forecasts.

        **Inflation module status:** 🔴 **Not active in displayed metrics.**
        All values shown in the dashboard are **Nominal (€)**.
        An inflation-adjusted display mode is planned but not yet implemented; no real-vs-nominal toggle exists today.
        """
    )

st.info("Select a page from the sidebar to get started.")
