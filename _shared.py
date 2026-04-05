"""Shared simulation wrappers used across pages."""

import numpy as np
import pandas as pd
import streamlit as st

from simulator import (
    AccCPPIEngine,
    AccConstantMixEngine,
    AccGlidepathEngine,
    DecConstantMixEngine,
    DecGlidepathEngine,
    HistoricalBootstrapSimulator,
    LifecycleParameters,
    LifecycleResult,
    LifecycleSimulator,
    MarketSimulator,
    MonteCarloAnalyzer,
    StrategyParameters,
)

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
    strategy_type: str = "CPPI",
    glidepath_initial: float = 0.80,
    glidepath_final: float = 0.20,
    glidepath_shape: str = "linear",
) -> dict:
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
    result = engine.run(sim_returns["equity"], include_nominal_arrays=False)

    analyzer = MonteCarloAnalyzer(result, params)

    dates = pd.date_range(
        start="2025-01-01",
        periods=params.n_steps,
        freq=FREQ_TO_PD_OFFSET[params.rebalance_freq],
    )

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
        "allocation_percentiles": analyzer.allocation_percentile_paths(),
        "survival_times": analyzer.survival_time_per_path(),
        "contribution_cumsum": contribution_cumsum,
        "withdrawal_percentiles": analyzer.withdrawal_percentile_paths(),
        "lambda": params.Lambda / 100.0,
    }


# ---------------------------------------------------------------------------
# Lifecycle simulation runner (pathwise handoff: acc terminal state → dec)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Running lifecycle simulation …")
def run_lifecycle_simulation(
    acc_initial_wealth: float,
    acc_time_horizon: int,
    acc_contribution: float,
    acc_floor_pct: float,
    acc_cppi_multiplier: float,
    dec_time_horizon: int,
    dec_withdrawal: float,
    dec_floor_pct: float = 0.0,
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
    """Run a full lifecycle simulation with pathwise retirement handoff."""
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
    """Compute a PoS matrix over withdrawal rates × retirement horizons."""
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
