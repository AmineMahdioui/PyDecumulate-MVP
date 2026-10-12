from __future__ import annotations

import numpy as np
import pytest

from _shared import (
    run_lifecycle_simulation,
    run_sensitivity_sweep,
    run_simulation,
)


@pytest.mark.fast
def test_run_simulation_accumulation_contract(base_simulation_kwargs: dict) -> None:
    sim = run_simulation(strategy_type="CM", **base_simulation_kwargs)

    expected_keys = {
        "prob_success",
        "prob_success_terminal",
        "prob_success_income",
        "expected_shortfall",
        "median_ending",
        "median_mdd",
        "percentiles",
        "ending_wealth",
        "floor_values",
        "risky_mean",
        "summary_df",
        "dates",
        "allocation_percentiles",
        "survival_times",
        "contribution_cumsum",
        "withdrawal_percentiles",
        "lambda",
    }
    assert expected_keys.issubset(sim.keys())
    assert sim["withdrawal_percentiles"] is None
    assert sim["contribution_cumsum"] is not None
    assert sim["ending_wealth"].shape == (base_simulation_kwargs["n_simulations"],)
    assert len(sim["floor_values"]) == len(sim["dates"])
    assert 0.0 <= float(sim["prob_success"]) <= 100.0


@pytest.mark.fast
def test_run_simulation_decumulation_contract(base_simulation_kwargs: dict) -> None:
    kw = dict(base_simulation_kwargs)
    kw["annual_withdrawal"] = 24_000.0
    kw["annual_contribution"] = 0.0

    sim = run_simulation(strategy_type="Glidepath", glidepath_initial=0.6, glidepath_final=0.3, **kw)

    assert sim["withdrawal_percentiles"] is not None
    assert sim["contribution_cumsum"] is None
    assert np.asarray(sim["survival_times"]).shape == (kw["n_simulations"],)
    assert 0.0 <= float(sim["prob_success_income"]) <= 100.0


@pytest.mark.fast
def test_run_simulation_decumulation_cppi_rejected(base_simulation_kwargs: dict) -> None:
    kw = dict(base_simulation_kwargs)
    kw["annual_withdrawal"] = 10_000.0
    kw["annual_contribution"] = 0.0

    with pytest.raises(ValueError, match="unsupported"):
        run_simulation(strategy_type="CPPI", **kw)


@pytest.mark.fast
def test_run_lifecycle_simulation_pathwise_contract() -> None:
    sim_acc, sim_dec, retirement_pot = run_lifecycle_simulation(
        acc_initial_wealth=50_000.0,
        acc_time_horizon=5,
        acc_contribution=10_000.0,
        acc_floor_pct=80.0,
        acc_cppi_multiplier=3.0,
        dec_time_horizon=5,
        dec_withdrawal=20_000.0,
        expected_return=6.0,
        market_volatility=12.0,
        risk_free_rate=2.0,
        n_simulations=24,
        rebalance_freq="monthly",
        lifecycle_mode="CPPI_TO_CM",
    )

    assert isinstance(retirement_pot, float)
    assert "retirement_wealths_nominal" in sim_acc
    assert "withdrawal_percentiles" in sim_dec
    assert np.asarray(sim_acc["retirement_wealths_nominal"]).shape == (24,)
    assert np.asarray(sim_dec["ending_wealth"]).shape == (24,)


@pytest.mark.fast
def test_run_lifecycle_simulation_supports_split_rebalance_freqs() -> None:
    sim_acc, sim_dec, _ = run_lifecycle_simulation(
        acc_initial_wealth=50_000.0,
        acc_time_horizon=2,
        acc_contribution=10_000.0,
        acc_floor_pct=80.0,
        acc_cppi_multiplier=3.0,
        dec_time_horizon=2,
        dec_withdrawal=20_000.0,
        expected_return=6.0,
        market_volatility=12.0,
        risk_free_rate=2.0,
        n_simulations=20,
        rebalance_freq="monthly",
        acc_rebalance_freq="monthly",
        dec_rebalance_freq="yearly",
        lifecycle_mode="CPPI_TO_CM",
    )

    assert len(sim_acc["dates"]) == 24
    assert len(sim_dec["dates"]) == 2


@pytest.mark.fast
def test_run_sensitivity_sweep_matrix_shape_and_bounds() -> None:
    df = run_sensitivity_sweep(
        withdrawal_rates=[3.0, 5.0],
        horizons=[15, 25],
        initial_wealth=500_000.0,
        floor_pct=0.0,
        cppi_multiplier=3.0,
        lambda_pct=60.0,
        expected_return=6.0,
        market_volatility=12.0,
        risk_free_rate=2.0,
        rebalance_freq="monthly",
        strategy_type="CM",
        n_sims_sweep=30,
    )

    assert df.shape == (2, 2)
    assert set(df.columns) == {"3 %", "5 %"}
    assert set(df.index.tolist()) == {15, 25}
    assert ((df.values >= 0.0) & (df.values <= 100.0)).all()
