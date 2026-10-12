from __future__ import annotations

import numpy as np
import pytest

from simulator.accumulation import AccumulationResult
from simulator.decumulation import DecumulationResult
from simulator.monte_carlo import MonteCarloAnalyzer
from simulator.parameters import StrategyParameters


def _base_params(*, annual_withdrawal: float = 0.0, annual_contribution: float = 0.0) -> StrategyParameters:
    return StrategyParameters(
        initial_wealth=100_000.0,
        time_horizon=2,
        expected_return=5.0,
        market_volatility=10.0,
        risk_free_rate=1.0,
        n_simulations=4,
        rebalance_freq="yearly",
        annual_withdrawal=annual_withdrawal,
        annual_contribution=annual_contribution,
        annual_inflation_rate=0.0,
        cppi_multiplier=3.0,
        floor_pct=80.0,
        Lambda=60.0,
    )


@pytest.mark.fast
def test_income_probability_of_success_accumulation_falls_back_to_terminal_floor() -> None:
    params = _base_params(annual_contribution=10_000.0)
    result = AccumulationResult(
        portfolio_values=np.array([[100.0, 100.0], [90.0, 110.0]]),
        floor_values=np.array([80.0, 95.0]),
        risky_paths=np.array([[100.0, 100.0], [95.0, 105.0]]),
        risky_allocation=np.array([[0.6, 0.6], [0.6, 0.6]]),
        contributions_made=np.zeros((2, 2)),
        portfolio_values_nominal=None,
        floor_values_nominal=None,
        contributions_made_nominal=None,
    )

    analyzer = MonteCarloAnalyzer(result, params)
    assert analyzer.terminal_floor_probability_of_success() == 50.0
    assert analyzer.income_probability_of_success() == 50.0


@pytest.mark.fast
def test_income_probability_of_success_decumulation_tracks_shortfalls() -> None:
    params = _base_params(annual_withdrawal=100.0)
    result = DecumulationResult(
        portfolio_values=np.array([[100.0, 100.0], [80.0, 120.0]]),
        floor_values=np.array([0.0, 0.0]),
        risky_paths=np.array([[100.0, 100.0], [90.0, 110.0]]),
        risky_allocation=np.array([[0.5, 0.5], [0.5, 0.5]]),
        withdrawals_made=np.array([[100.0, 100.0], [50.0, 100.0]]),
        portfolio_values_nominal=None,
        floor_values_nominal=None,
        withdrawals_made_nominal=None,
    )

    analyzer = MonteCarloAnalyzer(result, params)
    assert analyzer.income_probability_of_success() == 50.0


@pytest.mark.fast
def test_expected_shortfall_decumulation_uses_cumulative_missed_withdrawals() -> None:
    params = _base_params(annual_withdrawal=100.0)
    result = DecumulationResult(
        portfolio_values=np.array([[100.0, 100.0], [80.0, 120.0]]),
        floor_values=np.array([0.0, 0.0]),
        risky_paths=np.array([[100.0, 100.0], [90.0, 110.0]]),
        risky_allocation=np.array([[0.5, 0.5], [0.5, 0.5]]),
        withdrawals_made=np.array([[100.0, 80.0], [100.0, 90.0]]),
        portfolio_values_nominal=None,
        floor_values_nominal=None,
        withdrawals_made_nominal=None,
    )

    analyzer = MonteCarloAnalyzer(result, params)
    # Shortfall on path 2 is 20 + 10 = 30, only one failing path.
    assert analyzer.expected_shortfall() == 30.0


@pytest.mark.fast
def test_max_drawdown_per_path_is_bounded_percent() -> None:
    params = _base_params(annual_contribution=10_000.0)
    result = AccumulationResult(
        portfolio_values=np.array([
            [100.0, 100.0],
            [90.0, 110.0],
            [95.0, 120.0],
        ]),
        floor_values=np.array([80.0, 80.0, 80.0]),
        risky_paths=np.array([
            [100.0, 100.0],
            [90.0, 110.0],
            [95.0, 120.0],
        ]),
        risky_allocation=np.array([
            [0.6, 0.6],
            [0.6, 0.6],
            [0.6, 0.6],
        ]),
        contributions_made=np.zeros((3, 2)),
        portfolio_values_nominal=None,
        floor_values_nominal=None,
        contributions_made_nominal=None,
    )

    analyzer = MonteCarloAnalyzer(result, params)
    mdd = analyzer.max_drawdown_per_path()
    assert mdd.shape == (2,)
    assert np.all((mdd >= 0.0) & (mdd <= 100.0))


@pytest.mark.fast
def test_percentile_paths_and_allocation_shapes() -> None:
    params = _base_params(annual_contribution=10_000.0)
    result = AccumulationResult(
        portfolio_values=np.array([
            [80.0, 100.0, 120.0],
            [85.0, 105.0, 130.0],
        ]),
        floor_values=np.array([70.0, 70.0]),
        risky_paths=np.array([
            [80.0, 100.0, 120.0],
            [84.0, 103.0, 125.0],
        ]),
        risky_allocation=np.array([
            [0.2, 0.5, 0.8],
            [0.2, 0.5, 0.8],
        ]),
        contributions_made=np.zeros((2, 3)),
        portfolio_values_nominal=None,
        floor_values_nominal=None,
        contributions_made_nominal=None,
    )

    analyzer = MonteCarloAnalyzer(result, params)
    p = analyzer.percentile_paths([5, 50, 95])
    a = analyzer.allocation_percentile_paths([5, 50, 95])

    assert set(p.keys()) == {"P5", "P50", "P95"}
    assert set(a.keys()) == {"P5", "P50", "P95"}
    assert p["P50"].shape == (2,)
    assert a["P50"].shape == (2,)
    assert np.all((a["P50"] >= 0.0) & (a["P50"] <= 100.0))


@pytest.mark.fast
@pytest.mark.property
def test_survival_curve_values_are_bounded_and_time_aligned() -> None:
    params = _base_params(annual_contribution=10_000.0)
    result = AccumulationResult(
        portfolio_values=np.array([
            [100.0, 0.0, 50.0],
            [90.0, 0.0, 40.0],
            [80.0, 0.0, 30.0],
        ]),
        floor_values=np.array([0.0, 0.0, 0.0]),
        risky_paths=np.array([
            [100.0, 0.0, 50.0],
            [90.0, 0.0, 40.0],
            [80.0, 0.0, 30.0],
        ]),
        risky_allocation=np.array([
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ]),
        contributions_made=np.zeros((3, 3)),
        portfolio_values_nominal=None,
        floor_values_nominal=None,
        contributions_made_nominal=None,
    )

    analyzer = MonteCarloAnalyzer(result, params)
    t, alive = analyzer.survival_curve()

    assert t.shape == alive.shape == (3,)
    assert np.all(np.diff(t) > 0)
    assert np.all((alive >= 0.0) & (alive <= 1.0))
