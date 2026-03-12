"""Monte Carlo analysis of CPPI simulation results."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .accumulation import AccumulationResult
from .decumulation import DecumulationResult
from .parameters import StrategyParameters


class MonteCarloAnalyzer:
    """Compute aggregate statistics over Monte-Carlo CPPI paths.

    Parameters
    ----------
    result : AccumulationResult|DecumulationResult
        Raw simulation output.
    params : StrategyParameters
        Strategy configuration (used for reference values).
    """

    def __init__(
        self, result: AccumulationResult|DecumulationResult, params: StrategyParameters
    ) -> None:
        self._result = result
        self._params = params

    # -- scalar KPIs ---------------------------------------------------------

    def probability_of_success(self) -> float:
        """Percentage of paths whose final value exceeds the floor."""
        final_vals = self._result.portfolio_values[-1, :]
        final_floor = self._result.floor_values[-1]
        return float(np.mean(final_vals > final_floor) * 100.0)

    def expected_shortfall(self) -> float:
        """Average magnitude of failure across paths that experienced shortfall.

        For **decumulation** paths the shortfall is measured as the total
        cumulative missed withdrawals (in real terms) across every step where
        the portfolio could not pay the full intended withdrawal amount.

        For **accumulation** paths the original floor-based calculation is
        used: mean of ``(floor - wealth)`` for paths whose final wealth is
        below the final floor.

        Returns
        -------
        float
            Expected shortfall in absolute terms (same currency unit as wealth).
            Returns ``0.0`` if no path experiences any shortfall.
        """
        # Decumulation: floor_values[-1] == 0 by construction (no remaining
        # withdrawals at end of horizon), so a final-floor comparison always
        # yields zero. Use cumulative missed withdrawals instead.
        if hasattr(self._result, "withdrawals_made"):
            # intended real withdrawal per step = W / (1 + g)
            # because nominal intended at step t  = W*(1+g)^t
            # inflation deflator applied in the engine = (1+g)^(t+1)
            # → real intended = W*(1+g)^t / (1+g)^(t+1) = W / (1+g)
            step_g = self._params.step_inflation
            intended_real = self._params.step_withdrawal / (1.0 + step_g)
            shortfall_per_step = np.maximum(
                intended_real - self._result.withdrawals_made, 0.0
            )  # shape (t_steps, n_sims)
            total_shortfall = shortfall_per_step.sum(axis=0)  # shape (n_sims,)
            failing_mask = total_shortfall > 0
            if not np.any(failing_mask):
                return 0.0
            return float(np.mean(total_shortfall[failing_mask]))

        # Accumulation fallback: compare terminal wealth to terminal floor
        final_vals = self._result.portfolio_values[-1, :]
        final_floor = self._result.floor_values[-1]
        shortfalls = final_floor - final_vals
        failed_mask = shortfalls > 0
        if not np.any(failed_mask):
            return 0.0
        return float(np.mean(shortfalls[failed_mask]))

    def median_ending_wealth(self) -> float:
        """Median terminal portfolio value across all paths."""
        return float(
            np.median(self._result.portfolio_values[-1, :])
        )

    def max_drawdown_per_path(self) -> np.ndarray:
        """Maximum drawdown (%) for each simulation path.

        Returns
        -------
        np.ndarray, shape ``(n_simulations,)``
            Peak-to-trough drawdown percentage for every path.
        """
        vals = self._result.portfolio_values
        running_max = np.maximum.accumulate(vals, axis=0)
        drawdowns = np.where(
            running_max > 0,
            (running_max - vals) / running_max,
            0.0,
        )
        return np.max(drawdowns, axis=0) * 100.0

    def median_max_drawdown(self) -> float:
        """Median of the per-path maximum drawdowns (%)."""
        return float(np.median(self.max_drawdown_per_path()))

    # -- path-level outputs --------------------------------------------------

    def percentile_paths(
        self, percentiles: list[float] | None = None
    ) -> dict[str, np.ndarray]:
        """Compute percentile envelopes of the portfolio value paths.

        Returns a dict mapping labels like ``"P5"`` to arrays of shape
        ``(months,)``.
        """
        if percentiles is None:
            percentiles = [5, 25, 50, 75, 95]
        return {
            f"P{p}": np.percentile(
                self._result.portfolio_values, p, axis=1
            )
            for p in percentiles
        }

    def ending_wealth_array(self) -> np.ndarray:
        """Terminal portfolio value for every simulation."""
        return self._result.portfolio_values[-1, :]

    def summary_dataframe(
        self, dates: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Build a tidy DataFrame with median paths and the floor."""
        pcts = self.percentile_paths()
        risky_median = np.median(self._result.risky_paths, axis=1)
        return pd.DataFrame(
            {
                "Date": dates,
                "Risky Asset (Median)": np.round(risky_median, 2),
                "Portfolio (P5)": np.round(pcts["P5"], 2),
                "Portfolio (P25)": np.round(pcts["P25"], 2),
                "Portfolio (Median)": np.round(pcts["P50"], 2),
                "Portfolio (P75)": np.round(pcts["P75"], 2),
                "Portfolio (P95)": np.round(pcts["P95"], 2),
                "Guaranteed Floor": np.round(
                    self._result.floor_values, 2
                ),
            }
        )

    # -- new analytics -------------------------------------------------------

    def allocation_percentile_paths(
        self, percentiles: list[float] | None = None
    ) -> dict[str, np.ndarray]:
        """Percentile envelopes of the risky-asset allocation (%) over time.

        Returns a dict mapping ``"P5"`` … ``"P95"`` to arrays of shape
        ``(t_steps,)`` with values in ``[0, 100]``.
        """
        if percentiles is None:
            percentiles = [5, 25, 50, 75, 95]
        alloc_pct = self._result.risky_allocation * 100.0
        return {
            f"P{p}": np.percentile(alloc_pct, p, axis=1)
            for p in percentiles
        }

    def survival_time_per_path(self) -> np.ndarray:
        """Years until each simulation path first breaches the floor.

        Paths that survive the full horizon are right-censored at
        ``time_horizon + 1``.

        Returns
        -------
        np.ndarray, shape ``(n_simulations,)``
        """
        pv = self._result.portfolio_values   # (t_steps, n_sims)
        fv = self._result.floor_values       # (t_steps,)
        dt = self._params.dt
        at_or_below = pv <= fv[:, np.newaxis]   # (t_steps, n_sims)
        first_breach = np.argmax(at_or_below, axis=0)
        never_breached = ~at_or_below.any(axis=0)
        return np.where(
            never_breached,
            float(self._params.time_horizon) + 1.0,
            first_breach.astype(float) * dt,
        )

    def survival_curve(self) -> tuple[np.ndarray, np.ndarray]:
        """Step-by-step fraction of paths where the portfolio is still funded.

        At each simulation step *t*, computes the proportion of paths with
        strictly positive wealth: ``fraction_alive[t] = mean(W_t > 0)``.
        This is the proper survivor function; a monotonically non-increasing
        curve from ~1.0 down to PoS / 100 at the final step.

        Returns
        -------
        time_years : np.ndarray, shape ``(t_steps,)``
            Year corresponding to each step.
        fraction_alive : np.ndarray, shape ``(t_steps,)``
            Fraction of paths still alive (0–1) at each step.
        """
        pv = self._result.portfolio_values   # (t_steps, n_sims)
        dt = self._params.dt
        t_steps = pv.shape[0]
        time_years = np.arange(1, t_steps + 1) * dt
        fraction_alive = (pv > 0).mean(axis=1)
        return time_years, fraction_alive

    def withdrawal_percentile_paths(
        self, percentiles: list[float] | None = None
    ) -> dict[str, np.ndarray] | None:
        """Percentile envelopes of actual withdrawals per step.

        Returns ``None`` for accumulation results (no ``withdrawals_made``).
        """
        if not hasattr(self._result, "withdrawals_made"):
            return None
        if percentiles is None:
            percentiles = [5, 25, 50, 75, 95]
        return {
            f"P{p}": np.percentile(self._result.withdrawals_made, p, axis=1)
            for p in percentiles
        }