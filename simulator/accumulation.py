from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from numba import njit, prange
from dataclasses import dataclass
from .parameters import StrategyParameters
from .inflation import deflate

@dataclass
class AccumulationResult:
    """Output of an Accumulation simulation run (CPPI or Constant Mix)."""
    # Real (deflated) outputs; used for dashboard charts and KPIs
    portfolio_values: np.ndarray   # shape (t_steps, n_sims); portfolio value (real)
    floor_values: np.ndarray       # shape (t_steps,); guaranteed floor (zeros for CM, real)
    risky_paths: np.ndarray        # shape (t_steps, n_sims); buy-and-hold risky benchmark (real)
    risky_allocation: np.ndarray   # shape (t_steps, n_sims); fraction allocated to risky asset
    contributions_made: np.ndarray # shape (t_steps, n_sims); actual contribution per step (real)
    # turnover: np.ndarray           # shape (t_steps, n_sims); absolute value of assets traded (real)
    # Nominal (pre-deflation) outputs; used for lifecycle handoff
    portfolio_values_nominal: np.ndarray | None   # shape (t_steps, n_sims); nominal portfolio value
    floor_values_nominal: np.ndarray | None       # shape (t_steps,); nominal guaranteed floor
    contributions_made_nominal: np.ndarray | None # shape (t_steps, n_sims); nominal contributions


class BaseAccumulationEngine(ABC):
    """Abstract base class for accumulation strategies.
    
    Consolidates common initialization, array management, deflation, and result
    packaging logic. Child classes override _calculate_floor(), _initialize_state(),
    and _call_core_function() to implement specific strategies (CPPI, Glidepath, CM).
    """

    def __init__(self, params: StrategyParameters) -> None:
        self._params = params

    @abstractmethod
    def _calculate_floor(self, t_steps: int, exprdt: np.ndarray) -> np.ndarray:
        """Calculate deterministic floor for the strategy.
        
        Returns array of shape (t_steps + 1,). CPPI overrides with cushion-based
        floor; others return zeros.
        """
        pass

    @abstractmethod
    def _initialize_state(
        self,
        t_steps: int,
        n_sims: int,
        Pt_f: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Initialize state arrays Mt_f, Xt_f, Vt_f at t=0.
        
        Returns:
            (Mt_f, Xt_f, Vt_f) all shape (t_steps+1, n_sims) for Mt/Xt/Vt,
            with initial conditions set at [0, :].
        """
        pass

    @abstractmethod
    def _call_core_function(
        self,
        t_steps: int,
        n_sims: int,
        exprdt: np.ndarray,
        gross_returns: np.ndarray,
        Pt_f: np.ndarray,
        Mt_f: np.ndarray,
        Xt_f: np.ndarray,
        Vt_f: np.ndarray,
        contributions_arr: np.ndarray,
    ) -> None:
        """Execute the Numba core simulation loop in-place.
        
        Modifies Mt_f, Xt_f, Vt_f, contributions_arr arrays in-place.
        """
        pass

    def _setup_market_arrays(
        self,
        asset_log_returns: np.ndarray,
        riskless_returns: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepare exprdt and gross_returns from input log-returns.
        
        Returns:
            (exprdt, gross_returns) both shape (t_steps, n_sims).
        """
        t_steps, n_sims = asset_log_returns.shape
        
        if riskless_returns is not None:
            if riskless_returns.shape != asset_log_returns.shape:
                raise ValueError(
                    "riskless_returns must be either None or have the same shape "
                    "as asset_log_returns."
                )
            exprdt = np.exp(riskless_returns)
        else:
            exprdt = np.exp(self._params.step_rf).repeat(
                t_steps * n_sims
            ).reshape(t_steps, n_sims)
        
        gross_returns = np.exp(asset_log_returns)
        return exprdt, gross_returns

    def run(
        self,
        asset_log_returns: np.ndarray,
        riskless_returns: np.ndarray | None = None,
        include_nominal_arrays: bool = True,
    ) -> AccumulationResult:
        """Execute the accumulation strategy simulation.
        
        Parameters
        ----------
        asset_log_returns : np.ndarray
            Log-returns of risky asset, shape (t_steps, n_sims).
        riskless_returns : np.ndarray | None
            Optional log-returns of risk-free asset; if None, uses params.step_rf.
        include_nominal_arrays : bool
            If True, retain nominal (pre-deflation) arrays for lifecycle integration.
            If False, set to None to save memory.
        
        Returns
        -------
        AccumulationResult
            Deflated outputs ready for dashboard visualization and KPIs.
        """
        t_steps, n_sims = asset_log_returns.shape

        # 1. Setup market arrays
        exprdt, gross_returns = self._setup_market_arrays(asset_log_returns, riskless_returns)

        # 2. Calculate floor (strategy-specific)
        Pt_f = self._calculate_floor(t_steps, exprdt)

        # 3. Initialize state arrays
        Mt_f, Xt_f, Vt_f = self._initialize_state(t_steps, n_sims, Pt_f)

        # 4. Initialize contribution tracking
        contributions_arr = np.zeros((t_steps, n_sims))
        # Optional turnover tracking (kept commented-out for now to save memory)
        # turnover_arr = np.zeros((t_steps, n_sims))

        # 5. Call strategy-specific Numba core function
        self._call_core_function(
            t_steps, n_sims, exprdt, gross_returns, Pt_f,
            Mt_f, Xt_f, Vt_f, contributions_arr,
        )

        # 6. Calculate fractional allocation
        risky_allocation = Xt_f / np.maximum(Vt_f, 1e-8)
        risky_asset_paths = self._params.initial_wealth * np.cumprod(gross_returns, axis=0)

        # 7. Retain nominal arrays before deflation (for lifecycle handoff)
        portfolio_values_nominal = Vt_f[1:].copy() if include_nominal_arrays else None
        floor_values_nominal = Pt_f[1:].copy() if include_nominal_arrays else None
        contributions_made_nominal = contributions_arr.copy() if include_nominal_arrays else None
        # turnover_nominal = turnover_arr.copy() if include_nominal_arrays else None

        # 8. Deflate outputs to real terms (today's purchasing power)
        _ir = self._params.annual_inflation_rate
        _dt = self._params.dt
        portfolio_values_real = deflate(Vt_f[1:], _ir, _dt, start_step=1)
        floor_values_real = deflate(Pt_f[1:], _ir, _dt, start_step=1)
        risky_paths_real = deflate(risky_asset_paths, _ir, _dt, start_step=1)
        contributions_made_real = deflate(contributions_arr, _ir, _dt, start_step=1)
        # turnover_real = deflate(turnover_arr, _ir, _dt, start_step=1)

        return AccumulationResult(
            portfolio_values=portfolio_values_real,
            floor_values=floor_values_real,
            risky_paths=risky_paths_real,
            risky_allocation=risky_allocation[1:],
            contributions_made=contributions_made_real,
            # turnover=turnover_real,
            portfolio_values_nominal=portfolio_values_nominal,
            floor_values_nominal=floor_values_nominal,
            contributions_made_nominal=contributions_made_nominal,
            # turnover_nominal=turnover_nominal,
        )



@njit(parallel=True)
def _run_cppi_accumulation_core(
    t_steps,
    n_sims, m,
    exprdt,
    gross_returns,
    Pt_f, Mt_f, Xt_f, Vt_f, 
    step_contribution, step_inflation,
    contributions_arr, 
    # turnover_arr,
    ):
    """
    Core Numba loop for a CPPI Accumulation strategy.
    Contributions grow with inflation each step.
    """
    for i in prange(n_sims):
        for t in range(t_steps):
            # Calculate cumulative inflation factor for this step
            current_inflation_factor = (1.0 + step_inflation) ** t
            current_nominal_contribution = step_contribution * current_inflation_factor
            
            # 1. Market Move Phase
            Mt_grown = Mt_f[t, i] * exprdt[t, i]
            Xt_grown = Xt_f[t, i] * gross_returns[t, i]
            Vt_grown = Mt_grown + Xt_grown

            # 2. Contribution Phase (inflation-adjusted)
            Vt_f[t+1, i] = Vt_grown + current_nominal_contribution
            contributions_arr[t, i] = current_nominal_contribution

            # 3. Rebalancing Phase
            cushion = max(Vt_f[t+1, i] - Pt_f[t+1], 0.0)
            Xt_target = min(m * cushion, Vt_f[t+1, i])  # Apply no-leverage constraint
            Mt_target = Vt_f[t+1, i] - Xt_target
            
            # 4. Turnover Tracking
            # trade_amount = Xt_target - Xt_grown
            # turnover_arr[t, i] = abs(trade_amount)

            # 5. Lock in the new state
            Xt_f[t+1, i] = Xt_target
            Mt_f[t+1, i] = Mt_target
            


@njit(parallel=True)
def _run_cm_accumulation_core(t_steps, n_sims, alpha, exprdt, gross_returns, Mt_f, Xt_f, Vt_f, step_contribution, step_inflation, contributions_arr, 
                            #   turnover_arr
                              ):
    """
    Core Numba loop for a Constant Mix Accumulation strategy.
    Contributions grow with inflation each step.
    """
    for i in prange(n_sims):
        for t in range(t_steps):
            # Calculate cumulative inflation factor for this step
            current_inflation_factor = (1.0 + step_inflation) ** t
            current_nominal_contribution = step_contribution * current_inflation_factor
            
            # 1. Market Growth (Before contribution)
            Mt_grown = Mt_f[t, i] * exprdt[t, i]
            Xt_grown = Xt_f[t, i] * gross_returns[t, i]
            Vt_grown = Mt_grown + Xt_grown
            
            # 2. Contribution Phase (inflation-adjusted)
            Vt_f[t+1, i] = Vt_grown + current_nominal_contribution
            contributions_arr[t, i] = current_nominal_contribution
            
            # 3. Rebalancing Phase
            Xt_target = alpha * Vt_f[t+1, i]
            Mt_target = Vt_f[t+1, i] - Xt_target
            
            # 4. Turnover Tracking
            # trade_amount = Xt_target - Xt_grown
            # turnover_arr[t, i] = abs(trade_amount)
            
            # 5. Lock in the new state
            Xt_f[t+1, i] = Xt_target
            Mt_f[t+1, i] = Mt_target


class AccCPPIEngine(BaseAccumulationEngine):
    """Run the CPPI strategy over a matrix of market returns."""

    def _calculate_floor(self, t_steps: int, exprdt: np.ndarray) -> np.ndarray:
        """Calculate the deterministic floor for CPPI (exponential growth at risk-free rate)."""
        Pt_f = np.zeros(t_steps + 1)
        Pt_f[0] = (self._params.initial_wealth + self._params.step_contribution) * self._params.floor_pct / 100.0
        t_array = np.arange(t_steps + 1)
        Pt_f = Pt_f[0] * np.exp(self._params.step_rf * t_array)
        return Pt_f

    def _initialize_state(
        self,
        t_steps: int,
        n_sims: int,
        Pt_f: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Initialize CPPI state with cushion-based risky allocation."""
        Mt_f = np.zeros((t_steps + 1, n_sims))
        Xt_f = np.zeros((t_steps + 1, n_sims))
        Vt_f = np.zeros((t_steps + 1, n_sims))

        # Set initial conditions at t=0
        m = self._params.cppi_multiplier
        Vt_f[0, :] = self._params.initial_wealth
        cushion_0 = np.maximum(Vt_f[0, :] - Pt_f[0], 0.0)
        
        initial_target_risky = m * cushion_0
        Xt_f[0, :] = np.minimum(initial_target_risky, Vt_f[0, :])
        Mt_f[0, :] = Vt_f[0, :] - Xt_f[0, :]

        return Mt_f, Xt_f, Vt_f

    def _call_core_function(
        self,
        t_steps: int,
        n_sims: int,
        exprdt: np.ndarray,
        gross_returns: np.ndarray,
        Pt_f: np.ndarray,
        Mt_f: np.ndarray,
        Xt_f: np.ndarray,
        Vt_f: np.ndarray,
        contributions_arr: np.ndarray,
    ) -> None:
        """Execute the CPPI core loop."""
        m = self._params.cppi_multiplier
        C = self._params.step_contribution
        step_inflation = self._params.step_inflation
        
        _run_cppi_accumulation_core(
            t_steps, n_sims, m, exprdt, gross_returns, Pt_f,
            Mt_f, Xt_f, Vt_f, C, step_inflation, contributions_arr,
        )



class AccLinearGlidepath:
    """
    A glidepath strategy that adjusts asset allocation over time.

    Supports three shapes:
      - ``"linear"``:  constant-rate transition
      - ``"convex"``:  slow de-risking early, sharp drop near retirement
      - ``"concave"``: aggressive de-risking early, flattens later

    The allocation to risky assets (e.g., stocks) transitions from an initial
    value to a final value over the investment horizon.
    """

    VALID_SHAPES = ("linear", "convex", "concave")

    def __init__(
        self,
        initial_equity_allocation: float,
        final_equity_allocation: float,
        years: int,
        shape: str = "linear",
    ):
        """
        Initialize the glidepath.

        Args:
            initial_equity_allocation: Starting allocation to equities (0.0 to 1.0)
            final_equity_allocation: Ending allocation to equities (0.0 to 1.0)
            years: Number of years over which the glidepath operates
            shape: One of ``"linear"``, ``"convex"``, ``"concave"``
        """
        if not 0 <= initial_equity_allocation <= 1:
            raise ValueError("initial_equity_allocation must be between 0 and 1")
        if not 0 <= final_equity_allocation <= 1:
            raise ValueError("final_equity_allocation must be between 0 and 1")
        if years <= 0:
            raise ValueError("years must be positive")
        if shape not in self.VALID_SHAPES:
            raise ValueError(f"shape must be one of {self.VALID_SHAPES}, got '{shape}'")

        self.initial_equity_allocation = initial_equity_allocation
        self.final_equity_allocation = final_equity_allocation
        self.years = years
        self.shape = shape

    # -----------------------------------------------------------------
    # Core interpolation; works on normalised tau in [0, 1]
    # -----------------------------------------------------------------
    def _interpolate(self, tau: float) -> float:
        """Map normalised time *tau* ∈ [0, 1] to equity allocation."""
        a0 = self.initial_equity_allocation
        a1 = self.final_equity_allocation
        if self.shape == "linear":
            return a0 + (a1 - a0) * tau
        elif self.shape == "convex":
            return a0 + (a1 - a0) * tau ** 2
        else:  # concave
            return a0 + (a1 - a0) * (1.0 - (1.0 - tau) ** 2)

    def get_equity_allocation(self, year: int) -> float:
        """
        Get the equity allocation for a given year.

        Args:
            year: The current year (0-indexed)

        Returns:
            The equity allocation as a float between 0 and 1
        """
        if year < 0:
            return self.initial_equity_allocation
        if year >= self.years:
            return self.final_equity_allocation
        tau = year / self.years
        return self._interpolate(tau)

    def get_bond_allocation(self, year: int) -> float:
        """
        Get the bond allocation for a given year.

        Args:
            year: The current year (0-indexed)

        Returns:
            The bond allocation as a float between 0 and 1
        """
        return 1.0 - self.get_equity_allocation(year)

    def get_allocations(self, year: int) -> dict:
        """
        Get both equity and bond allocations for a given year.

        Args:
            year: The current year (0-indexed)

        Returns:
            Dictionary with 'equity' and 'bond' allocations
        """
        equity = self.get_equity_allocation(year)
        return {
            'equity': equity,
            'bond': 1.0 - equity
        }
        


@njit(parallel=True)
def _run_glidepath_accumulation_core(t_steps, n_sims, alpha_schedule, exprdt, gross_returns, Mt_f, Xt_f, Vt_f, step_contribution, step_inflation, contributions_arr, 
                                    #  turnover_arr
                                     ):
    """
    Core Numba loop for a Glidepath Accumulation strategy.
    alpha_schedule is a 1-D array of length t_steps with the equity fraction at each step.
    Contributions grow with inflation each step.
    """
    for i in prange(n_sims):
        for t in range(t_steps):
            alpha = alpha_schedule[t]
            # Calculate cumulative inflation factor for this step
            current_inflation_factor = (1.0 + step_inflation) ** t
            current_nominal_contribution = step_contribution * current_inflation_factor

            # 1. Market Growth (Before contribution)
            Mt_grown = Mt_f[t, i] * exprdt[t, i]
            Xt_grown = Xt_f[t, i] * gross_returns[t, i]
            Vt_grown = Mt_grown + Xt_grown

            # 2. Contribution Phase (inflation-adjusted)
            Vt_f[t+1, i] = Vt_grown + current_nominal_contribution
            contributions_arr[t, i] = current_nominal_contribution

            # 3. Rebalancing Phase (time-varying alpha)
            Xt_target = alpha * Vt_f[t+1, i]
            Mt_target = Vt_f[t+1, i] - Xt_target

            # 4. Turnover Tracking
            # trade_amount = Xt_target - Xt_grown
            # turnover_arr[t, i] = abs(trade_amount)

            # 5. Lock in the new state
            Xt_f[t+1, i] = Xt_target
            Mt_f[t+1, i] = Mt_target


class AccGlidepathEngine(BaseAccumulationEngine):
    """Run a glidepath strategy over a matrix of market returns.

    Unlike Constant Mix (fixed alpha), this engine adjusts the equity
    allocation at every rebalancing step according to an ``AccLinearGlidepath``
    schedule.  Supports ``linear``, ``convex``, and ``concave`` shapes.
    """

    def __init__(self, params: StrategyParameters,
                 initial_equity: float = 0.80,
                 final_equity: float = 0.20,
                 shape: str = "linear") -> None:
        super().__init__(params)
        self._shape = shape
        self._glidepath = AccLinearGlidepath(
            initial_equity_allocation=initial_equity,
            final_equity_allocation=final_equity,
            years=params.time_horizon,
            shape=shape,
        )

    def _build_alpha_schedule(self, t_steps: int) -> np.ndarray:
        """Build per-step alpha schedule from glidepath parameters."""
        a0 = self._glidepath.initial_equity_allocation
        a1 = self._glidepath.final_equity_allocation
        max_t = max(t_steps - 1, 1)
        alpha_schedule = np.empty(t_steps, dtype=np.float64)
        for t in range(t_steps):
            tau = t / max_t
            if self._shape == "linear":
                alpha_schedule[t] = a0 + (a1 - a0) * tau
            elif self._shape == "convex":
                alpha_schedule[t] = a0 + (a1 - a0) * tau ** 2
            else:  # concave
                alpha_schedule[t] = a0 + (a1 - a0) * (1.0 - (1.0 - tau) ** 2)
        return alpha_schedule

    def _calculate_floor(self, t_steps: int, exprdt: np.ndarray) -> np.ndarray:
        """Glidepath has no floor; return zeros."""
        return np.zeros(t_steps + 1)

    def _initialize_state(
        self,
        t_steps: int,
        n_sims: int,
        Pt_f: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Initialize glidepath state with initial equity allocation."""
        Mt_f = np.zeros((t_steps + 1, n_sims))
        Xt_f = np.zeros((t_steps + 1, n_sims))
        Vt_f = np.zeros((t_steps + 1, n_sims))

        # Build alpha schedule and get initial alpha
        self._alpha_schedule = self._build_alpha_schedule(t_steps)
        alpha_0 = self._alpha_schedule[0]

        # Set initial conditions at t=0
        Vt_f[0, :] = self._params.initial_wealth
        Xt_f[0, :] = alpha_0 * self._params.initial_wealth
        Mt_f[0, :] = Vt_f[0, :] - Xt_f[0, :]

        return Mt_f, Xt_f, Vt_f

    def _call_core_function(
        self,
        t_steps: int,
        n_sims: int,
        exprdt: np.ndarray,
        gross_returns: np.ndarray,
        Pt_f: np.ndarray,
        Mt_f: np.ndarray,
        Xt_f: np.ndarray,
        Vt_f: np.ndarray,
        contributions_arr: np.ndarray,
    ) -> None:
        """Execute the glidepath core loop with time-varying alpha."""
        C = self._params.step_contribution
        step_inflation = self._params.step_inflation
        
        _run_glidepath_accumulation_core(
            t_steps, n_sims, self._alpha_schedule, exprdt, gross_returns,
            Mt_f, Xt_f, Vt_f, C, step_inflation, contributions_arr,
        )


class AccConstantMixEngine(BaseAccumulationEngine):
    """Run the Constant Mix strategy over a matrix of market returns."""

    def _calculate_floor(self, t_steps: int, exprdt: np.ndarray) -> np.ndarray:
        """Constant Mix has no floor; return zeros."""
        return np.zeros(t_steps + 1)

    def _initialize_state(
        self,
        t_steps: int,
        n_sims: int,
        Pt_f: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Initialize Constant Mix state with fixed equity allocation."""
        Mt_f = np.zeros((t_steps + 1, n_sims))
        Xt_f = np.zeros((t_steps + 1, n_sims))
        Vt_f = np.zeros((t_steps + 1, n_sims))

        # Set initial conditions at t=0
        alpha = self._params.Lambda / 100.0
        Vt_f[0, :] = self._params.initial_wealth
        Xt_f[0, :] = alpha * self._params.initial_wealth
        Mt_f[0, :] = Vt_f[0, :] - Xt_f[0, :]

        return Mt_f, Xt_f, Vt_f

    def _call_core_function(
        self,
        t_steps: int,
        n_sims: int,
        exprdt: np.ndarray,
        gross_returns: np.ndarray,
        Pt_f: np.ndarray,
        Mt_f: np.ndarray,
        Xt_f: np.ndarray,
        Vt_f: np.ndarray,
        contributions_arr: np.ndarray,
    ) -> None:
        """Execute the Constant Mix core loop."""
        alpha = self._params.Lambda / 100.0
        C = self._params.step_contribution
        step_inflation = self._params.step_inflation
        
        _run_cm_accumulation_core(
            t_steps, n_sims, alpha, exprdt, gross_returns,
            Mt_f, Xt_f, Vt_f, C, step_inflation, contributions_arr,
        )

