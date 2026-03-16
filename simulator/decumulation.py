from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from numba import njit, prange
from dataclasses import dataclass
from .parameters import StrategyParameters
from .inflation import deflate

@dataclass
class DecumulationResult:
    """Output of a Decumulation simulation run (CPPI or Constant Mix)."""
    # Real (deflated) outputs; used for dashboard charts and KPIs
    portfolio_values: np.ndarray  # shape (t_steps, n_sims); portfolio value (real)
    floor_values: np.ndarray      # shape (t_steps,); guaranteed floor (zeros for CM, real)
    risky_paths: np.ndarray       # shape (t_steps, n_sims); buy-and-hold risky benchmark (real)
    risky_allocation: np.ndarray  # shape (t_steps, n_sims); fraction allocated to risky asset
    withdrawals_made: np.ndarray  # shape (t_steps, n_sims); actual withdrawal per step (real)
    # turnover: np.ndarray          # shape (t_steps, n_sims); absolute value of assets traded (real)
    # Nominal (pre-deflation) outputs; used for lifecycle integration and debugging
    portfolio_values_nominal: np.ndarray | None  # shape (t_steps, n_sims); nominal portfolio value
    floor_values_nominal: np.ndarray | None      # shape (t_steps,); nominal guaranteed floor
    withdrawals_made_nominal: np.ndarray | None  # shape (t_steps, n_sims); nominal withdrawals


class BaseDecumulationEngine(ABC):
    """Abstract base class for decumulation strategies.
    
    Consolidates common initialization, array management, deflation, and result
    packaging logic. Child classes override _calculate_floor(), _initialize_state(),
    and _call_core_function() to implement specific strategies (CPPI, Glidepath, CM).
    """

    def __init__(self, params: StrategyParameters) -> None:
        self._params = params

    @abstractmethod
    def _calculate_floor(self, t_steps: int) -> np.ndarray:
        """Calculate deterministic floor for the strategy.
        
        Returns array of shape (t_steps + 1,). CPPI overrides with PV-based
        floor; others return zeros.
        """
        pass

    @abstractmethod
    def _initialize_state(
        self,
        t_steps: int,
        n_sims: int,
        Pt_f: np.ndarray,
        initial_wealths: np.ndarray | None,
        initial_risky_allocation: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Initialize state arrays Mt_f, Xt_f, Vt_f at t=0.
        
        Supports lifecycle handoff (per-path initial state) or standalone
        (uniform initial wealth).
        
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
        withdrawals_arr: np.ndarray,
    ) -> None:
        """Execute the Numba core simulation loop in-place.
        
        Modifies Mt_f, Xt_f, Vt_f, withdrawals_arr arrays in-place.
        """
        pass

    def _validate_lifecycle_handoff(
        self,
        n_sims: int,
        initial_wealths: np.ndarray | None,
        initial_risky_allocation: np.ndarray | None,
    ) -> None:
        """Validate lifecycle handoff arguments.
        
        Raises ValueError if only one of initial_wealths or initial_risky_allocation
        is provided, or if shapes don't match.
        """
        _has_iw = initial_wealths is not None
        _has_ira = initial_risky_allocation is not None
        
        if _has_iw ^ _has_ira:
            raise ValueError(
                "lifecycle handoff requires both initial_wealths and "
                "initial_risky_allocation, or neither."
            )
        if _has_iw:
            if initial_wealths.shape != (n_sims,):
                raise ValueError(
                    f"initial_wealths must have shape ({n_sims},), "
                    f"got {initial_wealths.shape}."
                )
            if initial_risky_allocation.shape != (n_sims,):
                raise ValueError(
                    f"initial_risky_allocation must have shape ({n_sims},), "
                    f"got {initial_risky_allocation.shape}."
                )
            if np.any(initial_risky_allocation < 0.0) or np.any(initial_risky_allocation > 1.0):
                raise ValueError(
                    "initial_risky_allocation values must be in [0.0, 1.0]."
                )

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
            exprdt = np.full((t_steps, n_sims), np.exp(self._params.step_rf))
        
        gross_returns = np.exp(asset_log_returns)
        return exprdt, gross_returns

    def run(
        self,
        asset_log_returns: np.ndarray,
        riskless_returns: np.ndarray | None = None,
        initial_wealths: np.ndarray | None = None,
        initial_risky_allocation: np.ndarray | None = None,
        deflated: bool = True,
        include_nominal_arrays: bool = True,
    ) -> DecumulationResult:
        """Execute the decumulation strategy simulation.
        
        Parameters
        ----------
        asset_log_returns : np.ndarray
            Log-returns of risky asset, shape (t_steps, n_sims).
        riskless_returns : np.ndarray | None
            Optional log-returns of risk-free asset; if None, uses params.step_rf.
        initial_wealths : np.ndarray | None
            Per-path nominal wealth at retirement, shape (n_sims,).
            Provide together with initial_risky_allocation for lifecycle handoff.
        initial_risky_allocation : np.ndarray | None
            Per-path risky fraction at retirement, shape (n_sims,).
            Provide together with initial_wealths for lifecycle handoff.
        deflated : bool
            If True (default), outputs are deflated to real terms.
        include_nominal_arrays : bool
            If True, retain nominal (pre-deflation) arrays for lifecycle integration.
            If False, set to None to save memory.
        
        Returns
        -------
        DecumulationResult
            Deflated outputs ready for dashboard visualization and KPIs.
        """
        t_steps, n_sims = asset_log_returns.shape

        # 1. Validate lifecycle handoff arguments
        self._validate_lifecycle_handoff(n_sims, initial_wealths, initial_risky_allocation)

        # 2. Setup market arrays
        exprdt, gross_returns = self._setup_market_arrays(asset_log_returns, riskless_returns)

        # 3. Calculate floor (strategy-specific)
        Pt_f = self._calculate_floor(t_steps)

        # 4. Initialize state arrays
        Mt_f, Xt_f, Vt_f = self._initialize_state(
            t_steps, n_sims, Pt_f, initial_wealths, initial_risky_allocation
        )

        # 5. Initialize withdrawal tracking
        withdrawals_arr = np.zeros((t_steps, n_sims))
        # turnover_arr = np.zeros((t_steps, n_sims))

        # 6. Call strategy-specific Numba core function
        self._call_core_function(
            t_steps, n_sims, exprdt, gross_returns, Pt_f,
            Mt_f, Xt_f, Vt_f, withdrawals_arr,
        )

        # 7. Calculate fractional allocation
        risky_allocation = Xt_f / np.maximum(Vt_f, 1e-8)

        # 8. Risky benchmark path (respects per-path initial wealth for lifecycle handoff)
        if initial_wealths is not None:
            risky_asset_paths = initial_wealths[np.newaxis, :] * np.cumprod(gross_returns, axis=0)
        else:
            risky_asset_paths = self._params.initial_wealth * np.cumprod(gross_returns, axis=0)

        # 9. Retain nominal arrays before deflation (for lifecycle integration)
        portfolio_values_nominal = Vt_f[1:].copy() if include_nominal_arrays else None
        floor_values_nominal = Pt_f[1:].copy() if include_nominal_arrays else None
        withdrawals_made_nominal = withdrawals_arr.copy() if include_nominal_arrays else None
        # turnover_nominal = turnover_arr.copy() if include_nominal_arrays else None

        # 10. Deflate outputs to real terms (today's purchasing power)
        _ir = self._params.annual_inflation_rate
        _dt = self._params.dt
        
        def _d(arr: np.ndarray) -> np.ndarray:
            return deflate(arr, _ir, _dt, start_step=1) if deflated else arr

        return DecumulationResult(
            portfolio_values=_d(Vt_f[1:]),
            floor_values=_d(Pt_f[1:]),
            risky_paths=_d(risky_asset_paths),
            risky_allocation=risky_allocation[1:],
            withdrawals_made=_d(withdrawals_arr),
            # turnover=_d(turnover_arr),
            portfolio_values_nominal=portfolio_values_nominal,
            floor_values_nominal=floor_values_nominal,
            withdrawals_made_nominal=withdrawals_made_nominal,
            # turnover_nominal=turnover_nominal,
        )



@njit(parallel=True)
def _run_cppi_decumulation_core(t_steps, n_sims, m, exprdt, gross_returns, Pt_f, Mt_f, Xt_f, Vt_f, step_withdrawal, step_inflation, withdrawals_arr, 
                                # turnover_arr
                                ):
    """
    Core Numba loop for a CPPI Decumulation strategy.
    Withdrawals grow with inflation each step.
    """
    for i in prange(n_sims):
        for t in range(t_steps):
            # Calculate cumulative inflation factor for this step
            current_inflation_factor = (1.0 + step_inflation) ** t
            current_nominal_withdrawal = step_withdrawal * current_inflation_factor
            
            # 1. Market Growth (Before withdrawal)
            Mt_grown = Mt_f[t, i] * exprdt[t, i]
            Xt_grown = Xt_f[t, i] * gross_returns[t, i]
            Vt_grown = Mt_grown + Xt_grown
            
            # 2. Withdrawal Phase (inflation-adjusted)
            actual_withdrawal = min(current_nominal_withdrawal, Vt_grown)
            withdrawals_arr[t, i] = actual_withdrawal
            Vt_f[t+1, i] = Vt_grown - actual_withdrawal
            
            # 3. Rebalancing Phase (CPPI logic)
            cushion = max(Vt_f[t+1, i] - Pt_f[t+1], 0.0)
            Xt_target = min(m * cushion, Vt_f[t+1, i])  # no-leverage constraint
            Mt_target = Vt_f[t+1, i] - Xt_target
            
            # 4. Turnover Tracking
            # trade_amount = Xt_target - Xt_grown
            # turnover_arr[t, i] = abs(trade_amount)
            
            # 5. Lock in the new state
            Xt_f[t+1, i] = Xt_target
            Mt_f[t+1, i] = Mt_target

class DecCPPIEngine(BaseDecumulationEngine):
    """Run the CPPI strategy over a matrix of market returns."""

    def _calculate_floor(self, t_steps: int) -> np.ndarray:
        """Calculate the deterministic floor for CPPI (PV of inflation-adjusted withdrawals)."""
        W = self._params.step_withdrawal
        step_inflation = self._params.step_inflation
        r_step = float(np.expm1(self._params.step_rf))
        g_step = step_inflation
        
        Pt_f = np.zeros(t_steps + 1)
        # Build backwards from t_steps-1 to 0
        for k in range(t_steps - 1, -1, -1):
            # Withdrawal at step k (in nominal terms)
            withdrawal_at_k = W * ((1.0 + g_step) ** k)
            # Discount factor for one step
            discount = 1.0 / (1.0 + r_step) if r_step > 1e-12 else 1.0
            Pt_f[k] = withdrawal_at_k + Pt_f[k + 1] * discount
        
        # Apply the floor percentage factor
        Pt_f = Pt_f * (self._params.floor_pct / 100.0)
        return Pt_f

    def _initialize_state(
        self,
        t_steps: int,
        n_sims: int,
        Pt_f: np.ndarray,
        initial_wealths: np.ndarray | None,
        initial_risky_allocation: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Initialize CPPI state with cushion-based risky allocation."""
        Mt_f = np.zeros((t_steps + 1, n_sims))
        Xt_f = np.zeros((t_steps + 1, n_sims))
        Vt_f = np.zeros((t_steps + 1, n_sims))

        m = self._params.cppi_multiplier
        _has_iw = initial_wealths is not None

        # Set initial conditions at t=0
        if _has_iw:
            Vt_f[0, :] = initial_wealths
        else:
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
        withdrawals_arr: np.ndarray,
    ) -> None:
        """Execute the CPPI core loop."""
        m = self._params.cppi_multiplier
        W = self._params.step_withdrawal
        step_inflation = self._params.step_inflation
        
        _run_cppi_decumulation_core(
            t_steps, n_sims, m, exprdt, gross_returns, Pt_f,
            Mt_f, Xt_f, Vt_f, W, step_inflation, withdrawals_arr,
        )


@njit(parallel=True)
def _run_cm_decumulation_core(t_steps, n_sims, alpha, exprdt, gross_returns, Mt_f, Xt_f, Vt_f, step_withdrawal, step_inflation, withdrawals_arr, 
                            #   turnover_arr
                              ):
    """
    Core Numba loop for a Constant Mix Decumulation strategy.
    Withdrawals grow with inflation each step.
    """
    for i in prange(n_sims):
        for t in range(t_steps):
            # Calculate cumulative inflation factor for this step
            current_inflation_factor = (1.0 + step_inflation) ** t
            current_nominal_withdrawal = step_withdrawal * current_inflation_factor
            
            # 1. Market Growth (Before withdrawal)
            Mt_grown = Mt_f[t, i] * exprdt[t, i]
            Xt_grown = Xt_f[t, i] * gross_returns[t, i]
            Vt_grown = Mt_grown + Xt_grown
            
            # 2. Withdrawal Phase (inflation-adjusted)
            actual_withdrawal = min(current_nominal_withdrawal, Vt_grown)
            withdrawals_arr[t, i] = actual_withdrawal
            Vt_f[t+1, i] = Vt_grown - actual_withdrawal
            
            # 3. Rebalancing Phase
            Xt_target = alpha * Vt_f[t+1, i]
            Mt_target = Vt_f[t+1, i] - Xt_target
            
            # 4. Turnover Tracking
            # trade_amount = Xt_target - Xt_grown
            # turnover_arr[t, i] = abs(trade_amount)
            
            # 5. Lock in the new state
            Xt_f[t+1, i] = Xt_target
            Mt_f[t+1, i] = Mt_target



class DecConstantMixEngine(BaseDecumulationEngine):
    """Run the Constant Mix strategy over a matrix of market returns."""

    def _calculate_floor(self, t_steps: int) -> np.ndarray:
        """Constant Mix has no floor; return zeros."""
        return np.zeros(t_steps + 1)

    def _initialize_state(
        self,
        t_steps: int,
        n_sims: int,
        Pt_f: np.ndarray,
        initial_wealths: np.ndarray | None,
        initial_risky_allocation: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Initialize Constant Mix state with fixed equity allocation or per-path handoff."""
        Mt_f = np.zeros((t_steps + 1, n_sims))
        Xt_f = np.zeros((t_steps + 1, n_sims))
        Vt_f = np.zeros((t_steps + 1, n_sims))

        alpha = self._params.Lambda / 100.0
        _has_iw = initial_wealths is not None

        # Set initial conditions at t=0 (pathwise handoff or scalar fallback)
        if _has_iw:
            # Lifecycle handoff: per-path terminal accumulation state
            Vt_f[0, :] = initial_wealths
            Xt_f[0, :] = initial_wealths * initial_risky_allocation
            Mt_f[0, :] = initial_wealths - Xt_f[0, :]
        else:
            # Standalone decumulation: uniform initial wealth, CM target alpha
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
        withdrawals_arr: np.ndarray,
    ) -> None:
        """Execute the Constant Mix core loop."""
        alpha = self._params.Lambda / 100.0
        W = self._params.step_withdrawal
        step_inflation = self._params.step_inflation
        
        _run_cm_decumulation_core(
            t_steps, n_sims, alpha, exprdt, gross_returns,
            Mt_f, Xt_f, Vt_f, W, step_inflation, withdrawals_arr,
        )


# ---------------------------------------------------------------------------
# Glidepath Decumulation
# ---------------------------------------------------------------------------

@njit(parallel=True)
def _run_glidepath_decumulation_core(
    t_steps, n_sims, alpha_schedule, exprdt, gross_returns,
    Mt_f, Xt_f, Vt_f, step_withdrawal, step_inflation,
    withdrawals_arr, 
    # turnover_arr,
):
    """
    Core Numba loop for a Glidepath Decumulation strategy.

    ``alpha_schedule`` is a 1-D array of shape ``(t_steps,)`` giving the
    risky fraction at each step — identical mechanics to the CM kernel but
    with a time-varying alpha instead of a scalar.
    """
    for i in prange(n_sims):
        for t in range(t_steps):
            alpha = alpha_schedule[t]
            current_inflation_factor = (1.0 + step_inflation) ** t
            current_nominal_withdrawal = step_withdrawal * current_inflation_factor

            # 1. Market Growth
            Mt_grown = Mt_f[t, i] * exprdt[t, i]
            Xt_grown = Xt_f[t, i] * gross_returns[t, i]
            Vt_grown = Mt_grown + Xt_grown

            # 2. Withdrawal Phase (inflation-adjusted, clipped to available wealth)
            actual_withdrawal = min(current_nominal_withdrawal, Vt_grown)
            withdrawals_arr[t, i] = actual_withdrawal
            Vt_f[t + 1, i] = Vt_grown - actual_withdrawal

            # 3. Rebalancing Phase — step-specific alpha
            Xt_target = alpha * Vt_f[t + 1, i]
            Mt_target = Vt_f[t + 1, i] - Xt_target

            # 4. Turnover Tracking
            # turnover_arr[t, i] = abs(Xt_target - Xt_grown)

            # 5. Lock in new state
            Xt_f[t + 1, i] = Xt_target
            Mt_f[t + 1, i] = Mt_target


class DecGlidepathEngine(BaseDecumulationEngine):
    """Run a time-varying Glidepath strategy during decumulation.

    The risky equity allocation glides deterministically from
    ``initial_equity`` (at retirement) down to ``final_equity`` (at end of
    horizon) following one of three convexity shapes: ``"linear"``,
    ``"convex"``, or ``"concave"``.

    The same tau-formula as ``AccGlidepathEngine`` is used:
    ``tau = step / max(n_steps - 1, 1)``, guaranteeing exact endpoint
    values at both tau=0 and tau=1.

    Full **pathwise handoff** from any accumulation engine is supported via
    ``initial_wealths``.  The glidepath schedule always starts at
    ``initial_equity`` for all paths regardless of the terminal risky fraction
    coming from accumulation — the decumulation engine owns its allocation
    from retirement onwards.  ``initial_risky_allocation`` is accepted for
    API symmetry with ``DecConstantMixEngine`` but is **not used** for
    allocation.
    """

    def __init__(
        self,
        params: StrategyParameters,
        initial_equity: float = 0.60,
        final_equity: float = 0.20,
        shape: str = "linear",
    ) -> None:
        super().__init__(params)
        if shape not in {"linear", "convex", "concave"}:
            raise ValueError(
                f"shape must be 'linear', 'convex', or 'concave', got '{shape}'."
            )
        self._initial_equity = float(initial_equity)
        self._final_equity = float(final_equity)
        self._shape = shape

    def _build_alpha_schedule(self, t_steps: int) -> np.ndarray:
        """Return a ``(t_steps,)`` array of per-step risky fractions."""
        a0, a1 = self._initial_equity, self._final_equity
        tau = np.arange(t_steps, dtype=float) / max(t_steps - 1, 1)
        if self._shape == "linear":
            schedule = a0 + (a1 - a0) * tau
        elif self._shape == "convex":
            schedule = a0 + (a1 - a0) * tau ** 2
        else:  # concave
            schedule = a0 + (a1 - a0) * (1.0 - (1.0 - tau) ** 2)
        return np.clip(schedule, 0.0, 1.0)

    def _calculate_floor(self, t_steps: int) -> np.ndarray:
        """Glidepath has no floor; return zeros."""
        return np.zeros(t_steps + 1)

    def _initialize_state(
        self,
        t_steps: int,
        n_sims: int,
        Pt_f: np.ndarray,
        initial_wealths: np.ndarray | None,
        initial_risky_allocation: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Initialize glidepath state with initial equity allocation."""
        Mt_f = np.zeros((t_steps + 1, n_sims))
        Xt_f = np.zeros((t_steps + 1, n_sims))
        Vt_f = np.zeros((t_steps + 1, n_sims))

        # Build alpha schedule and get initial alpha
        self._alpha_schedule = self._build_alpha_schedule(t_steps)
        alpha0 = float(self._alpha_schedule[0])

        _has_iw = initial_wealths is not None

        # Initial conditions (t = 0) - glidepath schedule starts at initial_equity for all paths
        if _has_iw:
            # Lifecycle handoff: per-path nominal wealth; glidepath sets alpha
            Vt_f[0, :] = initial_wealths
            Xt_f[0, :] = initial_wealths * alpha0
            Mt_f[0, :] = initial_wealths - Xt_f[0, :]
        else:
            W0 = self._params.initial_wealth
            Vt_f[0, :] = W0
            Xt_f[0, :] = alpha0 * W0
            Mt_f[0, :] = W0 - Xt_f[0, :]

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
        withdrawals_arr: np.ndarray,
    ) -> None:
        """Execute the glidepath core loop with time-varying alpha."""
        W = self._params.step_withdrawal
        step_inflation = self._params.step_inflation
        
        _run_glidepath_decumulation_core(
            t_steps, n_sims, self._alpha_schedule, exprdt, gross_returns,
            Mt_f, Xt_f, Vt_f, W, step_inflation, withdrawals_arr,
        )