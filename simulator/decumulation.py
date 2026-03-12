from __future__ import annotations
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
    turnover: np.ndarray          # shape (t_steps, n_sims); absolute value of assets traded (real)
    # Nominal (pre-deflation) outputs; used for lifecycle integration and debugging
    portfolio_values_nominal: np.ndarray  # shape (t_steps, n_sims); nominal portfolio value
    floor_values_nominal: np.ndarray      # shape (t_steps,); nominal guaranteed floor
    withdrawals_made_nominal: np.ndarray  # shape (t_steps, n_sims); nominal withdrawals
    turnover_nominal: np.ndarray          # shape (t_steps, n_sims); nominal turnover


@njit(parallel=True)
def _run_cppi_decumulation_core(t_steps, n_sims, m, exprdt, gross_returns, Pt_f, Mt_f, Xt_f, Vt_f, step_withdrawal, step_inflation, withdrawals_arr, turnover_arr):
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
            trade_amount = Xt_target - Xt_grown
            turnover_arr[t, i] = abs(trade_amount)
            
            # 5. Lock in the new state
            Xt_f[t+1, i] = Xt_target
            Mt_f[t+1, i] = Mt_target

class DecCPPIEngine:
    """Run the CPPI strategy over a matrix of market returns."""

    def __init__(self, params: StrategyParameters) -> None:
        self._params = params
        
    def run(self, asset_log_returns: np.ndarray, riskless_returns: np.ndarray = None) -> DecumulationResult:
        """Prepares the data and executes the CPPI algorithm."""
        t_steps, n_sims = asset_log_returns.shape
        
        m = self._params.cppi_multiplier
        W = self._params.step_withdrawal
        step_inflation = self._params.step_inflation
        
        # 1. Setup Constants 
        if riskless_returns is not None:
            if riskless_returns.shape != asset_log_returns.shape:
                raise ValueError("riskless_returns must be either None or have the same shape as asset_log_returns.")
            exprdt = np.exp(riskless_returns)
        else:
            exprdt = np.exp(self._params.step_rf).repeat(t_steps * n_sims).reshape(t_steps, n_sims)
        
        gross_returns = np.exp(asset_log_returns)
        
        # 2. Pre-calculate the deterministic Floor for a GROWING annuity
        # Floor at time k = PV of inflation-adjusted withdrawals from step k to T-1
        # Uses backward recursion: Pt_f[k] = W*(1+g)^k + Pt_f[k+1]/(1+r)
        r_step = float(np.expm1(self._params.step_rf))  # (1+r) - 1 = r
        g_step = step_inflation
        
        Pt_f = np.zeros(t_steps + 1)
        # Pt_f[t_steps] = 0 (already initialized)
        # Build backwards from t_steps-1 to 0
        for k in range(t_steps - 1, -1, -1):
            # Withdrawal at step k (in nominal terms)
            withdrawal_at_k = W * ((1.0 + g_step) ** k)
            # Discount factor for one step
            discount = 1.0 / (1.0 + r_step) if r_step > 1e-12 else 1.0
            Pt_f[k] = withdrawal_at_k + Pt_f[k + 1] * discount
        
        # Apply the floor percentage factor if the user only wants to guarantee a subset (e.g., 80%)
        Pt_f = Pt_f * (self._params.floor_pct / 100.0)
        
        # 3. Initialize tracking arrays
        Mt_f = np.zeros((t_steps + 1, n_sims))
        Xt_f = np.zeros((t_steps + 1, n_sims))
        Vt_f = np.zeros((t_steps + 1, n_sims))
        withdrawals_arr = np.zeros((t_steps, n_sims))
        turnover_arr = np.zeros((t_steps, n_sims))
        
        # 4. Set Initial Conditions at t=0
        Vt_f[0, :] = self._params.initial_wealth
        cushion_0 = np.maximum(Vt_f[0, :] - Pt_f[0], 0.0)
        
        initial_target_risky = m * cushion_0
        Xt_f[0, :] = np.minimum(initial_target_risky, Vt_f[0, :])
        Mt_f[0, :] = Vt_f[0, :] - Xt_f[0, :]
        
        # 5. Call the fast Numba function
        _run_cppi_decumulation_core(
            t_steps, n_sims, m, exprdt, gross_returns, Pt_f, 
            Mt_f, Xt_f, Vt_f, W, step_inflation, withdrawals_arr, turnover_arr
        )
        
        # 6. Calculate fractional allocation
        risky_allocation = Xt_f / np.maximum(Vt_f, 1e-8)
        risky_asset_paths = self._params.initial_wealth * np.cumprod(gross_returns, axis=0)
        
        # 7. Retain nominal arrays before deflation
        portfolio_values_nominal = Vt_f[1:].copy()
        floor_values_nominal     = Pt_f[1:].copy()
        withdrawals_made_nominal = withdrawals_arr.copy()
        turnover_nominal         = turnover_arr.copy()

        # 8. Deflate outputs to Real terms (today's purchasing power)
        _ir = self._params.annual_inflation_rate
        _dt = self._params.dt
        portfolio_values_real = deflate(Vt_f[1:],          _ir, _dt, start_step=1)
        floor_values_real     = deflate(Pt_f[1:],          _ir, _dt, start_step=1)
        risky_paths_real      = deflate(risky_asset_paths, _ir, _dt, start_step=1)
        withdrawals_made_real  = deflate(withdrawals_arr,  _ir, _dt, start_step=1)
        turnover_real         = deflate(turnover_arr,      _ir, _dt, start_step=1)

        return DecumulationResult(
            portfolio_values=portfolio_values_real,
            floor_values=floor_values_real,
            risky_paths=risky_paths_real,
            risky_allocation=risky_allocation[1:],
            withdrawals_made=withdrawals_made_real,
            turnover=turnover_real,
            portfolio_values_nominal=portfolio_values_nominal,
            floor_values_nominal=floor_values_nominal,
            withdrawals_made_nominal=withdrawals_made_nominal,
            turnover_nominal=turnover_nominal,
        )


@njit(parallel=True)
def _run_cm_decumulation_core(t_steps, n_sims, alpha, exprdt, gross_returns, Mt_f, Xt_f, Vt_f, step_withdrawal, step_inflation, withdrawals_arr, turnover_arr):
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
            trade_amount = Xt_target - Xt_grown
            turnover_arr[t, i] = abs(trade_amount)
            
            # 5. Lock in the new state
            Xt_f[t+1, i] = Xt_target
            Mt_f[t+1, i] = Mt_target



class DecConstantMixEngine:
    """Run the Constant Mix strategy over a matrix of market returns."""

    def __init__(self, params: StrategyParameters) -> None:
        self._params = params
        
    def run(
        self,
        asset_log_returns: np.ndarray,
        riskless_returns: np.ndarray | None = None,
        initial_wealths: np.ndarray | None = None,
        initial_risky_allocation: np.ndarray | None = None,
        deflated: bool = True,
    ) -> DecumulationResult:
        """Prepares the data and executes the Constant Mix algorithm.

        When ``initial_wealths`` and ``initial_risky_allocation`` are both
        provided (lifecycle handoff mode), the simulation starts from the
        per-path terminal state of the accumulation phase.  When both are
        ``None`` (standalone mode), it falls back to ``params.initial_wealth``
        with a uniform alpha allocation.
        """
        t_steps, n_sims = asset_log_returns.shape

        # ---- Validate lifecycle handoff arguments --------------------------------
        _has_iw  = initial_wealths is not None
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

        alpha = self._params.Lambda / 100.0  # Convert percentage to decimal
        W = self._params.step_withdrawal
        step_inflation = self._params.step_inflation
        
        # 1. Setup Constants 
        if riskless_returns is not None:
            if riskless_returns.shape != asset_log_returns.shape:
                raise ValueError("riskless_returns must be either None or have the same shape as asset_log_returns.")
            exprdt = np.exp(riskless_returns)
        else:
            exprdt = np.exp(self._params.step_rf).repeat(t_steps * n_sims).reshape(t_steps, n_sims)
            
        gross_returns = np.exp(asset_log_returns)
        
        # Constant mix has no floor, so we just track zeros
        Pt_f = np.zeros(t_steps + 1)
        
        # 2. Initialize tracking arrays
        Mt_f = np.zeros((t_steps + 1, n_sims))
        Xt_f = np.zeros((t_steps + 1, n_sims))
        Vt_f = np.zeros((t_steps + 1, n_sims))
        withdrawals_arr = np.zeros((t_steps, n_sims))
        turnover_arr = np.zeros((t_steps, n_sims))

        # 3. Set Initial Conditions at t=0 (pathwise handoff or scalar fallback)
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

        # 4. Call the fast, multi-threaded Numba function
        _run_cm_decumulation_core(
            t_steps, n_sims, alpha, exprdt, gross_returns, 
            Mt_f, Xt_f, Vt_f, W, step_inflation, withdrawals_arr, turnover_arr
        )
        
        # 5. Calculate fractional allocation (avoiding divide-by-zero if wealth hits 0)
        risky_allocation = Xt_f / np.maximum(Vt_f, 1e-8)

        # 6. Risky benchmark: full wealth invested in risky asset — respects pathwise starting wealth
        if _has_iw:
            risky_asset_paths = initial_wealths[np.newaxis, :] * np.cumprod(gross_returns, axis=0)
        else:
            risky_asset_paths = self._params.initial_wealth * np.cumprod(gross_returns, axis=0)

        # 7. Retain nominal arrays before deflation (for lifecycle integration)
        portfolio_values_nominal = Vt_f[1:].copy()
        floor_values_nominal     = Pt_f[1:].copy()
        withdrawals_made_nominal = withdrawals_arr.copy()
        turnover_nominal         = turnover_arr.copy()

        # 8. Deflate outputs to Real terms (today's purchasing power)
        _ir = self._params.annual_inflation_rate
        _dt = self._params.dt
        portfolio_values_real = deflate(Vt_f[1:],          _ir, _dt, start_step=1) if deflated else Vt_f[1:]
        floor_values_real     = deflate(Pt_f[1:],          _ir, _dt, start_step=1) if deflated else Pt_f[1:]
        risky_paths_real      = deflate(risky_asset_paths, _ir, _dt, start_step=1) if deflated else risky_asset_paths
        withdrawals_made_real = deflate(withdrawals_arr,   _ir, _dt, start_step=1) if deflated else withdrawals_arr
        turnover_real         = deflate(turnover_arr,      _ir, _dt, start_step=1) if deflated else turnover_arr

        return DecumulationResult(
            portfolio_values=portfolio_values_real,
            floor_values=floor_values_real,
            risky_paths=risky_paths_real,
            risky_allocation=risky_allocation[1:],
            withdrawals_made=withdrawals_made_real,
            turnover=turnover_real,
            portfolio_values_nominal=portfolio_values_nominal,
            floor_values_nominal=floor_values_nominal,
            withdrawals_made_nominal=withdrawals_made_nominal,
            turnover_nominal=turnover_nominal,
        )


# ---------------------------------------------------------------------------
# Glidepath Decumulation
# ---------------------------------------------------------------------------

@njit(parallel=True)
def _run_glidepath_decumulation_core(
    t_steps, n_sims, alpha_schedule, exprdt, gross_returns,
    Mt_f, Xt_f, Vt_f, step_withdrawal, step_inflation,
    withdrawals_arr, turnover_arr,
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
            turnover_arr[t, i] = abs(Xt_target - Xt_grown)

            # 5. Lock in new state
            Xt_f[t + 1, i] = Xt_target
            Mt_f[t + 1, i] = Mt_target


class DecGlidepathEngine:
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

    Parameters
    ----------
    params:
        Simulation parameters (``time_horizon``, ``annual_withdrawal``, etc.).
    initial_equity:
        Risky fraction at retirement (tau = 0).  Default ``0.60``.
    final_equity:
        Risky fraction at end of horizon (tau = 1).  Default ``0.20``.
    shape:
        ``"linear"`` | ``"convex"`` | ``"concave"``.  Default ``"linear"``.
    """

    def __init__(
        self,
        params: StrategyParameters,
        initial_equity: float = 0.60,
        final_equity: float = 0.20,
        shape: str = "linear",
    ) -> None:
        if shape not in {"linear", "convex", "concave"}:
            raise ValueError(
                f"shape must be 'linear', 'convex', or 'concave', got '{shape}'."
            )
        self._params         = params
        self._initial_equity = float(initial_equity)
        self._final_equity   = float(final_equity)
        self._shape          = shape

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

    def run(
        self,
        asset_log_returns: np.ndarray,
        riskless_returns: np.ndarray | None = None,
        initial_wealths: np.ndarray | None = None,
        initial_risky_allocation: np.ndarray | None = None,
        deflated: bool = True,
    ) -> DecumulationResult:
        """Execute the glidepath decumulation simulation.

        Parameters
        ----------
        asset_log_returns:
            Log-return matrix, shape ``(t_steps, n_sims)``.
        riskless_returns:
            Optional per-step risk-free log-returns, same shape.
        initial_wealths:
            Per-path nominal wealth at retirement, shape ``(n_sims,)``.
            Provide together with ``initial_risky_allocation`` for lifecycle
            pathwise handoff; omit for standalone decumulation.
        initial_risky_allocation:
            Per-path risky fraction at retirement, shape ``(n_sims,)``.
            Accepted for API symmetry — not used for allocation (the
            glidepath schedule starts at ``initial_equity`` for all paths).
        deflated:
            If ``True`` (default), outputs are deflated to real terms.
        """
        t_steps, n_sims = asset_log_returns.shape

        # ---- Validate lifecycle handoff arguments ----------------------------
        _has_iw  = initial_wealths is not None
        _has_ira = initial_risky_allocation is not None
        if _has_iw ^ _has_ira:
            raise ValueError(
                "lifecycle handoff requires both initial_wealths and "
                "initial_risky_allocation, or neither."
            )
        if _has_iw and initial_wealths.shape != (n_sims,):
            raise ValueError(
                f"initial_wealths must have shape ({n_sims},), "
                f"got {initial_wealths.shape}."
            )

        W              = self._params.step_withdrawal
        step_inflation = self._params.step_inflation
        alpha_schedule = self._build_alpha_schedule(t_steps)

        # ---- Market return arrays --------------------------------------------
        if riskless_returns is not None:
            if riskless_returns.shape != asset_log_returns.shape:
                raise ValueError("riskless_returns must match asset_log_returns shape.")
            exprdt = np.exp(riskless_returns)
        else:
            exprdt = np.full((t_steps, n_sims), np.exp(self._params.step_rf))

        gross_returns = np.exp(asset_log_returns)

        # ---- No CPPI floor — zeros throughout --------------------------------
        Pt_f = np.zeros(t_steps + 1)

        # ---- State arrays ----------------------------------------------------
        Mt_f            = np.zeros((t_steps + 1, n_sims))
        Xt_f            = np.zeros((t_steps + 1, n_sims))
        Vt_f            = np.zeros((t_steps + 1, n_sims))
        withdrawals_arr = np.zeros((t_steps, n_sims))
        turnover_arr    = np.zeros((t_steps, n_sims))

        # ---- Initial conditions (t = 0) -------------------------------------
        alpha0 = float(alpha_schedule[0])
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

        # ---- Numba kernel ---------------------------------------------------
        _run_glidepath_decumulation_core(
            t_steps, n_sims, alpha_schedule, exprdt, gross_returns,
            Mt_f, Xt_f, Vt_f, W, step_inflation, withdrawals_arr, turnover_arr,
        )

        # ---- Fractional allocation (guard against zero wealth) ---------------
        risky_allocation = Xt_f / np.maximum(Vt_f, 1e-8)

        # ---- Risky benchmark path -------------------------------------------
        if _has_iw:
            risky_asset_paths = initial_wealths[np.newaxis, :] * np.cumprod(gross_returns, axis=0)
        else:
            risky_asset_paths = self._params.initial_wealth * np.cumprod(gross_returns, axis=0)

        # ---- Retain nominal arrays ------------------------------------------
        portfolio_values_nominal = Vt_f[1:].copy()
        floor_values_nominal     = Pt_f[1:].copy()
        withdrawals_made_nominal = withdrawals_arr.copy()
        turnover_nominal         = turnover_arr.copy()

        # ---- Deflate to real terms ------------------------------------------
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
            turnover=_d(turnover_arr),
            portfolio_values_nominal=portfolio_values_nominal,
            floor_values_nominal=floor_values_nominal,
            withdrawals_made_nominal=withdrawals_made_nominal,
            turnover_nominal=turnover_nominal,
        )