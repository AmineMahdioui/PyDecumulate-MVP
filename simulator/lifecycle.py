from __future__ import annotations

import dataclasses
import warnings

import numpy as np
from dataclasses import dataclass

from .accumulation import (
    AccumulationResult,
    AccCPPIEngine,
    AccGlidepathEngine,
    AccConstantMixEngine,
)
from .decumulation import (
    DecumulationResult,
    DecCPPIEngine,
    DecConstantMixEngine,
    DecGlidepathEngine,
)
from .parameters import StrategyParameters

_VALID_STRATEGIES = {"cppi", "glidepath", "cm"}


@dataclass
class LifecycleResult:
    """Combined result of a full accumulation and decumulation lifecycle simulation."""
    accumulation: AccumulationResult
    decumulation: DecumulationResult
    retirement_wealths_nominal: np.ndarray   # shape (n_sims,) \u2014 nominal wealth at retirement
    retirement_risky_allocation: np.ndarray  # shape (n_sims,) \u2014 risky fraction at retirement


class LifecycleSimulator:
    """
    Orchestrates a full lifecycle simulation with configurable accumulation
    and decumulation strategies.

    Supported strategies
    --------------------
    acc_strategy : ``"cppi"`` | ``"glidepath"`` | ``"cm"``
    dec_strategy : ``"cppi"`` | ``"glidepath"`` | ``"cm"``

    Pathwise handoff
    ----------------
    * ``dec_strategy="cm"``        -> :class:`DecConstantMixEngine` — full per-path handoff
    * ``dec_strategy="glidepath"`` -> :class:`DecGlidepathEngine`   — full per-path handoff
    * ``dec_strategy="cppi"``      -> :class:`DecCPPIEngine`         — scalar median handoff
      (DecCPPIEngine does not accept per-path starting wealth; a ``UserWarning``
      is issued and the median retirement pot is used as the uniform starting value)

    Parameters
    ----------
    acc_params / dec_params:
        Separate :class:`StrategyParameters` for each phase.
    acc_strategy / dec_strategy:
        Strategy names (case-insensitive).  Default: acc=``"cppi"``, dec=``"cm"``.
    acc_initial_equity / acc_final_equity / acc_glidepath_shape:
        Glidepath knobs for the accumulation phase (used only when
        ``acc_strategy="glidepath"``).
    dec_initial_equity / dec_final_equity / dec_glidepath_shape:
        Glidepath knobs for the decumulation phase (used only when
        ``dec_strategy="glidepath"``).
    """

    def __init__(
        self,
        acc_params: StrategyParameters,
        dec_params: StrategyParameters,
        # -- Accumulation strategy --------------------------------------------
        acc_strategy: str = "cppi",
        acc_initial_equity: float = 0.80,
        acc_final_equity: float = 0.20,
        acc_glidepath_shape: str = "linear",
        # -- Decumulation strategy --------------------------------------------
        dec_strategy: str = "cm",
        dec_initial_equity: float = 0.60,
        dec_final_equity: float = 0.20,
        dec_glidepath_shape: str = "linear",
    ) -> None:
        self._acc_params = acc_params
        self._dec_params = dec_params

        self._acc_strategy = acc_strategy.lower()
        self._dec_strategy = dec_strategy.lower()

        self._acc_initial_equity  = acc_initial_equity
        self._acc_final_equity    = acc_final_equity
        self._acc_glidepath_shape = acc_glidepath_shape
        self._dec_initial_equity  = dec_initial_equity
        self._dec_final_equity    = dec_final_equity
        self._dec_glidepath_shape = dec_glidepath_shape

        if self._acc_strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"acc_strategy must be one of {_VALID_STRATEGIES}, "
                f"got '{acc_strategy}'."
            )
        if self._dec_strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"dec_strategy must be one of {_VALID_STRATEGIES}, "
                f"got '{dec_strategy}'."
            )

    # ------------------------------------------------------------------
    def _build_acc_engine(self):
        s = self._acc_strategy
        if s == "cppi":
            return AccCPPIEngine(self._acc_params)
        elif s == "glidepath":
            return AccGlidepathEngine(
                self._acc_params,
                initial_equity=self._acc_initial_equity,
                final_equity=self._acc_final_equity,
                shape=self._acc_glidepath_shape,
            )
        else:  # cm
            return AccConstantMixEngine(self._acc_params)

    def _build_dec_engine(self, override_wealth: float | None = None):
        s = self._dec_strategy
        params = self._dec_params
        if override_wealth is not None:
            params = dataclasses.replace(params, initial_wealth=override_wealth)
        if s == "cppi":
            return DecCPPIEngine(params)
        elif s == "glidepath":
            return DecGlidepathEngine(
                params,
                initial_equity=self._dec_initial_equity,
                final_equity=self._dec_final_equity,
                shape=self._dec_glidepath_shape,
            )
        else:  # cm
            return DecConstantMixEngine(params)

    # ------------------------------------------------------------------
    def run(
        self,
        acc_returns: np.ndarray,
        dec_returns: np.ndarray,
        riskless_acc: np.ndarray | None = None,
        riskless_dec: np.ndarray | None = None,
    ) -> LifecycleResult:
        """Run the full lifecycle and return a combined result.

        Parameters
        ----------
        acc_returns:
            Log-return matrix for the accumulation phase, shape ``(t_acc, n_sims)``.
        dec_returns:
            Log-return matrix for the decumulation phase, shape ``(t_dec, n_sims)``.
        riskless_acc, riskless_dec:
            Optional per-step risk-free return matrices.
        """
        n_sims_acc = self._acc_params.n_simulations
        n_sims_dec = self._dec_params.n_simulations

        # --- Consistency validation -------------------------------------------
        if n_sims_acc != n_sims_dec:
            raise ValueError(
                f"accumulation and decumulation must use the same n_simulations, "
                f"got acc={n_sims_acc} vs dec={n_sims_dec}."
            )
        n_sims = n_sims_acc

        if acc_returns.shape[1] != n_sims:
            raise ValueError(
                f"acc_returns has {acc_returns.shape[1]} paths but params expect {n_sims}."
            )
        if dec_returns.shape[1] != n_sims:
            raise ValueError(
                f"dec_returns has {dec_returns.shape[1]} paths but params expect {n_sims}."
            )

        # --- Step 1: Run accumulation -----------------------------------------
        acc_result = self._build_acc_engine().run(acc_returns, riskless_acc)

        # --- Step 2: Extract terminal retirement state (nominal currency) -----
        retirement_wealths_nominal  = acc_result.portfolio_values_nominal[-1, :]
        retirement_risky_allocation = acc_result.risky_allocation[-1, :]

        # --- Step 3: Run decumulation with appropriate handoff mode -----------
        if self._dec_strategy == "cppi":
            # DecCPPIEngine does not support per-path initial wealth.
            # Fall back to the median retirement pot as a scalar initial_wealth.
            median_pot = float(np.median(retirement_wealths_nominal))
            warnings.warn(
                f"dec_strategy='cppi' does not support full pathwise handoff. "
                f"Using median retirement pot (\u20ac{median_pot:,.0f}) as a scalar "
                f"initial_wealth for all paths. "
                f"For path-level heterogeneity use dec_strategy='cm' or 'glidepath'.",
                UserWarning,
                stacklevel=2,
            )
            dec_result = self._build_dec_engine(override_wealth=median_pot).run(
                dec_returns, riskless_dec
            )
        else:
            # cm and glidepath both accept (initial_wealths, initial_risky_allocation)
            dec_result = self._build_dec_engine().run(
                dec_returns,
                riskless_dec,
                initial_wealths=retirement_wealths_nominal,
                initial_risky_allocation=retirement_risky_allocation,
            )

        return LifecycleResult(
            accumulation=acc_result,
            decumulation=dec_result,
            retirement_wealths_nominal=retirement_wealths_nominal,
            retirement_risky_allocation=retirement_risky_allocation,
        )
