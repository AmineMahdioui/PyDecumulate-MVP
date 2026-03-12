from __future__ import annotations

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


@dataclass
class LifecycleResult:
    """Combined result of a full accumulation and decumulation lifecycle simulation."""
    accumulation: AccumulationResult
    decumulation: DecumulationResult
    retirement_wealths_nominal: np.ndarray   # shape (n_sims,) \u2014 nominal wealth at retirement
    retirement_risky_allocation: np.ndarray  # shape (n_sims,) \u2014 risky fraction at retirement


class LifecycleSimulator:
    """
    Orchestrates a full lifecycle simulation:
      1. Accumulation phase (CPPI) and terminal nominal state per path
      2. Decumulation phase (Constant Mix) fed per-path terminal state

    The handoff uses nominal (pre-deflation) values so that wealth at
    retirement is expressed in retirement-date currency, not today's
    purchasing power.
    """

    def __init__(
        self,
        acc_params: StrategyParameters,
        dec_params: StrategyParameters,
    ) -> None:
        self._acc_params = acc_params
        self._dec_params = dec_params

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
            Optional per-step risk-free return matrices with the same shapes.
        """
        n_sims_acc = self._acc_params.n_simulations
        n_sims_dec = self._dec_params.n_simulations

        # --- Consistency validation -----------------------------------------------
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

        # --- Step 1: Run accumulation (CPPI) --------------------------------------
        acc_result = AccCPPIEngine(self._acc_params).run(acc_returns, riskless_acc)

        if acc_result.portfolio_values_nominal.shape[1] != n_sims:
            raise ValueError(
                f"Accumulation result has {acc_result.portfolio_values_nominal.shape[1]} "
                f"paths, expected {n_sims}."
            )

        # --- Step 2: Extract terminal retirement state (nominal) ------------------
        retirement_wealths_nominal  = acc_result.portfolio_values_nominal[-1, :]
        retirement_risky_allocation = acc_result.risky_allocation[-1, :]

        if retirement_wealths_nominal.shape != (n_sims,):
            raise ValueError(
                f"retirement_wealths_nominal shape mismatch: "
                f"{retirement_wealths_nominal.shape} != ({n_sims},)."
            )
        if retirement_risky_allocation.shape != (n_sims,):
            raise ValueError(
                f"retirement_risky_allocation shape mismatch: "
                f"{retirement_risky_allocation.shape} != ({n_sims},)."
            )

        # --- Step 3: Run decumulation (CM) with pathwise handoff ------------------
        dec_result = DecConstantMixEngine(self._dec_params).run(
            dec_returns,
            riskless_dec,
            initial_wealths=retirement_wealths_nominal,
            initial_risky_allocation=retirement_risky_allocation,
        )

        if dec_result.portfolio_values.shape[1] != n_sims:
            raise ValueError(
                f"Decumulation result has {dec_result.portfolio_values.shape[1]} "
                f"paths, expected {n_sims}."
            )

        return LifecycleResult(
            accumulation=acc_result,
            decumulation=dec_result,
            retirement_wealths_nominal=retirement_wealths_nominal,
            retirement_risky_allocation=retirement_risky_allocation,
        )
