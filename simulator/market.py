"""Geometric Brownian Motion market simulator."""

from __future__ import annotations

import numpy as np

from .parameters import StrategyParameters


class MarketSimulator:
    """Generates risky-asset return paths via GBM.

    Parameters
    ----------
    params : StrategyParameters
        Strategy configuration (per-step drift and vol are derived
        from the chosen ``rebalance_freq``).
    """

    def __init__(self, params: StrategyParameters) -> None:
        self._params = params

    def generate_returns(
        self, seed: int | None = 42
    ) -> np.ndarray:
        """Sample simple returns.
        Returns
        -------
        np.ndarray, shape ``(steps,n_simulations)``
        """
        rng = np.random.default_rng(seed)
        return rng.normal(
            self._params.step_mu-self._params.step_sigma**2/2, 
            self._params.step_sigma,
            size=(self._params.n_steps,self._params.n_simulations),
        )

# TODO add a portfolio adaptation