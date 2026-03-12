from __future__ import annotations

from pathlib import Path
import pickle
from typing import Optional


import numpy.typing as npt
import pandas as pd


from simulator.parameters import LifecycleParameters
import yfinance as yf, numpy as np

_frequency_mapping = {
        "daily": None,
        "weekly": "W-FRI",
        "monthly": "ME",
        "quarterly": "QE",
        "yearly": "Y",
    }
FloatArray = npt.NDArray[np.float64]
_LOCAL_CSV = Path(__file__).resolve().parent / "data" / "historical_prices.csv"


class HistoricalBootstrapSimulator:
    """
    Historical i.i.d. paired bootstrap simulator for equity and bond log returns.

    The simulator:
    - downloads adjusted close prices for equity and bond
    - resamples prices to the requested rebalance frequency
    - computes aligned float64 log returns
    - stores a historical return matrix of shape (T_hist, 2)
    - bootstraps rows with replacement, preserving cross-asset correlation
    """

    _memory_cache: dict[tuple[str, str, str, str], pd.DataFrame] = {}



    def __init__(
        self,
        params: LifecycleParameters,
        cache_path: str | Path | None = None,
    ) -> None:
        self.params = params
        self.cache_path = Path(cache_path) if cache_path is not None else None


        prices = self._get_prices()
        returns_df = self._compute_log_returns(prices)

        if returns_df.empty:
            raise ValueError("Historical return matrix is empty after preprocessing.")

        self.return_matrix: FloatArray = returns_df.to_numpy(dtype=np.float64, copy=True)

    def generate_returns(
        self,
        n_steps: Optional[int] = None,
        n_simulations: Optional[int] = None,
        seed: int = 42,
        block_size: int = 1,
    ) -> dict[str, FloatArray]:
        """
        Generate bootstrapped log returns using block bootstrap.

        If block_size=1, this reduces to standard i.i.d. bootstrap.
        Blocks are drawn with replacement and concatenated, then trimmed to n_steps.

        If n_steps or n_simulations are omitted, defaults are taken from params.
        Output shapes:
            equity: (n_steps, n_simulations)
            bond:   (n_steps, n_simulations)
        """
        n_steps = self.params.n_steps if n_steps is None else int(n_steps)
        n_simulations = self.params.n_simulations if n_simulations is None else int(n_simulations)

        if n_steps <= 0:
            raise ValueError("n_steps must be positive.")
        if n_simulations <= 0:
            raise ValueError("n_simulations must be positive.")
        if block_size <= 0:
            raise ValueError("block_size must be positive.")

        t_hist = self.return_matrix.shape[0]
        if t_hist == 0:
            raise ValueError("Historical return matrix is empty.")

        if block_size > t_hist:
            raise ValueError(
                f"block_size ({block_size}) cannot exceed the number of "
                f"historical observations ({t_hist})."
            )

        rng = np.random.default_rng(seed)

        n_valid_starts = t_hist - block_size + 1

        n_blocks_needed = int(np.ceil(n_steps / block_size))

        block_starts = rng.integers(
            low=0,
            high=n_valid_starts,
            size=(n_blocks_needed, n_simulations),
        )

        offsets = np.arange(block_size)

        all_indices = block_starts[:, np.newaxis, :] + offsets[np.newaxis, :, np.newaxis]

        all_indices = all_indices.reshape(n_blocks_needed * block_size, n_simulations)
        all_indices = all_indices[:n_steps, :]  # shape: (n_steps, n_simulations)

        # Gather returns: shape (n_steps, n_simulations, 2)
        bootstrapped = self.return_matrix[all_indices]

        return {
            "equity": bootstrapped[:, :, 0],
            "bond": bootstrapped[:, :, 1],
        }

    def _get_prices(self) -> pd.DataFrame:
        key = (
            self.params.equity_ticker,
            self.params.bond_ticker,
            self.params.start_date,
            self.params.rebalance_freq,
        )

        if key in self._memory_cache:
            return self._memory_cache[key].copy()

        if self.cache_path is not None:
            disk_cache = self._load_disk_cache()
            if key in disk_cache:
                prices = disk_cache[key].copy()
                self._memory_cache[key] = prices
                return prices.copy()

        prices = self._download_prices()
        self._memory_cache[key] = prices.copy()

        if self.cache_path is not None:
            disk_cache = self._load_disk_cache()
            disk_cache[key] = prices.copy()
            self._save_disk_cache(disk_cache)

        return prices.copy()

    def _download_prices(self) -> pd.DataFrame:
        # raw = yf.download(
        #     tickers=[self.params.equity_ticker, self.params.bond_ticker],
        #     start=self.params.start_date,
        #     auto_adjust=True,
        #     progress=False,
        #     actions=False,
        # )

        raw = pd.read_csv(_LOCAL_CSV ,index_col=0, header=[0, 1])
        if raw.empty:
            raise ValueError(
                f"No price data returned for tickers "
                f"{self.params.equity_ticker}, {self.params.bond_ticker} starting from {self.params.start_date}."
            )

        if "Close" not in raw.columns.get_level_values(0):
            raise ValueError("Downloaded data does not contain 'Close' prices.")

        prices = raw["Close"].copy()

        expected_cols = [self.params.equity_ticker, self.params.bond_ticker]
        missing = [col for col in expected_cols if col not in prices.columns]
        if missing:
            raise ValueError(f"Missing close prices for ticker(s): {missing}")

        prices = prices[expected_cols].sort_index()

        if prices.dropna(how="all").empty:
            raise ValueError("Price data is empty after dropping fully missing rows.")

        return prices

    def _compute_log_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        rule = _frequency_mapping[self.params.rebalance_freq]

        if rule is not None:
            prices = prices.resample(rule).last()

        prices = prices.dropna()

        if prices.empty:
            raise ValueError("Price data is empty after alignment/resampling.")

        log_returns = np.log(prices).diff().dropna()

        log_returns.columns = ["equity", "bond"]
        return log_returns.astype(np.float64)

    def _load_disk_cache(self) -> dict[tuple[str, str, str, str], pd.DataFrame]:
        if self.cache_path is None or not self.cache_path.exists():
            return {}

        with self.cache_path.open("rb") as f:
            cache = pickle.load(f)

        if not isinstance(cache, dict):
            raise ValueError("Disk cache is corrupted: expected a dictionary.")

        return cache

    def _save_disk_cache(
        self,
        cache: dict[tuple[str, str, str, str], pd.DataFrame],
    ) -> None:
        if self.cache_path is None:
            return

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("wb") as f:
            pickle.dump(cache, f)