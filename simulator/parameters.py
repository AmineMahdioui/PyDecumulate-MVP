"""Strategy parameter container."""

from dataclasses import dataclass

import numpy as np

# Mapping from human-readable frequency labels to steps per year.
FREQ_MAP: dict[str, int] = {
    "daily": 252,
    "weekly": 52,
    "monthly": 12,
    "quarterly": 4,
    "yearly": 1,
}


@dataclass(frozen=True)
class StrategyParameters:
    """Immutable container for all CPPI strategy inputs.

    All percentage fields are expressed as whole numbers
    (e.g. ``7.0`` means 7 %).
    """

    initial_wealth: float
    time_horizon: int  # years
    expected_return: float  # annual, e.g. 7.0
    market_volatility: float  # annual, e.g. 15.0
    risk_free_rate: float = 2.0  # annual, e.g. 2.0
    n_simulations: int = 1000
    rebalance_freq: str = "monthly"  # key in FREQ_MAP
    annual_withdrawal: float = 0.0  # absolute amount withdrawn per year (decumulation)
    annual_contribution: float = 0.0  # absolute amount contributed per year (accumulation)
    annual_inflation_rate: float = 0.0  # annual inflation rate, e.g. 2.0 for 2%
    cppi_multiplier: float = 3.0  # m in CPPI formula
    floor_pct: float = 80.0  # e.g. 80 → 80 %
    Lambda: float = 60.0  # target risky asset allocation for Constant Mix (e.g. 60.0 → 60%)
    # -- derived helpers -----------------------------------------------------

    @property
    def step_withdrawal(self) -> float:
        """Amount withdrawn at each step, derived from annual withdrawal."""
        if self.annual_withdrawal > 0.0:
            return self.annual_withdrawal / self.steps_per_year
        return 0.0

    @property
    def step_contribution(self) -> float:
        """Amount contributed at each step, derived from annual contribution."""
        if self.annual_contribution > 0.0:
            return self.annual_contribution / self.steps_per_year
        return 0.0

    @property
    def is_decumulation(self) -> bool:
        """True when running in decumulation (withdrawal) mode."""
        return self.annual_withdrawal > 0.0
    
    
    @property
    def steps_per_year(self) -> int:
        """Number of rebalancing steps in one year."""
        try:
            return FREQ_MAP[self.rebalance_freq]
        except KeyError:
            raise ValueError(
                f"Unknown rebalance_freq '{self.rebalance_freq}'. "
                f"Choose from {list(FREQ_MAP)}."
            )

    @property
    def n_steps(self) -> int:
        """Total number of time-steps over the full horizon."""
        return self.time_horizon * self.steps_per_year

    @property
    def dt(self) -> float:
        """Length of one time-step expressed in years."""
        return 1.0 / self.steps_per_year

    @property
    def step_mu(self) -> float:
        """Expected return of the risky asset per time-step."""
        return self.expected_return / 100.0 / self.steps_per_year

    @property
    def step_sigma(self) -> float:
        """Volatility of the risky asset per time-step."""
        return self.market_volatility / 100.0 / np.sqrt(self.steps_per_year)

    @property
    def step_rf(self) -> float:
        """Risk-free rate per time-step."""
        return self.risk_free_rate / 100.0 / self.steps_per_year

    @property
    def step_inflation(self) -> float:
        """Inflation rate per time-step, compounded correctly."""
        return (1.0 + self.annual_inflation_rate / 100.0) ** self.dt - 1.0

    @property
    def floor_value(self) -> float:
        """Initial guaranteed floor in absolute terms."""
        return self.initial_wealth * self.floor_pct / 100.0

@dataclass(frozen=True)
class LifecycleParameters:
    """Immutable container for strategy inputs."""
    
    current_age: int = 28 # unused in current simulations but could be used for age-based floor or contribution adjustments
    starting_pot: float = 50_000  # (e.g. 50_000)
    # annual_contribution: float # (e.g. 10_000)
    # inflation_rate: float # (e.g. 2.5, in %)
    time_horizon: int = 43   # years
    retirement_age: int = 64 
    target_pot: float = 500_000 # 
    equity_ticker ="^STOXX"  #Index for now used to be "MEUD.PA"   # Amundi STOXX Europe 600 ETF
    bond_ticker = "EGRI.PA"   # Amundi Euro Aggregate Bond ETF
    n_simulations: int = 1_000
    
    rebalance_freq: str = "monthly"
    start_date: str = "2001-01-01"
    annual_withdrawal: float = 40_000.0 # for decumulation phase, or to be derive from target income
    cppi_multiplier: float = 3.0
    floor_pct: float = 80.0
    risk_free_rate: float = 2.0
    
    @property
    def n_steps(self) -> int:
        """Total number of time-steps over the full horizon."""
        return self.time_horizon * self.steps_per_year
    @property
    def steps_per_year(self) -> int:
        """Number of rebalancing steps in one year."""
        try:
            return FREQ_MAP[self.rebalance_freq]
        except KeyError:
            raise ValueError(
                f"Unknown rebalance_freq '{self.rebalance_freq}'. "
                f"Choose from {list(FREQ_MAP)}."
            )



@dataclass(frozen=True)
class GlidepathParameters:
    """Immutable container for strategy inputs."""
    
    initial_equity_allocation: float = 80.0
    final_equity_allocation: float = 40.0
    years: int = 43
    frequency: str = "monthly" # how often to adjust the allocation, e.g. "monthly", "yearly", etc.
    
    @property
    def n_steps(self) -> int:
        """Total number of time-steps over the full horizon."""
        return self.years * self.steps_per_year
    @property
    def steps_per_year(self) -> int:
        """Number of rebalancing steps in one year."""
        try:
            return FREQ_MAP[self.frequency]
        except KeyError:
            raise ValueError(
                f"Unknown frequency '{self.frequency}'. "
                f"Choose from {list(FREQ_MAP)}."
            )
            
