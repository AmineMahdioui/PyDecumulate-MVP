# PyDecumulate (MVP)

A Streamlit-based Monte Carlo simulator for retirement decumulation planning.

## Motivation
As pension systems across Europe and France specifically shift from Defined Benefit (DB) to Defined Contribution (DC) models, individuals increasingly bear market, inflation, and sequencing risks. PyDecumulate is a proof-of-concept built to evaluate the probability of success for different retirement withdrawal strategies.

## Core Features
* **Lifecycle Modeling:** Connects the accumulation and decumulation phases, capturing path-dependent terminal wealth at the point of retirement.
* **Strategy Evaluation:** Provides a framework to test and compare dynamic, static, and deterministic de-risking trajectories.
* **Simulation Engine:** Uses Numba JIT-compiled Monte Carlo loops to execute pathwise scenarios.
## Planned Features
* **Purchasing Power:** Tracks real and nominal wealth to account for inflation over long horizons.
