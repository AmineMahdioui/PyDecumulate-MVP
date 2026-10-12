PyDecumulate is an interactive Monte Carlo simulation tool for designing retirement lifecycle strategies under realistic market conditions. It models four allocation approaches — CPPI (accumulation only), glidepath, constant mix, and age-in-bonds — across both accumulation and decumulation phases. You can adjust parameters and see in real time how withdrawals and asset allocation affect your projected income sustainability and the risk of running out of money before death. The returns are simulated across thousands of paths, allowing you to visualize the full distribution of outcomes and stress-test strategies conditional on actual simulated market trajectories. The goal is to make retirement strategy design tangible for someone building or presenting a retirement mandate.
### PyDecumulate — Project Pitch (Interview-Ready)

PyDecumulate is a prototype I developed to simulate and stress-test retirement lifecycle strategies across accumulation and decumulation phases.

The core idea is to model how a portfolio behaves under realistic market conditions using a Monte Carlo framework that supports multiple allocation strategies and enables path-dependent lifecycle analysis—preserving full trajectory information from accumulation through decumulation.

One of the key design choices I made was to preserve full path-by-path information during the accumulation phase and feed it directly into the decumulation engine. This allows me to evaluate how a given withdrawal strategy performs conditional on actual simulated market trajectories, rather than relying only on averaged outcomes.

The goal of the project is not just performance optimization, but building a framework that is:
- transparent in its assumptions  
- robust to different market regimes  
- rigorous in path-conditional lifecycle analysis

At this stage, it's an MVP. The architecture uses clean abstractions that support multiple allocation strategies and lifecycle transitions, providing a foundation for future extensions.

---

### PyDecumulate — 30s Version

I built PyDecumulate to simulate retirement lifecycle strategies using Monte Carlo methods. 

It models portfolio behavior under realistic market scenarios with multiple allocation approaches (glidepath, constant mix, age-in-bonds), and a key feature is that it preserves full path-by-path information from accumulation and feeds it into decumulation to stress-test outcomes realistically.

It's an MVP designed with clean abstractions for lifecycle-consistent strategy analysis.

---

### PyDecumulate — 1-line Version

A Monte Carlo-based framework to simulate and stress-test retirement decumulation strategies with lifecycle-consistent path dependency.
