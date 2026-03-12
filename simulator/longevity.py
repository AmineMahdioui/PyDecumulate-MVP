"""
standard mortality table (like the French INSEE tables or a Gompertz-Makeham mathematical model).
This will give each simulated path a "Date of Death" rather than a fixed time_horizon, allowing 
us to calculate if the client outlived their money.
"""

# TODO: Implement longevity 