import numpy as np
from numpy import ndarray


def deflate(nominal_values: ndarray, inflation_rate: float, dt: float, start_step: int = 0) -> ndarray:
    """
    Deflate nominal values to real values using a constant inflation rate.

    Args:
        nominal_values: Array of nominal values. First axis is time (t_steps);
                        can be 1-D (t_steps,) or N-D (t_steps, ...).
        inflation_rate: Annual inflation rate in percentage (e.g., 2.0 for 2%)
        dt: Time step length in years (e.g., 1/12 for monthly)
        start_step: Index of the first time-step represented in nominal_values
                    (default 0). Pass 1 when the array starts at t=1 rather
                    than t=0, as is the case for post-simulation output arrays.

    Returns:
        Array of real values with the same shape as nominal_values.
    """
    # Convert annual inflation rate to per-step factor
    per_step_inflation_factor = (1 + inflation_rate / 100) ** dt

    # Build cumulative factors for the relevant time window
    steps = np.arange(start_step, start_step + nominal_values.shape[0])
    cumulative_inflation_factors = per_step_inflation_factor ** steps

    # Reshape to broadcast correctly against N-D arrays (time is axis 0)
    shape = (-1,) + (1,) * (nominal_values.ndim - 1)
    cumulative_inflation_factors = cumulative_inflation_factors.reshape(shape)

    return nominal_values / cumulative_inflation_factors

