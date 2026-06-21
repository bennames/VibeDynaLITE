"""CFL timestep computation module.

Computes the maximum stable explicit time-step based on the
Courant–Friedrichs–Lewy (CFL) condition for the spring–mass network.
"""

from __future__ import annotations

import numpy as np


def compute_cfl_timestep(
    stiffnesses: np.ndarray,
    masses: np.ndarray,
    dx: float,
    cfl: float,
) -> float:
    """Compute the CFL-limited stable time-step.

    The critical time-step is derived from the maximum natural frequency
    of any spring element, scaled by the user-specified CFL number.

    Parameters
    ----------
    stiffnesses : np.ndarray
        Axial stiffness per spring, shape ``(n_springs,)``.
    masses : np.ndarray
        Lumped mass per node, shape ``(n_nodes,)``.
    dx : float
        Characteristic element length (metres).
    cfl : float
        CFL safety factor (typically 0.5–0.9).

    Returns
    -------
    float
        Stable time-step size (seconds).
    """
    if cfl <= 0.0 or cfl > 1.0:
        raise ValueError("CFL safety factor must be in the range (0, 1].")

    m_min = np.min(masses)
    k_max = np.max(stiffnesses)

    # dt_crit = sqrt(m / k) representing physical wave traversal limit
    dt_crit = np.sqrt(m_min / k_max)

    return float(cfl * dt_crit)
