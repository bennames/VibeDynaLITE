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
    youngs_modulus_gpa: float | None = None,
    thickness: float | None = None,
    density_kgm3: float | None = None,
    poisson_ratio: float = 0.3,
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
    youngs_modulus_gpa : float, optional
        Young's modulus in GPa (for shell bending CFL limit).
    thickness : float, optional
        Shell element thickness in metres.
    density_kgm3 : float, optional
        Density in kg/m^3.
    poisson_ratio : float, optional
        Poisson's ratio (defaults to 0.3).

    Returns
    -------
    float
        Stable time-step size (seconds).
    """
    if cfl <= 0.0 or cfl > 1.0:
        raise ValueError("CFL safety factor must be in the range (0, 1].")

    m_min = np.min(masses)
    k_max = np.max(stiffnesses)

    # dt_crit = sqrt(m / k) representing physical wave traversal limit (membrane-only)
    dt_crit = np.sqrt(m_min / k_max)

    # Account for shell bending limits if thin shell parameters are present
    if (
        youngs_modulus_gpa is not None
        and thickness is not None
        and density_kgm3 is not None
        and thickness > 0.0
    ):
        E = youngs_modulus_gpa * 1.0e9
        term1 = (dx * dx) / (thickness * np.sqrt(3.0))
        term2 = np.sqrt(density_kgm3 * (1.0 - poisson_ratio * poisson_ratio) / E)
        dt_bend = term1 * term2
        dt_crit = min(dt_crit, dt_bend)

    return float(cfl * dt_crit)
