"""Boundary conditions module.

Provides routines for enforcing clamped (zero-velocity) boundary conditions
and for auto-sizing the grid radius so that stress waves do not reach the
boundary during the simulation window.
"""

from __future__ import annotations

import numpy as np


def apply_clamped_boundary(
    velocities: np.ndarray,
    boundary_mask: np.ndarray,
) -> np.ndarray:
    """Enforce zero velocity on clamped boundary nodes.

    Parameters
    ----------
    velocities : np.ndarray
        Node velocities, shape ``(n_nodes, 3)``.
    boundary_mask : np.ndarray
        Boolean mask identifying boundary nodes, shape ``(n_nodes,)``.

    Returns
    -------
    np.ndarray
        Updated velocities with boundary nodes zeroed, shape ``(n_nodes, 3)``.
    """
    vel_new = velocities.copy()
    vel_new[boundary_mask] = 0.0
    return vel_new


def compute_min_radius(
    wave_speed: float,
    sim_duration: float,
    safety_factor: float,
) -> float:
    """Compute the minimum grid radius for an "infinite" boundary.

    The grid must be large enough that the fastest stress wave cannot
    reach the boundary within the simulation duration.

    Parameters
    ----------
    wave_speed : float
        Maximum transverse wave speed in the fabric (m/s).
    sim_duration : float
        Total simulated time (seconds).
    safety_factor : float
        Multiplicative safety factor (≥1.0).

    Returns
    -------
    float
        Minimum grid half-width (metres).
    """
    return float(wave_speed * sim_duration * safety_factor)
