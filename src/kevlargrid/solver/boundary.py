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


def apply_impedance_boundary(
    forces: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    springs: np.ndarray,
    stiffnesses: np.ndarray,
    damage: np.ndarray,
    failed: np.ndarray,
    boundary_mask: np.ndarray,
) -> np.ndarray:
    """Apply dynamic boundary dashpots matching local acoustic impedance.

    Parameters
    ----------
    forces : np.ndarray
        Node forces array, shape ``(n_nodes, 3)``. Modified in-place.
    velocities : np.ndarray
        Node velocities array, shape ``(n_nodes, 3)``.
    masses : np.ndarray
        Node masses array, shape ``(n_nodes,)``.
    springs : np.ndarray
        Spring connectivity array, shape ``(n_springs, 2)``.
    stiffnesses : np.ndarray
        Axial stiffnesses array, shape ``(n_springs,)``.
    damage : np.ndarray
        Spring damage array, shape ``(n_springs,)``.
    failed : np.ndarray
        Spring failed boolean array, shape ``(n_springs,)``.
    boundary_mask : np.ndarray
        Integer boundary mask array, shape ``(n_nodes,)`` where 2 represents non-reflecting.

    Returns
    -------
    np.ndarray
        Updated forces array.
    """
    idx_boundary = np.where(boundary_mask == 2)[0]
    if len(idx_boundary) == 0:
        return forces

    # Calculate effective spring stiffnesses
    effective_k = stiffnesses * (1.0 - damage)
    effective_k[failed] = 0.0

    # Accumulate stiffness on boundary nodes
    nodal_stiffness = np.zeros(len(forces), dtype=forces.dtype)
    np.add.at(nodal_stiffness, springs[:, 0], effective_k)
    np.add.at(nodal_stiffness, springs[:, 1], effective_k)

    # Compute impedance damping coefficient C_i = sqrt(m_i * k_eff)
    k_eff = np.maximum(nodal_stiffness[idx_boundary], 1e-4)
    C_i = np.sqrt(masses[idx_boundary] * k_eff)

    # Apply forces: F = -C_i * v_i
    forces[idx_boundary] -= C_i[:, np.newaxis] * velocities[idx_boundary]
    return forces


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
