"""Damping models module.

Implements viscous (velocity-proportional) and Rayleigh (mass + stiffness
proportional) damping force computations for the explicit solver.
"""

from __future__ import annotations

import numpy as np

from kevlargrid.solver import backend


@backend.jit
def viscous_damping(
    velocities: np.ndarray,
    coefficient: float,
) -> np.ndarray:
    """Compute simple velocity-proportional viscous damping forces.

    Parameters
    ----------
    velocities : np.ndarray
        Node velocities, shape ``(n_nodes, 3)``.
    coefficient : float
        Damping coefficient (N·s/m per node).

    Returns
    -------
    np.ndarray
        Damping force on each node, shape ``(n_nodes, 3)``.
    """
    return -coefficient * velocities


@backend.jit
def rayleigh_damping(
    velocities: np.ndarray,
    masses: np.ndarray,
    stiffnesses: np.ndarray,
    alpha: float,
    beta: float,
    springs: np.ndarray | None = None,
    positions: np.ndarray | None = None,
) -> np.ndarray:
    """Compute Rayleigh damping forces (mass + stiffness proportional).

    The Rayleigh damping matrix is ``C = α·M + β·K``.  In the lumped-mass
    explicit context this is evaluated per-node / per-spring without
    assembling global matrices.

    Parameters
    ----------
    velocities : np.ndarray
        Node velocities, shape ``(n_nodes, 3)``.
    masses : np.ndarray
        Lumped mass per node, shape ``(n_nodes,)``.
    stiffnesses : np.ndarray
        Axial stiffness per spring, shape ``(n_springs,)``.
    alpha : float
        Mass-proportional Rayleigh coefficient.
    beta : float
        Stiffness-proportional Rayleigh coefficient.
    springs : np.ndarray, optional
        Spring connectivity, shape ``(n_springs, 2)``.
    positions : np.ndarray, optional
        Node positions, shape ``(n_nodes, 3)``.

    Returns
    -------
    np.ndarray
        Damping force on each node, shape ``(n_nodes, 3)``.
    """
    masses_col = masses.reshape(-1, 1) if masses.ndim == 1 else masses

    # Mass-proportional damping: F_m = -alpha * M * v
    f_mass = -alpha * masses_col * velocities

    # Stiffness-proportional damping: F_k = -beta * K * v
    f_stiff = np.zeros_like(velocities)
    if beta > 0.0 and springs is not None and positions is not None:
        p1 = positions[springs[:, 0]]
        p2 = positions[springs[:, 1]]
        diff = p2 - p1
        lengths = np.sqrt(np.sum(diff**2, axis=1))
        lengths_safe = np.where(lengths == 0.0, 1.0, lengths)

        v1 = velocities[springs[:, 0]]
        v2 = velocities[springs[:, 1]]
        v_rel = v2 - v1

        # Projected relative velocity onto spring axis
        unit_axes = diff / lengths_safe[:, np.newaxis]
        v_proj = np.sum(v_rel * unit_axes, axis=1)

        # Force magnitude and vector
        damp_mag = beta * stiffnesses * v_proj
        damp_vecs = damp_mag[:, np.newaxis] * unit_axes

        # Accumulate damping forces
        for idx in range(len(springs)):
            u = springs[idx, 0]
            v = springs[idx, 1]
            d_vec = damp_vecs[idx]
            f_stiff[u] += d_vec
            f_stiff[v] -= d_vec

    return f_mass + f_stiff  # type: ignore[no-any-return]
