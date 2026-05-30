"""Damping models module.

Implements viscous (velocity-proportional) and Rayleigh (mass + stiffness
proportional) damping force computations for the explicit solver.
"""

from __future__ import annotations

import numpy as np


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
    raise NotImplementedError("Stub")


def rayleigh_damping(
    velocities: np.ndarray,
    masses: np.ndarray,
    stiffnesses: np.ndarray,
    alpha: float,
    beta: float,
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

    Returns
    -------
    np.ndarray
        Damping force on each node, shape ``(n_nodes, 3)``.
    """
    raise NotImplementedError("Stub")
