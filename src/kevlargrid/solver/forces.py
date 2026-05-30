"""Spring force computation module.

Provides vectorised routines for computing internal spring forces and
per-spring strain values using 3-D Euclidean distances.
"""

from __future__ import annotations

import numpy as np


def compute_spring_forces(
    positions: np.ndarray,
    springs: np.ndarray,
    stiffnesses: np.ndarray,
    rest_lengths: np.ndarray,
    failed: np.ndarray,
) -> np.ndarray:
    """Compute the net nodal force vector from all active springs.

    Uses vectorised 3-D Euclidean distance calculations.  Springs whose
    corresponding entry in *failed* is ``True`` contribute zero force.

    Parameters
    ----------
    positions : np.ndarray
        Current node positions, shape ``(n_nodes, 3)``.
    springs : np.ndarray
        Spring connectivity, shape ``(n_springs, 2)``.
    stiffnesses : np.ndarray
        Axial stiffness per spring, shape ``(n_springs,)``.
    rest_lengths : np.ndarray
        Natural (rest) length per spring, shape ``(n_springs,)``.
    failed : np.ndarray
        Boolean failure flags per spring, shape ``(n_springs,)``.

    Returns
    -------
    np.ndarray
        Net force on each node, shape ``(n_nodes, 3)``.
    """
    raise NotImplementedError("Stub")


def compute_spring_strains(
    positions: np.ndarray,
    springs: np.ndarray,
    rest_lengths: np.ndarray,
) -> np.ndarray:
    """Compute the engineering strain for every spring.

    Strain is defined as ``(L - L0) / L0`` where *L* is the current
    deformed length and *L0* is the rest length.

    Parameters
    ----------
    positions : np.ndarray
        Current node positions, shape ``(n_nodes, 3)``.
    springs : np.ndarray
        Spring connectivity, shape ``(n_springs, 2)``.
    rest_lengths : np.ndarray
        Natural (rest) length per spring, shape ``(n_springs,)``.

    Returns
    -------
    np.ndarray
        Strain per spring, shape ``(n_springs,)``.
    """
    raise NotImplementedError("Stub")
