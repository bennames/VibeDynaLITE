"""Grid generation module.

Provides the :class:`Grid` data structure that stores the lumped-mass spring
network and a factory function for building regular rectangular grids with
orthogonal and diagonal spring connectivity.
"""

from __future__ import annotations

import numpy as np


class Grid:
    """Lumped-mass spring–network representation of a woven fabric panel.

    Attributes
    ----------
    nodes : np.ndarray
        Node position array of shape ``(n_nodes, 3)``.
    springs : np.ndarray
        Spring connectivity array of shape ``(n_springs, 2)`` — each row is
        a pair of node indices.
    masses : np.ndarray
        Lumped mass for every node, shape ``(n_nodes,)``.
    stiffnesses : np.ndarray
        Axial stiffness for every spring, shape ``(n_springs,)``.
    n_nodes : int
        Total number of nodes in the grid.
    n_springs : int
        Total number of springs (edges) in the grid.
    """

    nodes: np.ndarray
    springs: np.ndarray
    masses: np.ndarray
    stiffnesses: np.ndarray
    n_nodes: int
    n_springs: int

    def __init__(
        self,
        nodes: np.ndarray,
        springs: np.ndarray,
        masses: np.ndarray,
        stiffnesses: np.ndarray,
    ) -> None: ...


def generate_rectangular_grid(
    nx: int,
    ny: int,
    dx: float,
    material: dict,
) -> Grid:
    """Create a rectangular grid with orthogonal and diagonal springs.

    Parameters
    ----------
    nx : int
        Number of nodes along the x-axis.
    ny : int
        Number of nodes along the y-axis.
    dx : float
        Spacing between adjacent nodes (metres).
    material : dict
        Material property dictionary (see :mod:`kevlargrid.materials.library`).

    Returns
    -------
    Grid
        A fully initialised :class:`Grid` instance.
    """
    raise NotImplementedError("Stub")
