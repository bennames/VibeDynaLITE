"""Projectile model module.

Defines the rigid-body projectile representation and routines for evolving
contact detection and inverse-distance-weighted contact force distribution
onto the spring–mass grid.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from kevlargrid.solver.grid import Grid


class Projectile:
    """Rigid-body projectile interacting with a Kevlar grid.

    Attributes
    ----------
    mass : float
        Projectile mass (kg).
    velocity : np.ndarray
        Current velocity vector, shape ``(3,)``.
    position : np.ndarray
        Current centre-of-mass position, shape ``(3,)``.
    blade_width : float
        Width of the blade / impactor face (metres).
    edge_thickness : float
        Edge thickness of the blade (metres).
    contact_nodes : np.ndarray
        Indices of grid nodes currently in contact.
    """

    mass: float
    velocity: np.ndarray
    position: np.ndarray
    blade_width: float
    edge_thickness: float
    contact_nodes: np.ndarray

    def __init__(
        self,
        mass: float,
        velocity: np.ndarray,
        position: np.ndarray,
        blade_width: float,
        edge_thickness: float,
    ) -> None: ...


def update_contact_zone(
    projectile: Projectile,
    grid: Grid,
    proximity_threshold: float,
) -> np.ndarray:
    """Determine which grid nodes fall within the projectile's contact zone.

    Uses a proximity-based detection that evolves as the projectile
    penetrates the fabric.

    Parameters
    ----------
    projectile : Projectile
        The projectile instance.
    grid : Grid
        The fabric grid.
    proximity_threshold : float
        Distance threshold for contact detection (metres).

    Returns
    -------
    np.ndarray
        Integer array of node indices currently in contact.
    """
    raise NotImplementedError("Stub")


def distribute_contact_forces(
    projectile: Projectile,
    grid: Grid,
) -> np.ndarray:
    """Distribute projectile contact force onto grid nodes.

    Uses inverse-distance weighting so that nodes closer to the projectile
    centre receive proportionally larger forces.

    Parameters
    ----------
    projectile : Projectile
        The projectile instance (contains contact_nodes).
    grid : Grid
        The fabric grid.

    Returns
    -------
    np.ndarray
        Contact force on each node, shape ``(grid.n_nodes, 3)``.
    """
    raise NotImplementedError("Stub")
