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
        velocity: np.ndarray | list[float],
        position: np.ndarray | list[float],
        blade_width: float,
        edge_thickness: float,
    ) -> None:
        """Initialize the projectile.

        Parameters
        ----------
        mass : float
            Impactor mass (kg).
        velocity : np.ndarray or list of float
            Initial velocity vector, shape ``(3,)``.
        position : np.ndarray or list of float
            Initial center-of-mass position, shape ``(3,)``.
        blade_width : float
            Width of the flat blade face (metres).
        edge_thickness : float
            Thickness of the blade edge (metres).
        """
        self.mass = float(mass)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.position = np.array(position, dtype=np.float64)
        self.blade_width = float(blade_width)
        self.edge_thickness = float(edge_thickness)
        self.contact_nodes = np.array([], dtype=np.int32)


def update_contact_zone(
    projectile: Projectile,
    grid: Grid,
    proximity_threshold: float,
    positions: np.ndarray | None = None,
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
    positions : np.ndarray, optional
        Current node positions, shape ``(n_nodes, 3)``. If None, uses grid.nodes.

    Returns
    -------
    np.ndarray
        Integer array of node indices currently in contact.
    """
    pos = positions if positions is not None else grid.nodes

    x_p = projectile.position[0]
    y_p = projectile.position[1]
    z_p = projectile.position[2]
    w_h = projectile.blade_width / 2.0
    t_h = projectile.edge_thickness / 2.0

    # Project node onto the blade contact patch
    x_proj = np.clip(pos[:, 0], x_p - w_h, x_p + w_h)
    y_proj = np.clip(pos[:, 1], y_p - t_h, y_p + t_h)
    z_proj = z_p

    # Compute 3D distance
    dists = np.sqrt(
        (pos[:, 0] - x_proj) ** 2 + (pos[:, 1] - y_proj) ** 2 + (pos[:, 2] - z_proj) ** 2
    )

    contact_mask = dists <= proximity_threshold
    contact_nodes = np.flatnonzero(contact_mask).astype(np.int32)
    projectile.contact_nodes = contact_nodes
    return contact_nodes


def distribute_contact_forces(
    projectile: Projectile,
    grid: Grid,
    positions: np.ndarray | None = None,
    k_contact: float | None = None,
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
    positions : np.ndarray, optional
        Current node positions, shape ``(grid.n_nodes, 3)``. If None, uses grid.nodes.
    k_contact : float, optional
        Contact penalty stiffness. If None, defaults to 10 * average ortho stiffness.

    Returns
    -------
    np.ndarray
        Contact force on each node, shape ``(grid.n_nodes, 3)``.
    """
    pos = positions if positions is not None else grid.nodes
    forces = np.zeros_like(pos)

    contact_nodes = projectile.contact_nodes
    if len(contact_nodes) == 0:
        return forces

    # Direction of projectile movement
    direction = np.sign(projectile.velocity[2]) if projectile.velocity[2] != 0.0 else 1.0

    c_pos = pos[contact_nodes]
    x_p = projectile.position[0]
    y_p = projectile.position[1]
    z_p = projectile.position[2]
    w_h = projectile.blade_width / 2.0
    t_h = projectile.edge_thickness / 2.0

    # Project contact nodes to compute distance
    x_proj = np.clip(c_pos[:, 0], x_p - w_h, x_p + w_h)
    y_proj = np.clip(c_pos[:, 1], y_p - t_h, y_p + t_h)
    z_proj = z_p

    d_i = np.sqrt(
        (c_pos[:, 0] - x_proj) ** 2 + (c_pos[:, 1] - y_proj) ** 2 + (c_pos[:, 2] - z_proj) ** 2
    )

    # Penetration in direction of motion
    penetration = np.maximum(0.0, (z_p - c_pos[:, 2]) * direction)

    if k_contact is None:
        k_val = 10.0 * np.mean(grid.stiffnesses)
    else:
        k_val = k_contact

    # Inverse-distance weights
    w_i = 1.0 / np.maximum(d_i, 1e-4)
    w_mean = np.mean(w_i) if len(w_i) > 0 else 1.0
    w_normalized = w_i / w_mean if w_mean > 0.0 else np.ones_like(w_i)

    f_i = k_val * w_normalized * penetration

    # Distribute forces along the Z-axis (transverse motion)
    forces_c = np.zeros((len(contact_nodes), 3))
    forces_c[:, 2] = f_i * direction

    forces[contact_nodes] = forces_c
    return forces
