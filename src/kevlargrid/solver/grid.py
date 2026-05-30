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
    rest_lengths : np.ndarray
        Natural rest length per spring, shape ``(n_springs,)``.
    failed : np.ndarray
        Boolean failure flag per spring, shape ``(n_springs,)``.
    tension_only : np.ndarray
        Boolean flag marking orthogonal springs, shape ``(n_springs,)``.
    n_nodes : int
        Total number of nodes in the grid.
    n_springs : int
        Total number of springs (edges) in the grid.
    """

    nodes: np.ndarray
    springs: np.ndarray
    masses: np.ndarray
    stiffnesses: np.ndarray
    rest_lengths: np.ndarray
    failed: np.ndarray
    tension_only: np.ndarray
    n_nodes: int
    n_springs: int

    def __init__(
        self,
        nodes: np.ndarray,
        springs: np.ndarray,
        masses: np.ndarray,
        stiffnesses: np.ndarray,
        rest_lengths: np.ndarray,
        failed: np.ndarray,
        tension_only: np.ndarray,
    ) -> None:
        self.nodes = nodes
        self.springs = springs
        self.masses = masses
        self.stiffnesses = stiffnesses
        self.rest_lengths = rest_lengths
        self.failed = failed
        self.tension_only = tension_only
        self.n_nodes = len(nodes)
        self.n_springs = len(springs)


def generate_rectangular_grid(
    nx: int,
    ny: int,
    dx: float,
    material: dict,
    n_plies: int = 1,
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
    n_plies : int
        Number of plies (acts as scalar multiplier for mass/stiffness in Mode A).

    Returns
    -------
    Grid
        A fully initialised :class:`Grid` instance.
    """
    # 1. Generate node coordinates
    x = np.arange(nx) * dx
    y = np.arange(ny) * dx
    # Center grid around (0, 0)
    x = x - (nx - 1) * dx / 2.0
    y = y - (ny - 1) * dx / 2.0

    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")
    nodes = np.stack([x_grid, y_grid, np.zeros_like(x_grid)], axis=-1).reshape(-1, 3)

    # 2. Lump masses by tributary area
    areal_density = material.get("areal_density_kgm2", 0.47)
    m_cell = n_plies * areal_density * dx * dx

    masses = np.zeros(nx * ny, dtype=np.float64)
    for i in range(nx):
        for j in range(ny):
            idx = i * ny + j
            is_x_boundary = i == 0 or i == nx - 1
            is_y_boundary = j == 0 or j == ny - 1
            if is_x_boundary and is_y_boundary:
                masses[idx] = 0.25 * m_cell
            elif is_x_boundary or is_y_boundary:
                masses[idx] = 0.5 * m_cell
            else:
                masses[idx] = m_cell

    # 3. Connect springs
    springs_list = []
    tension_only_list = []

    for i in range(nx):
        for j in range(ny):
            idx = i * ny + j
            # Orthogonal: warp (x)
            if i < nx - 1:
                idx_next = (i + 1) * ny + j
                springs_list.append((idx, idx_next))
                tension_only_list.append(True)
            # Orthogonal: weft (y)
            if j < ny - 1:
                idx_next = i * ny + (j + 1)
                springs_list.append((idx, idx_next))
                tension_only_list.append(True)
            # Diagonal: +45 deg
            if i < nx - 1 and j < ny - 1:
                idx_diag1 = (i + 1) * ny + (j + 1)
                springs_list.append((idx, idx_diag1))
                tension_only_list.append(False)
            # Diagonal: -45 deg
            if i < nx - 1 and j > 0:
                idx_diag2 = (i + 1) * ny + (j - 1)
                springs_list.append((idx, idx_diag2))
                tension_only_list.append(False)

    springs = np.array(springs_list, dtype=np.int32)
    tension_only = np.array(tension_only_list, dtype=bool)

    # 4. Calculate spring stiffnesses
    tensile_modulus_gpa = material.get("tensile_modulus_gpa", 71.0)
    fiber_density_gcc = material.get("fiber_density_gcc", 1.44)
    shear_ratio = material.get("shear_ratio", 0.0004)

    # Thickness t = areal_density / density
    t = areal_density / (fiber_density_gcc * 1000.0)
    e_mod = tensile_modulus_gpa * 1e9

    k_ortho = n_plies * e_mod * t
    k_shear = k_ortho * shear_ratio

    stiffnesses = np.where(tension_only, k_ortho, k_shear)
    rest_lengths = np.where(tension_only, dx, np.sqrt(2.0) * dx)
    failed = np.zeros(len(springs), dtype=bool)

    return Grid(
        nodes=nodes,
        springs=springs,
        masses=masses,
        stiffnesses=stiffnesses,
        rest_lengths=rest_lengths,
        failed=failed,
        tension_only=tension_only,
    )
