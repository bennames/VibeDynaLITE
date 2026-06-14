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
    initial_spring_counts: np.ndarray
    node_spring_offsets: np.ndarray
    node_spring_ids: np.ndarray
    node_spring_signs: np.ndarray

    def __init__(
        self,
        nodes: np.ndarray,
        springs: np.ndarray,
        masses: np.ndarray,
        stiffnesses: np.ndarray,
        rest_lengths: np.ndarray,
        failed: np.ndarray,
        tension_only: np.ndarray,
        initial_spring_counts: np.ndarray | None = None,
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

        # Count how many springs connect to each node
        node_counts = np.zeros(self.n_nodes, dtype=np.int32)
        if self.n_springs > 0:
            np.add.at(node_counts, springs[:, 0], 1)
            np.add.at(node_counts, springs[:, 1], 1)

        if initial_spring_counts is None:
            self.initial_spring_counts = node_counts
        else:
            self.initial_spring_counts = initial_spring_counts

        # Build CSR node-to-spring adjacency
        node_spring_offsets = np.zeros(self.n_nodes + 1, dtype=np.int32)
        node_spring_offsets[1:] = np.cumsum(node_counts)

        current_offset = node_spring_offsets[:-1].copy()
        node_spring_ids = np.zeros(2 * self.n_springs, dtype=np.int32)
        node_spring_signs = np.zeros(2 * self.n_springs, dtype=np.float64)

        for j in range(self.n_springs):
            n0 = springs[j, 0]
            n1 = springs[j, 1]

            offset_0 = current_offset[n0]
            node_spring_ids[offset_0] = j
            node_spring_signs[offset_0] = 1.0
            current_offset[n0] += 1

            offset_1 = current_offset[n1]
            node_spring_ids[offset_1] = j
            node_spring_signs[offset_1] = -1.0
            current_offset[n1] += 1

        self.node_spring_offsets = node_spring_offsets
        self.node_spring_ids = node_spring_ids
        self.node_spring_signs = node_spring_signs


def generate_rectangular_grid(
    nx: int,
    ny: int,
    dx: float,
    material: dict,
    n_plies: int = 1,
    t_ply: float | None = None,
) -> Grid:
    """Create a rectangular grid with orthogonal and diagonal springs.

    Supports both Sizing Mode (Mode A: single layer with scaled mass/stiffness)
    and Checkout Mode (Mode B: physical stacked layers spaced along the Z axis).

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
        Number of plies (acts as scalar multiplier in Mode A, or number of stacks in Mode B).
    t_ply : float, optional
        Vertical spacing between discrete layers (metres). If provided, enables Mode B.

    Returns
    -------
    Grid
        A fully initialised :class:`Grid` instance.
    """
    # 1. Base properties (single layer)
    x = np.arange(nx) * dx
    y = np.arange(ny) * dx
    # Center grid around (0, 0)
    x = x - (nx - 1) * dx / 2.0
    y = y - (ny - 1) * dx / 2.0

    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")
    base_nodes = np.stack([x_grid, y_grid, np.zeros_like(x_grid)], axis=-1).reshape(-1, 3)
    n_nodes_per_layer = nx * ny

    # 2. Lump base masses by tributary area
    areal_density = material.get("areal_density_kgm2", 0.47)

    # In Mode B, each layer has standard mass. In Mode A, we scale a single layer.
    is_mode_b = t_ply is not None and n_plies > 1
    m_scale = 1.0 if is_mode_b else float(n_plies)
    m_cell = m_scale * areal_density * dx * dx

    base_masses = np.zeros(n_nodes_per_layer, dtype=np.float64)
    for i in range(nx):
        for j in range(ny):
            idx = i * ny + j
            is_x_boundary = i == 0 or i == nx - 1
            is_y_boundary = j == 0 or j == ny - 1
            if is_x_boundary and is_y_boundary:
                base_masses[idx] = 0.25 * m_cell
            elif is_x_boundary or is_y_boundary:
                base_masses[idx] = 0.5 * m_cell
            else:
                base_masses[idx] = m_cell

    # 3. Connect base springs
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

    base_springs = np.array(springs_list, dtype=np.int32)
    base_tension_only = np.array(tension_only_list, dtype=bool)

    # 4. Calculate spring stiffnesses
    tensile_modulus_gpa = material.get("tensile_modulus_gpa", 71.0)
    fiber_density_gcc = material.get("fiber_density_gcc", 1.44)
    shear_ratio = material.get("shear_ratio", 0.0004)

    # Thickness t = areal_density / density
    t = areal_density / (fiber_density_gcc * 1000.0)
    e_mod = tensile_modulus_gpa * 1e9

    k_scale = 1.0 if is_mode_b else float(n_plies)
    k_ortho = k_scale * e_mod * t
    k_shear = k_ortho * shear_ratio

    base_stiffnesses = np.where(base_tension_only, k_ortho, k_shear)
    base_rest_lengths = np.where(base_tension_only, dx, np.sqrt(2.0) * dx)

    # 5. Stacking if Mode B
    if is_mode_b:
        assert t_ply is not None
        all_nodes = []
        all_springs = []
        all_masses = []
        all_stiffnesses = []
        all_rest_lengths = []
        all_tension_only = []

        for ply in range(n_plies):
            ply_nodes = base_nodes.copy()
            # Stack along Z-axis
            ply_nodes[:, 2] = ply * t_ply
            all_nodes.append(ply_nodes)

            # Offset spring connectivity indices to target correct nodes in this layer
            offset = ply * n_nodes_per_layer
            ply_springs = base_springs + offset
            all_springs.append(ply_springs)

            # Duplicate stiffnesses and mass profiles
            all_masses.append(base_masses)
            all_stiffnesses.append(base_stiffnesses)
            all_rest_lengths.append(base_rest_lengths)
            all_tension_only.append(base_tension_only)

        nodes = np.concatenate(all_nodes, axis=0)
        springs = np.concatenate(all_springs, axis=0)
        masses = np.concatenate(all_masses, axis=0)
        stiffnesses = np.concatenate(all_stiffnesses, axis=0)
        rest_lengths = np.concatenate(all_rest_lengths, axis=0)
        tension_only = np.concatenate(all_tension_only, axis=0)
    else:
        nodes = base_nodes
        springs = base_springs
        masses = base_masses
        stiffnesses = base_stiffnesses
        rest_lengths = base_rest_lengths
        tension_only = base_tension_only

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
