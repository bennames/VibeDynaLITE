"""Spring force computation module.

Provides vectorised routines for computing internal spring forces and
per-spring strain values using 3-D Euclidean distances.
"""

from __future__ import annotations

import numpy as np

from kevlargrid.solver import backend
from kevlargrid.solver.backend import (
    maximum,
    minimum,
    min,
    scatter_add,
    sqrt,
    stack_z,
    sum,
    where,
    zeros,
)


try:
    import numba
except ImportError:
    numba = None


@backend.jit(parallel=True, fastmath=True)
def numba_gather_spring_forces(
    positions: np.ndarray,
    springs: np.ndarray,
    stiffnesses: np.ndarray,
    rest_lengths: np.ndarray,
    failed: np.ndarray,
    tension_only: np.ndarray,
    node_spring_offsets: np.ndarray,
    node_spring_ids: np.ndarray,
    node_spring_signs: np.ndarray,
    damage_onset_strain: float,
    failure_strain: float,
) -> np.ndarray:
    n_springs = len(springs)
    n_nodes = len(positions)
    force_vecs = np.zeros((n_springs, 3), dtype=positions.dtype)

    for i in numba.prange(n_springs):
        n0 = springs[i, 0]
        n1 = springs[i, 1]

        dx = positions[n1, 0] - positions[n0, 0]
        dy = positions[n1, 1] - positions[n0, 1]
        dz = positions[n1, 2] - positions[n0, 2]

        length = np.sqrt(dx*dx + dy*dy + dz*dz)
        length_safe = length if length != 0.0 else 1.0

        strain = (length - rest_lengths[i]) / rest_lengths[i]

        # Progressive damage model
        denom = failure_strain - damage_onset_strain
        denom_safe = denom if denom != 0.0 else 1.0
        val = (strain - damage_onset_strain) / denom_safe
        damage = 0.0
        if val > 0.0:
            damage = val if val < 1.0 else 1.0

        effective_k = stiffnesses[i] * (1.0 - damage)
        f_mag = effective_k * strain * rest_lengths[i]

        # Tension-only check
        if tension_only[i] and strain < 0.0:
            f_mag = 0.0

        if failed[i]:
            f_mag = 0.0

        f_coeff = f_mag / length_safe
        force_vecs[i, 0] = f_coeff * dx
        force_vecs[i, 1] = f_coeff * dy
        force_vecs[i, 2] = f_coeff * dz

    forces = np.zeros((n_nodes, 3), dtype=positions.dtype)
    for i in numba.prange(n_nodes):
        start = node_spring_offsets[i]
        end = node_spring_offsets[i+1]
        f_x = 0.0
        f_y = 0.0
        f_z = 0.0
        for idx in range(start, end):
            sp_id = node_spring_ids[idx]
            sign = node_spring_signs[idx]
            f_x += sign * force_vecs[sp_id, 0]
            f_y += sign * force_vecs[sp_id, 1]
            f_z += sign * force_vecs[sp_id, 2]
        forces[i, 0] = f_x
        forces[i, 1] = f_y
        forces[i, 2] = f_z

    return forces


@backend.jit(parallel=False, static_argnames=("damage_onset_strain", "failure_strain"))
def compute_spring_forces(
    positions: np.ndarray,
    springs: np.ndarray,
    stiffnesses: np.ndarray,
    rest_lengths: np.ndarray,
    failed: np.ndarray,
    tension_only: np.ndarray | None = None,
    node_spring_offsets: np.ndarray | None = None,
    node_spring_ids: np.ndarray | None = None,
    node_spring_signs: np.ndarray | None = None,
    damage_onset_strain: float = 0.9,
    failure_strain: float = 1.0,
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
    tension_only : np.ndarray, optional
        Boolean flags marking orthogonal springs, shape ``(n_springs,)``.
    node_spring_offsets : np.ndarray, optional
        CSR adjacency offsets array for parallel gather.
    node_spring_ids : np.ndarray, optional
        CSR adjacency ids array for parallel gather.
    node_spring_signs : np.ndarray, optional
        CSR adjacency signs array for parallel gather.
    damage_onset_strain : float, optional
        Onset strain for stiffness degradation (progressive damage).
    failure_strain : float, optional
        Strain at complete failure.

    Returns
    -------
    np.ndarray
        Net force on each node, shape ``(n_nodes, 3)``.
    """
    if tension_only is not None:
        t_only = tension_only
    else:
        # Fallback: springs close to min rest length are considered orthogonal
        min_l0 = min(rest_lengths)
        t_only = rest_lengths < 1.1 * min_l0

    if backend.BACKEND == "numba" and backend.HAS_NUMBA:
        if node_spring_offsets is not None and node_spring_ids is not None and node_spring_signs is not None:
            return numba_gather_spring_forces(
                positions,
                springs,
                stiffnesses,
                rest_lengths,
                failed,
                t_only,
                node_spring_offsets,
                node_spring_ids,
                node_spring_signs,
                damage_onset_strain,
                failure_strain,
            )

    # Vectorized fallback
    p1 = positions[springs[:, 0]]
    p2 = positions[springs[:, 1]]
    diff = p2 - p1
    lengths = sqrt(sum(diff**2, axis=1))

    # Avoid divide-by-zero for overlapping nodes
    lengths_safe = where(lengths == 0.0, 1.0, lengths)
    strains = (lengths - rest_lengths) / rest_lengths

    # Progressive damage model
    denom = failure_strain - damage_onset_strain
    denom_safe = where(denom == 0.0, 1.0, denom)
    damage = minimum(maximum((strains - damage_onset_strain) / denom_safe, 0.0), 1.0)
    effective_k = stiffnesses * (1.0 - damage)

    # Spring force magnitude: F = k * strain * L_rest
    f_mag = effective_k * strains * rest_lengths

    # Orthogonal springs are tension-only
    f_mag = where(t_only & (strains < 0.0), 0.0, f_mag)

    # Failed springs carry zero load
    f_mag = where(failed, 0.0, f_mag)

    # Force vectors directed along spring axes
    force_vecs = (f_mag / lengths_safe)[:, np.newaxis] * diff

    # Accumulate forces on nodes
    forces = zeros(positions.shape, dtype=positions.dtype)
    forces = scatter_add(forces, springs[:, 0], force_vecs)
    forces = scatter_add(forces, springs[:, 1], -force_vecs)

    return forces  # type: ignore[no-any-return]


@backend.jit
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
    p1 = positions[springs[:, 0]]
    p2 = positions[springs[:, 1]]
    diff = p2 - p1
    lengths = sqrt(sum(diff**2, axis=1))
    return (lengths - rest_lengths) / rest_lengths  # type: ignore[no-any-return]


@backend.jit(static_argnames=("n_nodes_per_layer", "n_plies"))
def compute_interply_contact_forces(
    positions: np.ndarray,
    n_nodes_per_layer: int,
    n_plies: int,
    t_ply: float,
    k_penalty: float,
    active_counts: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Compute vectorised inter-ply penalty contact forces and potential energy.

    For each corresponding node index across adjacent layers, if layer n
    penetrates layer n+1 along the Z axis, we apply equal and opposite
    forces resisting interpenetration.

    Parameters
    ----------
    positions : np.ndarray
        Current node positions, shape ``(n_nodes, 3)``.
    n_nodes_per_layer : int
        Number of nodes in a single ply.
    n_plies : int
        Number of discrete plies.
    t_ply : float
        Inter-ply spacing (metres).
    k_penalty : float
        Penalty contact stiffness.
    active_counts : np.ndarray, optional
        Number of active springs per node, shape ``(n_nodes,)``.

    Returns
    -------
    tuple[np.ndarray, float]
        - Nodal contact forces array, shape ``(n_nodes, 3)``.
        - Total contact potential energy (Joules).
    """
    forces = zeros(positions.shape, dtype=positions.dtype)
    total_energy = 0.0

    if n_plies <= 1:
        return forces, total_energy

    for ply in range(n_plies - 1):
        # Compute range of node indices for the current layer and next layer
        start_idx = ply * n_nodes_per_layer
        end_idx = start_idx + n_nodes_per_layer

        # Positions along the Z axis
        z_n = positions[start_idx:end_idx, 2]
        z_n1 = positions[end_idx : end_idx + n_nodes_per_layer, 2]

        # Penetration depth: delta = z_n - z_n1 + t_ply
        delta = z_n - z_n1 + t_ply
        penetration = maximum(0.0, delta)

        if active_counts is not None:
            active_n = active_counts[start_idx:end_idx] > 0
            active_n1 = active_counts[end_idx : end_idx + n_nodes_per_layer] > 0
            both_active = active_n & active_n1
            penetration = where(both_active, penetration, 0.0)

        # Force magnitude
        f_mag = k_penalty * penetration

        # Accumulate forces: layer n is pushed in -Z, layer n+1 in +Z
        indices_n = np.arange(start_idx, end_idx)
        indices_n1 = np.arange(end_idx, end_idx + n_nodes_per_layer)

        forces_n = stack_z(-f_mag)
        forces_n1 = stack_z(f_mag)

        forces = scatter_add(forces, indices_n, forces_n)
        forces = scatter_add(forces, indices_n1, forces_n1)

        # Potential energy: 0.5 * k * x^2
        total_energy += sum(0.5 * k_penalty * penetration**2)

    return forces, total_energy
