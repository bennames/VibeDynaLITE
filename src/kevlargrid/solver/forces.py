"""Spring force computation module.

Provides vectorised routines for computing internal spring forces and
per-spring strain values using 3-D Euclidean distances.
"""

from __future__ import annotations

import numpy as np

from kevlargrid.solver import backend


@backend.jit
def compute_spring_forces(
    positions: np.ndarray,
    springs: np.ndarray,
    stiffnesses: np.ndarray,
    rest_lengths: np.ndarray,
    failed: np.ndarray,
    tension_only: np.ndarray | None = None,
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

    Returns
    -------
    np.ndarray
        Net force on each node, shape ``(n_nodes, 3)``.
    """
    p1 = positions[springs[:, 0]]
    p2 = positions[springs[:, 1]]
    diff = p2 - p1
    lengths = np.sqrt(np.sum(diff**2, axis=1))

    # Avoid divide-by-zero for overlapping nodes
    lengths_safe = np.where(lengths == 0.0, 1.0, lengths)
    strains = (lengths - rest_lengths) / rest_lengths

    # Spring force magnitude: F = k * strain * L_rest
    f_mag = stiffnesses * strains * rest_lengths

    # Orthogonal springs are tension-only
    if tension_only is not None:
        f_mag = np.where(tension_only & (strains < 0.0), 0.0, f_mag)
    else:
        # Fallback: springs close to min rest length are considered orthogonal
        min_l0 = np.min(rest_lengths)
        is_ortho = rest_lengths < 1.1 * min_l0
        f_mag = np.where(is_ortho & (strains < 0.0), 0.0, f_mag)

    # Failed springs carry zero load
    f_mag = np.where(failed, 0.0, f_mag)

    # Force vectors directed along spring axes
    force_vecs = (f_mag / lengths_safe)[:, np.newaxis] * diff

    # Accumulate forces on nodes
    forces = np.zeros_like(positions)
    for idx in range(len(springs)):
        u = springs[idx, 0]
        v = springs[idx, 1]
        f_vec = force_vecs[idx]
        forces[u] += f_vec
        forces[v] -= f_vec

    return forces


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
    lengths = np.sqrt(np.sum(diff**2, axis=1))
    return (lengths - rest_lengths) / rest_lengths  # type: ignore[no-any-return]


@backend.jit
def compute_interply_contact_forces(
    positions: np.ndarray,
    n_nodes_per_layer: int,
    n_plies: int,
    t_ply: float,
    k_penalty: float,
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

    Returns
    -------
    tuple[np.ndarray, float]
        - Nodal contact forces array, shape ``(n_nodes, 3)``.
        - Total contact potential energy (Joules).
    """
    forces = np.zeros_like(positions)
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
        penetration = np.maximum(0.0, delta)

        # Force magnitude
        f_mag = k_penalty * penetration

        # Accumulate forces: layer n is pushed in -Z, layer n+1 in +Z
        for i in range(n_nodes_per_layer):
            forces[start_idx + i, 2] -= f_mag[i]
            forces[end_idx + i, 2] += f_mag[i]

        # Potential energy: 0.5 * k * x^2
        total_energy += float(np.sum(0.5 * k_penalty * penetration**2))

    return forces, total_energy
