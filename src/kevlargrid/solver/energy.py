"""Energy tracking module.

Computes kinetic energy, strain energy, and a combined energy balance
for monitoring numerical stability and physical fidelity during a
simulation run.
"""

from __future__ import annotations

import numpy as np

from kevlargrid.solver import backend


@backend.jit
def compute_kinetic_energy(
    velocities: np.ndarray,
    masses: np.ndarray,
) -> float:
    """Compute the total kinetic energy of all nodes.

    Parameters
    ----------
    velocities : np.ndarray
        Node velocities, shape ``(n_nodes, 3)``.
    masses : np.ndarray
        Lumped mass per node, shape ``(n_nodes,)``.

    Returns
    -------
    float
        Total kinetic energy (Joules).
    """
    v_sq = np.sum(velocities**2, axis=1)
    return float(np.sum(0.5 * masses * v_sq))


@backend.jit
def compute_strain_energy(
    strains: np.ndarray,
    stiffnesses: np.ndarray,
    rest_lengths: np.ndarray,
    failed: np.ndarray | None = None,
) -> float:
    """Compute the total elastic strain energy stored in all springs.

    Parameters
    ----------
    strains : np.ndarray
        Engineering strain per spring, shape ``(n_springs,)``.
    stiffnesses : np.ndarray
        Axial stiffness per spring, shape ``(n_springs,)``.
    rest_lengths : np.ndarray
        Natural (rest) length per spring, shape ``(n_springs,)``.
    failed : np.ndarray, optional
        Boolean failure flags per spring, shape ``(n_springs,)``.

    Returns
    -------
    float
        Total strain energy (Joules).
    """
    # SE = 0.5 * k * dx^2 = 0.5 * k * (strain * L0)^2
    se_springs = 0.5 * stiffnesses * (np.maximum(0.0, strains) * rest_lengths) ** 2
    if failed is not None:
        se_springs = np.where(failed, 0.0, se_springs)
    return float(np.sum(se_springs))


def compute_energy_balance(
    ke: float,
    se: float,
    damped: float,
) -> dict:
    """Return a summary dictionary of the energy balance.

    Parameters
    ----------
    ke : float
        Kinetic energy (Joules).
    se : float
        Strain energy (Joules).
    damped : float
        Cumulative energy dissipated by damping (Joules).

    Returns
    -------
    dict
        Dictionary with keys ``"kinetic"``, ``"strain"``, ``"damped"``,
        and ``"total"``.
    """
    total = ke + se + damped
    return {
        "kinetic": float(ke),
        "strain": float(se),
        "damped": float(damped),
        "total": float(total),
    }


def compute_layer_kinetic_energy(
    velocities: np.ndarray,
    masses: np.ndarray,
    n_nodes_per_layer: int,
    n_plies: int,
) -> np.ndarray:
    """Compute the kinetic energy per fabric layer.

    Parameters
    ----------
    velocities : np.ndarray
        Node velocities, shape ``(n_nodes, 3)``.
    masses : np.ndarray
        Lumped mass per node, shape ``(n_nodes,)``.
    n_nodes_per_layer : int
        Number of nodes in a single layer.
    n_plies : int
        Number of discrete plies.

    Returns
    -------
    np.ndarray
        Kinetic energy per layer, shape ``(n_plies,)``.
    """
    ke_layers = np.zeros(n_plies, dtype=np.float64)
    if n_plies <= 1:
        v_sq = np.sum(velocities**2, axis=1)
        ke_layers[0] = float(np.sum(0.5 * masses * v_sq))
        return ke_layers

    for ply in range(n_plies):
        start = ply * n_nodes_per_layer
        end = start + n_nodes_per_layer
        v_sq = np.sum(velocities[start:end] ** 2, axis=1)
        ke_layers[ply] = float(np.sum(0.5 * masses[start:end] * v_sq))

    return ke_layers


def compute_layer_strain_energy(
    strains: np.ndarray,
    stiffnesses: np.ndarray,
    rest_lengths: np.ndarray,
    springs: np.ndarray,
    n_nodes_per_layer: int,
    n_plies: int,
    failed: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the elastic strain energy per fabric layer.

    Parameters
    ----------
    strains : np.ndarray
        Engineering strain per spring, shape ``(n_springs,)``.
    stiffnesses : np.ndarray
        Axial stiffness per spring, shape ``(n_springs,)``.
    rest_lengths : np.ndarray
        Natural (rest) length per spring, shape ``(n_springs,)``.
    springs : np.ndarray
        Spring connectivity array, shape ``(n_springs, 2)``.
    n_nodes_per_layer : int
        Number of nodes in a single layer.
    n_plies : int
        Number of discrete plies.
    failed : np.ndarray, optional
        Boolean failure flags per spring, shape ``(n_springs,)``.

    Returns
    -------
    np.ndarray
        Strain energy per layer, shape ``(n_plies,)``.
    """
    se_layers = np.zeros(n_plies, dtype=np.float64)
    se_springs = 0.5 * stiffnesses * (np.maximum(0.0, strains) * rest_lengths) ** 2
    if failed is not None:
        se_springs = np.where(failed, 0.0, se_springs)

    if n_plies <= 1:
        se_layers[0] = float(np.sum(se_springs))
        return se_layers

    spring_layers = springs[:, 0] // n_nodes_per_layer
    for ply in range(n_plies):
        se_layers[ply] = float(np.sum(se_springs[spring_layers == ply]))

    return se_layers
