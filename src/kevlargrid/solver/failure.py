"""Failure criteria module.

Provides binary strain-threshold failure checking for springs in the
lumped-mass network.  Once a spring fails it is permanently deactivated.
"""

from __future__ import annotations

import numpy as np


def scale_failure_strain(epsilon_0: float, dx: float) -> float:
    """Scale the failure strain using Bazant regularization for fine meshes (dx < 1mm)."""
    if dx < 0.001:
        # h_0 = 10mm = 0.01m
        return epsilon_0 * np.sqrt(0.01 / dx)
    return epsilon_0


def check_failures(
    strains: np.ndarray,
    failed: np.ndarray,
    epsilon_fail: float,
) -> np.ndarray:
    """Mark springs as failed if their strain exceeds the failure threshold.

    Failure is irreversible: once a spring is marked as failed it remains
    failed for the rest of the simulation.

    Parameters
    ----------
    strains : np.ndarray
        Current engineering strain per spring, shape ``(n_springs,)``.
    failed : np.ndarray
        Boolean failure flags per spring, shape ``(n_springs,)``.
        Modified **in-place** and also returned.
    epsilon_fail : float
        Failure strain threshold.

    Returns
    -------
    np.ndarray
        Updated boolean failure array, shape ``(n_springs,)``.
    """
    failed |= strains > epsilon_fail
    return failed


def check_progressive_damage(
    strains: np.ndarray,
    damage: np.ndarray,
    failed: np.ndarray,
    damage_onset_strain: float,
    failure_strain: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Update irreversible damage and failure arrays based on strain.

    Parameters
    ----------
    strains : np.ndarray
        Current engineering strain per spring, shape ``(n_springs,)``.
    damage : np.ndarray
        Irreversible damage per spring, shape ``(n_springs,)``.
        Modified **in-place** and also returned.
    failed : np.ndarray
        Boolean failure flags per spring, shape ``(n_springs,)``.
        Modified **in-place** and also returned.
    damage_onset_strain : float
        Onset strain for stiffness degradation.
    failure_strain : float
        Failure strain threshold.

    Returns
    -------
    (np.ndarray, np.ndarray)
        Updated damage and failed arrays.
    """
    denom = failure_strain - damage_onset_strain
    denom_safe = 1.0 if denom == 0.0 else denom
    d_val = np.minimum(np.maximum((strains - damage_onset_strain) / denom_safe, 0.0), 1.0)
    np.maximum(damage, d_val, out=damage)
    failed |= damage >= 1.0
    return damage, failed


def get_layer_failure_stats(
    springs: np.ndarray,
    failed: np.ndarray,
    n_nodes_per_layer: int,
    n_plies: int,
) -> np.ndarray:
    """Compute the number of failed springs in each ply.

    Parameters
    ----------
    springs : np.ndarray
        Spring connectivity array, shape ``(n_springs, 2)``.
    failed : np.ndarray
        Boolean failure flag per spring, shape ``(n_springs,)``.
    n_nodes_per_layer : int
        Number of nodes in a single ply.
    n_plies : int
        Number of discrete plies.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(n_plies,)`` containing the number of failed springs per ply.
    """
    stats = np.zeros(n_plies, dtype=np.int32)
    if n_plies <= 1:
        stats[0] = int(np.sum(failed))
        return stats

    # Vectorised spring layer index calculation
    spring_layers = springs[:, 0] // n_nodes_per_layer

    for ply in range(n_plies):
        stats[ply] = int(np.sum(failed & (spring_layers == ply)))

    return stats
