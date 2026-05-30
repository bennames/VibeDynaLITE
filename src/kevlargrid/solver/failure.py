"""Failure criteria module.

Provides binary strain-threshold failure checking for springs in the
lumped-mass network.  Once a spring fails it is permanently deactivated.
"""

from __future__ import annotations

import numpy as np

from kevlargrid.solver import backend


@backend.jit
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
