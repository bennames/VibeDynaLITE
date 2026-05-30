"""Failure criteria module.

Provides binary strain-threshold failure checking for springs in the
lumped-mass network.  Once a spring fails it is permanently deactivated.
"""

from __future__ import annotations

import numpy as np


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
    raise NotImplementedError("Stub")
