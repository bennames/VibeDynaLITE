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
    se_springs = 0.5 * stiffnesses * (strains * rest_lengths) ** 2
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
