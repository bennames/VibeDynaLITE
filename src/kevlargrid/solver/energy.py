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
    v_sq = backend.sum(velocities**2, axis=1)
    return backend.sum(0.5 * masses * v_sq)


@backend.jit
def compute_strain_energy(
    strains: np.ndarray,
    stiffnesses: np.ndarray,
    rest_lengths: np.ndarray,
    failed: np.ndarray | None = None,
    damage: np.ndarray | None = None,
    tension_only: np.ndarray | None = None,
    elements: np.ndarray | None = None,
    element_stress: np.ndarray | None = None,
    thickness: float = 0.002,
    dx: float = 0.01,
    E: float = 71e9,
    nu: float = 0.3,
) -> float:
    """Compute the total elastic strain energy stored in all springs or shell elements.

    Parameters
    ----------
    strains : np.ndarray
        Engineering strain per spring, shape ``(n_springs,)``.
    stiffnesses : np.ndarray
        Axial stiffness per spring, shape ``(n_springs,)``.
    rest_lengths : np.ndarray
        Rest length per spring, shape ``(n_springs,)``.
    failed : np.ndarray, optional
        Boolean failure array.
    damage : np.ndarray, optional
        Scalar damage array.
    tension_only : np.ndarray, optional
        Boolean indicating if tension-only behavior is active.
    elements : np.ndarray, optional
        Shell element connectivity array.
    element_stress : np.ndarray, optional
        Shell element thickness point stresses, shape ``(n_elements, 3, 3)``.
    thickness : float, default 0.002
        Shell thickness.
    dx : float, default 0.01
        Element length.
    E : float, default 71e9
        Young's modulus.
    nu : float, default 0.3
        Poisson's ratio.

    Returns
    -------
    float
        Total strain energy (Joules).
    """
    if elements is not None and element_stress is not None:
        se = 0.0
        n_elements = len(elements)
        se = 0.0
        n_elements = len(elements)
        n_pts = element_stress.shape[1]
        if n_pts == 5:
            w_pts = np.array(
                [
                    thickness / 12.0,
                    4.0 * thickness / 12.0,
                    2.0 * thickness / 12.0,
                    4.0 * thickness / 12.0,
                    thickness / 12.0,
                ]
            )
        else:
            w_pts = np.array([thickness / 6.0, 4.0 * thickness / 6.0, thickness / 6.0])
        for e in range(n_elements):
            if failed is None or not failed[e]:
                el_se = 0.0
                for k in range(n_pts):
                    wk = w_pts[k]
                    s_xx = element_stress[e, k, 0]
                    s_yy = element_stress[e, k, 1]
                    t_xy = element_stress[e, k, 2]
                    d_factor = 1.0 - damage[e, k] if damage is not None else 1.0
                    u0 = (
                        (0.5 / E)
                        * (s_xx**2 + s_yy**2 - 2.0 * nu * s_xx * s_yy + 2.0 * (1.0 + nu) * t_xy**2)
                        * d_factor
                    )
                    el_se += u0 * wk
                se += el_se * (dx * dx)
        return float(se)

    # SE = 0.5 * k * (1 - D) * dx^2 = 0.5 * k_eff * (strain * L0)^2
    eff_k = stiffnesses
    if damage is not None:
        eff_k = stiffnesses * (1.0 - damage)
    if tension_only is not None:
        strains_eff = backend.where(tension_only & (strains < 0.0), 0.0, strains)
    else:
        strains_eff = strains
    se_springs = 0.5 * eff_k * (strains_eff * rest_lengths) ** 2
    if failed is not None:
        se_springs = backend.where(failed, 0.0, se_springs)
    return backend.sum(se_springs)


def compute_energy_balance(
    ke: float,
    se: float,
    damped: float,
    failure_dissipated: float = 0.0,
    clamp_dissipated: float = 0.0,
    proj_ke: float = 0.0,
    friction_dissipated: float = 0.0,
) -> dict:
    """Return a summary dictionary of the energy balance.

    Parameters
    ----------
    ke : float
        Fabric kinetic energy (Joules).
    se : float
        Fabric strain energy (Joules).
    damped : float
        Cumulative energy dissipated by damping (Joules).
    failure_dissipated : float, optional
        Cumulative energy dissipated by spring fracture (Joules).
    clamp_dissipated : float, optional
        Cumulative energy dissipated by velocity clamping (Joules).
    proj_ke : float, optional
        Projectile kinetic energy (Joules).
    friction_dissipated : float, optional
        Cumulative energy dissipated by friction (Joules).

    Returns
    -------
    dict
        Dictionary of energy components and the total system energy.
    """
    total = ke + se + damped + failure_dissipated + clamp_dissipated + proj_ke + friction_dissipated
    return {
        "kinetic": float(ke),
        "strain": float(se),
        "damped": float(damped),
        "failure_dissipated": float(failure_dissipated),
        "clamp_dissipated": float(clamp_dissipated),
        "friction_dissipated": float(friction_dissipated),
        "projectile_kinetic": float(proj_ke),
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
    damage: np.ndarray | None = None,
    tension_only: np.ndarray | None = None,
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
    damage : np.ndarray, optional
        Damage fraction per spring, shape ``(n_springs,)``.
    tension_only : np.ndarray, optional
        Boolean tension-only flags per spring, shape ``(n_springs,)``.

    Returns
    -------
    np.ndarray
        Strain energy per layer, shape ``(n_plies,)``.
    """
    se_layers = np.zeros(n_plies, dtype=np.float64)
    eff_k = stiffnesses
    if damage is not None:
        eff_k = stiffnesses * (1.0 - damage)
    if tension_only is not None:
        strains_eff = np.where(tension_only & (strains < 0.0), 0.0, strains)
    else:
        strains_eff = strains
    se_springs = 0.5 * eff_k * (strains_eff * rest_lengths) ** 2
    if failed is not None:
        se_springs = np.where(failed, 0.0, se_springs)

    if n_plies <= 1:
        se_layers[0] = float(np.sum(se_springs))
        return se_layers

    spring_layers = springs[:, 0] // n_nodes_per_layer
    for ply in range(n_plies):
        se_layers[ply] = float(np.sum(se_springs[spring_layers == ply]))

    return se_layers
