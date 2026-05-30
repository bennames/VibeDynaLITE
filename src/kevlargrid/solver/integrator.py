"""Time integration module.

Implements the central-difference (Störmer–Verlet / leapfrog) explicit
time-integration scheme for the lumped-mass spring network.
"""

from __future__ import annotations

import numpy as np

from kevlargrid.solver import backend


@backend.jit
def leapfrog_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    forces: np.ndarray,
    masses: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance one time-step using the central-difference (leapfrog) method.

    The leapfrog (Störmer–Verlet) integrator is symplectic and
    second-order accurate, making it well-suited for explicit dynamics.

    Parameters
    ----------
    positions : np.ndarray
        Current node positions, shape ``(n_nodes, 3)``.
    velocities : np.ndarray
        Current node velocities at t - dt/2, shape ``(n_nodes, 3)``.
    forces : np.ndarray
        Net forces on each node at t, shape ``(n_nodes, 3)``.
    masses : np.ndarray
        Lumped mass per node, shape ``(n_nodes,)`` or ``(n_nodes, 1)``.
    dt : float
        Time-step size (seconds).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Updated ``(positions, velocities)`` arrays at t + dt and t + dt/2.
    """
    # Reshape masses to support broad-casting
    masses_col = masses.reshape(-1, 1) if masses.ndim == 1 else masses

    # a(t) = F(t) / m
    accel = forces / masses_col

    # v(t + dt/2) = v(t - dt/2) + a(t) * dt
    v_new = velocities + accel * dt

    # x(t + dt) = x(t) + v(t + dt/2) * dt
    x_new = positions + v_new * dt

    return x_new, v_new
