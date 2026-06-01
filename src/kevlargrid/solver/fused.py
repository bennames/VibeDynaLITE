"""Fused multi-step time integration runner.

Runs a large block (chunk) of dynamic explicit integration steps entirely in native
machine code using JIT compilation, bypassing Python interpreter overhead.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import numba
except ImportError:
    numba = None

from kevlargrid.solver import backend
from kevlargrid.solver.backend import (
    clamp_boundary,
    maximum,
    sqrt,
    stack_z,
    sum,
    where,
    zeros,
)
from kevlargrid.solver.energy import compute_kinetic_energy, compute_strain_energy
from kevlargrid.solver.forces import compute_interply_contact_forces, compute_spring_forces

# Define helper to set values JAX/Numba-compatibly
if backend.HAS_NUMBA and numba is not None:

    @numba.njit(cache=True)
    def numba_set_index_3d(target: np.ndarray, index: int, value: np.ndarray) -> np.ndarray:
        target[index] = value
        return target

    @numba.njit(cache=True)
    def numba_set_index_2d_bool(target: np.ndarray, index: int, value: np.ndarray) -> np.ndarray:
        target[index] = value
        return target

    @numba.njit(cache=True)
    def numba_set_index_2d_float(target: np.ndarray, index: int, value: np.ndarray) -> np.ndarray:
        target[index] = value
        return target

    @numba.njit(cache=True)
    def numba_set_index_1d(target: np.ndarray, index: int, value: float) -> np.ndarray:
        target[index] = value
        return target
else:
    numba_set_index_3d = None
    numba_set_index_2d_bool = None
    numba_set_index_2d_float = None
    numba_set_index_1d = None


def py_set_index_3d(target: Any, index: int, value: Any) -> Any:
    if backend.BACKEND == "jax" and backend.HAS_JAX:
        return target.at[index].set(value)
    target[index] = value
    return target


def py_set_index_2d_bool(target: Any, index: int, value: Any) -> Any:
    if backend.BACKEND == "jax" and backend.HAS_JAX:
        return target.at[index].set(value)
    target[index] = value
    return target


def py_set_index_2d_float(target: Any, index: int, value: Any) -> Any:
    if backend.BACKEND == "jax" and backend.HAS_JAX:
        return target.at[index].set(value)
    target[index] = value
    return target


def py_set_index_1d(target: Any, index: int, value: float) -> Any:
    if backend.BACKEND == "jax" and backend.HAS_JAX:
        return target.at[index].set(value)
    target[index] = value
    return target


if backend.BACKEND == "numba" and backend.HAS_NUMBA:
    import numba

    set_index_3d = numba_set_index_3d
    set_index_2d_bool = numba_set_index_2d_bool
    set_index_2d_float = numba_set_index_2d_float
    set_index_1d = numba_set_index_1d
else:
    set_index_3d = py_set_index_3d
    set_index_2d_bool = py_set_index_2d_bool
    set_index_2d_float = py_set_index_2d_float
    set_index_1d = py_set_index_1d


@backend.jit
def fused_leapfrog_loop(
    positions: np.ndarray,
    velocities: np.ndarray,
    grid_springs: np.ndarray,
    grid_stiffnesses: np.ndarray,
    grid_rest_lengths: np.ndarray,
    grid_failed: np.ndarray,
    grid_masses: np.ndarray,
    boundary_mask: np.ndarray,
    proj_position: np.ndarray,
    proj_velocity: np.ndarray,
    proj_mass: float,
    proj_blade_width: float,
    proj_edge_thickness: float,
    n_plies: int,
    n_nodes_per_layer: int,
    t_ply: float,
    k_penalty: float,
    damping_coeff: float,
    failure_strain: float,
    dt: float,
    n_steps: int,
    save_interval: int,
    damp_dissipated_init: float,
    t_sim_init: float,
) -> tuple[
    np.ndarray,  # positions
    np.ndarray,  # velocities
    np.ndarray,  # grid_failed
    np.ndarray,  # proj_position
    np.ndarray,  # proj_velocity
    float,  # damp_dissipated
    float,  # t_sim
    np.ndarray,  # hist_positions
    np.ndarray,  # hist_failed
    np.ndarray,  # hist_proj_pos
    np.ndarray,  # hist_time
    np.ndarray,  # hist_ke
    np.ndarray,  # hist_se
    np.ndarray,  # hist_proj_ke
]:
    """Execute a chunk of explicit dynamics solver steps in compiled machine code."""
    n_nodes = len(positions)
    n_springs = len(grid_springs)
    m_frames = n_steps // save_interval

    # Pre-allocate history structures (compatible with JIT vector allocations)
    hist_positions = zeros((m_frames, n_nodes, 3), dtype=positions.dtype)
    hist_failed = zeros((m_frames, n_springs), dtype=np.bool_)
    hist_proj_pos = zeros((m_frames, 3), dtype=positions.dtype)
    hist_time = zeros(m_frames, dtype=positions.dtype)
    hist_ke = zeros(m_frames, dtype=positions.dtype)
    hist_se = zeros(m_frames, dtype=positions.dtype)
    hist_proj_ke = zeros(m_frames, dtype=positions.dtype)

    w_h = proj_blade_width / 2.0
    t_h = proj_edge_thickness / 2.0
    # Proximity threshold matches dx * 2.0.
    # In VibeDynaLITE, grid node spacing dx can be computed from the grid nodes
    # coordinate difference.
    # Let's compute average coordinate difference dx
    dx = 0.05  # fallback
    if n_nodes_per_layer > 1:
        dx = float(positions[1, 0] - positions[0, 0])
        if dx == 0.0:
            dx = 0.05
    proximity_threshold = dx * 2.0

    damp_dissipated = damp_dissipated_init
    t_sim = t_sim_init

    # Loop steps inside JIT boundary
    for step in range(n_steps):
        # 1. Projectile Contact Forces (IDW Distribution)
        x_proj = np.clip(positions[:, 0], proj_position[0] - w_h, proj_position[0] + w_h)
        y_proj = np.clip(positions[:, 1], proj_position[1] - t_h, proj_position[1] + t_h)
        dists = sqrt(
            (positions[:, 0] - x_proj) ** 2
            + (positions[:, 1] - y_proj) ** 2
            + (positions[:, 2] - proj_position[2]) ** 2
        )
        contact_mask = dists <= proximity_threshold

        direction = np.sign(proj_velocity[2]) if proj_velocity[2] != 0.0 else 1.0

        w_i = where(contact_mask, 1.0 / maximum(dists, 1e-4), 0.0)
        n_contacts = sum(contact_mask)

        # Vectorized force distribution
        proj_forces = zeros(positions.shape, dtype=positions.dtype)
        if n_contacts > 0:
            w_sum = sum(w_i)
            w_mean = w_sum / n_contacts
            w_normalized = where(contact_mask, w_i / w_mean if w_mean > 0.0 else w_i, 0.0)

            penetration = maximum(0.0, (proj_position[2] - positions[:, 2]) * direction)
            f_i = k_penalty * w_normalized * penetration
            proj_forces = stack_z(f_i * direction)

        # 2. Inter-ply Contact Forces (Checkout Mode)
        interply_forces, _ = compute_interply_contact_forces(
            positions, n_nodes_per_layer, n_plies, t_ply, k_penalty
        )

        # 3. Internal Spring Forces
        spring_forces = compute_spring_forces(
            positions, grid_springs, grid_stiffnesses, grid_rest_lengths, grid_failed
        )

        # 4. Viscous Damping Forces & Energy Dissipation
        damp_forces = -damping_coeff * velocities
        p_damp = sum(damp_forces * velocities)
        damp_dissipated += float(-p_damp * dt)

        # Net acceleration calculation
        net_forces = spring_forces + proj_forces + interply_forces + damp_forces

        # Reset forces/velocities on boundary clamped nodes
        net_forces = clamp_boundary(net_forces, boundary_mask)
        velocities = clamp_boundary(velocities, boundary_mask)

        # Integrate node dynamics (leapfrog Verlet)
        masses_col = grid_masses.reshape(-1, 1)
        accel = net_forces / masses_col
        velocities = velocities + accel * dt
        velocities = clamp_boundary(velocities, boundary_mask)
        positions = positions + velocities * dt

        # Integrate rigid-body projectile kinematics
        proj_reaction_force = -sum(proj_forces, axis=0)
        proj_accel = proj_reaction_force / proj_mass
        proj_velocity = proj_velocity + proj_accel * dt
        proj_position = proj_position + proj_velocity * dt

        # Evolve failures irreversibly
        p1 = positions[grid_springs[:, 0]]
        p2 = positions[grid_springs[:, 1]]
        diff = p2 - p1
        lengths = sqrt(sum(diff**2, axis=1))
        strains = (lengths - grid_rest_lengths) / grid_rest_lengths
        grid_failed = grid_failed | (strains > failure_strain)

        t_sim += dt

        # Periodically capture frame telemetry
        step_1indexed = step + 1
        if step_1indexed % save_interval == 0:
            frame_idx = (step_1indexed // save_interval) - 1

            # Energy Metrics S6.5.5
            ke = compute_kinetic_energy(velocities, grid_masses)
            se = compute_strain_energy(strains, grid_stiffnesses, grid_rest_lengths, grid_failed)
            proj_ke = 0.5 * proj_mass * sum(proj_velocity**2)

            hist_positions = set_index_3d(hist_positions, frame_idx, positions)
            hist_failed = set_index_2d_bool(hist_failed, frame_idx, grid_failed)
            hist_proj_pos = set_index_2d_float(hist_proj_pos, frame_idx, proj_position)
            hist_time = set_index_1d(hist_time, frame_idx, t_sim)
            hist_ke = set_index_1d(hist_ke, frame_idx, ke)
            hist_se = set_index_1d(hist_se, frame_idx, se)
            hist_proj_ke = set_index_1d(hist_proj_ke, frame_idx, proj_ke)

    return (
        positions,
        velocities,
        grid_failed,
        proj_position,
        proj_velocity,
        damp_dissipated,
        t_sim,
        hist_positions,
        hist_failed,
        hist_proj_pos,
        hist_time,
        hist_ke,
        hist_se,
        hist_proj_ke,
    )
