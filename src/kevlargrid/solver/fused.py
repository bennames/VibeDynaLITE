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
    NUMBA_CACHE,
    clamp_boundary,
    maximum,
    min,
    minimum,
    scatter_add,
    sqrt,
    stack_z,
    sum,
    where,
    zeros,
)
from kevlargrid.solver.energy import compute_kinetic_energy, compute_strain_energy
from kevlargrid.solver.forces import compute_interply_contact_forces

# Define helper to set values JAX/Numba-compatibly
if backend.HAS_NUMBA and numba is not None:

    @numba.njit(cache=NUMBA_CACHE)
    def numba_set_index_3d(target: np.ndarray, index: int, value: np.ndarray) -> np.ndarray:
        target[index] = value
        return target

    @numba.njit(cache=NUMBA_CACHE)
    def numba_set_index_2d_bool(target: np.ndarray, index: int, value: np.ndarray) -> np.ndarray:
        target[index] = value
        return target

    @numba.njit(cache=NUMBA_CACHE)
    def numba_set_index_2d_float(target: np.ndarray, index: int, value: np.ndarray) -> np.ndarray:
        target[index] = value
        return target

    @numba.njit(cache=NUMBA_CACHE)
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


@backend.jit(parallel=True, fastmath=True)
def numba_compute_effective_k(
    positions: np.ndarray,
    springs: np.ndarray,
    stiffnesses: np.ndarray,
    rest_lengths: np.ndarray,
    failed: np.ndarray,
    damage_onset_strain: float,
    failure_strain: float,
) -> np.ndarray:
    n_springs = len(springs)
    effective_k = np.zeros(n_springs, dtype=positions.dtype)
    for i in numba.prange(n_springs):
        if failed[i]:
            effective_k[i] = 0.0
            continue
        n0 = springs[i, 0]
        n1 = springs[i, 1]
        dx = positions[n1, 0] - positions[n0, 0]
        dy = positions[n1, 1] - positions[n0, 1]
        dz = positions[n1, 2] - positions[n0, 2]
        length = np.sqrt(dx * dx + dy * dy + dz * dz)
        strain = (length - rest_lengths[i]) / rest_lengths[i]

        denom = failure_strain - damage_onset_strain
        denom_safe = denom if denom != 0.0 else 1.0
        val = (strain - damage_onset_strain) / denom_safe
        damage = 0.0
        if val > 0.0:
            damage = val if val < 1.0 else 1.0
        effective_k[i] = stiffnesses[i] * (1.0 - damage)
    return effective_k


@backend.jit(parallel=True, fastmath=True)
def numba_sum_nodal_k_springs(
    effective_k: np.ndarray,
    node_spring_offsets: np.ndarray,
    node_spring_ids: np.ndarray,
) -> np.ndarray:
    n_nodes = len(node_spring_offsets) - 1
    nodal_k_springs = np.zeros(n_nodes, dtype=effective_k.dtype)
    for i in numba.prange(n_nodes):
        start = node_spring_offsets[i]
        end = node_spring_offsets[i + 1]
        sum_k = 0.0
        for idx in range(start, end):
            sp_id = node_spring_ids[idx]
            sum_k += effective_k[sp_id]
        nodal_k_springs[i] = sum_k
    return nodal_k_springs


@backend.jit(parallel=True, fastmath=True)
def numba_compute_failure_dissipated(
    positions: np.ndarray,
    springs: np.ndarray,
    stiffnesses: np.ndarray,
    rest_lengths: np.ndarray,
    failed: np.ndarray,
    damage_onset_strain: float,
    failure_strain: float,
    fracture_energy_multiplier: float,
) -> float:
    n_springs = len(springs)
    total_diss = 0.0
    for i in numba.prange(n_springs):
        k = stiffnesses[i]
        L0 = rest_lengths[i]
        x_onset = damage_onset_strain
        x_fail = failure_strain

        if failed[i]:
            w_failed = (k * L0**2 / 6.0) * (x_fail**2 + x_fail * x_onset + x_onset**2)
            total_diss += fracture_energy_multiplier * w_failed
            continue

        n0 = springs[i, 0]
        n1 = springs[i, 1]
        dx = positions[n1, 0] - positions[n0, 0]
        dy = positions[n1, 1] - positions[n0, 1]
        dz = positions[n1, 2] - positions[n0, 2]
        length = np.sqrt(dx * dx + dy * dy + dz * dz)
        x = (length - L0) / L0

        if x >= x_onset:
            denom = x_fail - x_onset
            denom_safe = denom if denom != 0.0 else 1.0
            damage = (x - x_onset) / denom_safe
            if damage > 1.0:
                damage = 1.0
            effective_k = k * (1.0 - damage)

            w_input = (k * L0**2 / 6.0) * (x**2 + x * x_onset + x_onset**2)
            se_actual = 0.5 * effective_k * (x * L0) ** 2
            total_diss += fracture_energy_multiplier * (w_input - se_actual)

    return total_diss


@backend.jit(parallel=True, fastmath=True)
def numba_gather_active_counts(
    grid_failed: np.ndarray,
    node_spring_offsets: np.ndarray,
    node_spring_ids: np.ndarray,
) -> np.ndarray:
    n_nodes = len(node_spring_offsets) - 1
    active_counts = np.zeros(n_nodes, dtype=np.float64)
    for i in numba.prange(n_nodes):
        start = node_spring_offsets[i]
        end = node_spring_offsets[i + 1]
        cnt = 0
        for idx in range(start, end):
            sp_id = node_spring_ids[idx]
            if not grid_failed[sp_id]:
                cnt += 1
        active_counts[i] = cnt
    return active_counts


@backend.jit(parallel=True, fastmath=True)
def numba_step_internal_forces_and_failures(
    positions: np.ndarray,
    velocities: np.ndarray,
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
    rayleigh_beta: float,
) -> tuple[np.ndarray, float, float]:
    n_springs = len(springs)
    n_nodes = len(positions)
    force_vecs = np.zeros((n_springs, 3), dtype=positions.dtype)

    step_fracture_energy = 0.0
    step_stiff_damp_power = 0.0

    for i in numba.prange(n_springs):
        n0 = springs[i, 0]
        n1 = springs[i, 1]

        dx = positions[n1, 0] - positions[n0, 0]
        dy = positions[n1, 1] - positions[n0, 1]
        dz = positions[n1, 2] - positions[n0, 2]

        length = np.sqrt(dx * dx + dy * dy + dz * dz)
        length_safe = length if length != 0.0 else 1.0

        strain = (length - rest_lengths[i]) / rest_lengths[i]

        is_failed = failed[i]
        if not is_failed:
            if strain > failure_strain:
                is_failed = True
                failed[i] = True
                step_fracture_energy += 0.5 * stiffnesses[i] * (strain * rest_lengths[i]) ** 2

        # Progressive damage
        denom = failure_strain - damage_onset_strain
        denom_safe = denom if denom != 0.0 else 1.0
        val = (strain - damage_onset_strain) / denom_safe
        damage = 0.0
        if val > 0.0:
            damage = val if val < 1.0 else 1.0

        effective_k = stiffnesses[i] * (1.0 - damage)
        f_mag = effective_k * strain * rest_lengths[i]

        if tension_only[i] and strain < 0.0:
            f_mag = 0.0

        if is_failed:
            f_mag = 0.0

        damp_mag = 0.0
        if not is_failed and rayleigh_beta > 0.0:
            dvx = velocities[n1, 0] - velocities[n0, 0]
            dvy = velocities[n1, 1] - velocities[n0, 1]
            dvz = velocities[n1, 2] - velocities[n0, 2]
            v_proj = (dvx * dx + dvy * dy + dvz * dz) / length_safe
            damp_mag = rayleigh_beta * stiffnesses[i] * v_proj
            step_stiff_damp_power += damp_mag * v_proj

        total_mag = f_mag + damp_mag
        f_coeff = total_mag / length_safe
        force_vecs[i, 0] = f_coeff * dx
        force_vecs[i, 1] = f_coeff * dy
        force_vecs[i, 2] = f_coeff * dz

    forces = np.zeros((n_nodes, 3), dtype=positions.dtype)
    for i in numba.prange(n_nodes):
        start = node_spring_offsets[i]
        end = node_spring_offsets[i + 1]
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

    return forces, step_fracture_energy, step_stiff_damp_power


@backend.jit(parallel=True, fastmath=True)
def numba_clamp_velocities(
    velocities: np.ndarray, masses: np.ndarray, v_max: float
) -> tuple[np.ndarray, float]:
    n_nodes = len(velocities)
    excess_ke = 0.0
    for i in numba.prange(n_nodes):
        vx = velocities[i, 0]
        vy = velocities[i, 1]
        vz = velocities[i, 2]
        v_mag = np.sqrt(vx * vx + vy * vy + vz * vz)
        if v_mag > v_max:
            scale = v_max / v_mag
            velocities[i, 0] = vx * scale
            velocities[i, 1] = vy * scale
            velocities[i, 2] = vz * scale
            excess_ke += 0.5 * masses[i] * (v_mag * v_mag - v_max * v_max)
    return velocities, excess_ke


@backend.jit(
    parallel=False,
    static_argnames=(
        "n_plies",
        "n_nodes_per_layer",
        "n_steps",
        "save_interval",
        "use_viscous",
        "cfl_factor",
    ),
)
def fused_leapfrog_loop(
    positions: np.ndarray,
    velocities: np.ndarray,
    grid_springs: np.ndarray,
    grid_stiffnesses: np.ndarray,
    grid_rest_lengths: np.ndarray,
    grid_failed: np.ndarray,
    grid_masses: np.ndarray,
    grid_tension_only: np.ndarray,
    boundary_mask: np.ndarray,
    nodal_external_forces: np.ndarray,
    proj_position: np.ndarray,
    proj_velocity: np.ndarray,
    proj_mass: float,
    proj_blade_width: float,
    proj_edge_thickness: float,
    n_plies: int,
    n_nodes_per_layer: int,
    t_ply: float,
    dx: float,
    k_penalty: float,
    rayleigh_alpha: float,
    rayleigh_beta: float,
    failure_strain: float,
    damage_onset_strain: float,
    fracture_energy_multiplier: float,
    dt: float,
    n_steps: int,
    save_interval: int,
    damp_dissipated_init: float,
    failure_dissipated_init: float,
    clamp_dissipated_init: float,
    t_sim_init: float,
    strike_direction: float,
    node_initial_springs: np.ndarray,
    node_spring_offsets: np.ndarray,
    node_spring_ids: np.ndarray,
    node_spring_signs: np.ndarray,
    use_viscous: bool = False,
    cfl_factor: float = -1.0,
) -> tuple[
    np.ndarray,  # positions
    np.ndarray,  # velocities
    np.ndarray,  # grid_failed
    np.ndarray,  # proj_position
    np.ndarray,  # proj_velocity
    float,  # damp_dissipated
    float,  # failure_dissipated
    float,  # clamp_dissipated
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
    proximity_threshold = dx * 2.0

    damp_dissipated = damp_dissipated_init
    failure_dissipated = failure_dissipated_init
    clamp_dissipated = clamp_dissipated_init
    t_sim = t_sim_init

    masses_col = grid_masses.reshape(-1, 1)

    # Loop steps inside JIT boundary
    for step in range(n_steps):
        # 0. Compute Nodal Stiffnesses & CFL Timestep
        if cfl_factor > 0.0:
            if backend.BACKEND == "numba" and backend.HAS_NUMBA:
                effective_k = numba_compute_effective_k(
                    positions,
                    grid_springs,
                    grid_stiffnesses,
                    grid_rest_lengths,
                    grid_failed,
                    damage_onset_strain,
                    failure_strain,
                )
                nodal_k_springs = numba_sum_nodal_k_springs(
                    effective_k,
                    node_spring_offsets,
                    node_spring_ids,
                )
                active_counts = numba_gather_active_counts(
                    grid_failed, node_spring_offsets, node_spring_ids
                )
            else:
                p1 = positions[grid_springs[:, 0]]
                p2 = positions[grid_springs[:, 1]]
                diff = p2 - p1
                lengths = sqrt(sum(diff**2, axis=1))
                strains = (lengths - grid_rest_lengths) / grid_rest_lengths

                denom = failure_strain - damage_onset_strain
                denom_safe = where(denom == 0.0, 1.0, denom)
                damage = minimum(maximum((strains - damage_onset_strain) / denom_safe, 0.0), 1.0)
                effective_k = grid_stiffnesses * (1.0 - damage)
                effective_k = where(grid_failed, 0.0, effective_k)

                nodal_k_springs = zeros(n_nodes, dtype=positions.dtype)
                nodal_k_springs = scatter_add(nodal_k_springs, grid_springs[:, 0], effective_k)
                nodal_k_springs = scatter_add(nodal_k_springs, grid_springs[:, 1], effective_k)

                active_springs = where(grid_failed, 0, 1)
                active_counts = zeros(n_nodes, dtype=positions.dtype)
                active_counts = scatter_add(active_counts, grid_springs[:, 0], active_springs)
                active_counts = scatter_add(active_counts, grid_springs[:, 1], active_springs)

            # Contact mask & weights
            x_proj = maximum(
                proj_position[0] - w_h, minimum(positions[:, 0], proj_position[0] + w_h)
            )
            y_proj = maximum(
                proj_position[1] - t_h, minimum(positions[:, 1], proj_position[1] + t_h)
            )
            dists = sqrt(
                (positions[:, 0] - x_proj) ** 2
                + (positions[:, 1] - y_proj) ** 2
                + (positions[:, 2] - proj_position[2]) ** 2
            )
            contact_mask = dists <= proximity_threshold

            w_i = where(contact_mask, 1.0 / maximum(dists, 1e-4), 0.0)
            n_contacts = sum(contact_mask)
            w_sum = sum(w_i)
            w_mean = w_sum / where(n_contacts > 0, n_contacts, 1)
            w_mean_safe = where(w_mean > 0.0, w_mean, 1.0)
            w_normalized = where(contact_mask, w_i / w_mean_safe, 0.0)

            scale_factor = where(
                node_initial_springs > 0, active_counts / node_initial_springs, 0.0
            )
            nodal_k_contact = where(contact_mask, k_penalty * w_normalized * scale_factor, 0.0)

            nodal_k_interply = zeros(n_nodes, dtype=positions.dtype)
            if n_plies > 1:
                for ply in range(n_plies - 1):
                    start_idx = ply * n_nodes_per_layer
                    end_idx = start_idx + n_nodes_per_layer
                    z_n = positions[start_idx:end_idx, 2]
                    z_n1 = positions[end_idx : end_idx + n_nodes_per_layer, 2]
                    delta = z_n - z_n1 + t_ply
                    penetrating = delta > 0.0
                    active_n = active_counts[start_idx:end_idx] > 0
                    active_n1 = active_counts[end_idx : end_idx + n_nodes_per_layer] > 0
                    both_active = active_n & active_n1 & penetrating
                    k_add = where(both_active, k_penalty, 0.0)

                    indices_n = np.arange(start_idx, end_idx)
                    indices_n1 = np.arange(end_idx, end_idx + n_nodes_per_layer)

                    nodal_k_interply = scatter_add(nodal_k_interply, indices_n, k_add)
                    nodal_k_interply = scatter_add(nodal_k_interply, indices_n1, k_add)

            total_nodal_k = nodal_k_springs + nodal_k_contact + nodal_k_interply
            total_nodal_k = maximum(total_nodal_k, 1e-4)

            # Calculate dynamic stable timestep dt
            dt_crit = min(sqrt(grid_masses / total_nodal_k))
            dt = cfl_factor * dt_crit
            v_max = dx / dt
        else:
            # Statically use the passed dt
            if backend.BACKEND == "numba" and backend.HAS_NUMBA:
                active_counts = numba_gather_active_counts(
                    grid_failed, node_spring_offsets, node_spring_ids
                )
            else:
                active_springs = where(grid_failed, 0, 1)
                active_counts = zeros(n_nodes, dtype=positions.dtype)
                active_counts = scatter_add(active_counts, grid_springs[:, 0], active_springs)
                active_counts = scatter_add(active_counts, grid_springs[:, 1], active_springs)

            # Contact mask & weights
            x_proj = maximum(
                proj_position[0] - w_h, minimum(positions[:, 0], proj_position[0] + w_h)
            )
            y_proj = maximum(
                proj_position[1] - t_h, minimum(positions[:, 1], proj_position[1] + t_h)
            )
            dists = sqrt(
                (positions[:, 0] - x_proj) ** 2
                + (positions[:, 1] - y_proj) ** 2
                + (positions[:, 2] - proj_position[2]) ** 2
            )
            contact_mask = dists <= proximity_threshold

            w_i = where(contact_mask, 1.0 / maximum(dists, 1e-4), 0.0)
            n_contacts = sum(contact_mask)
            w_sum = sum(w_i)
            w_mean = w_sum / where(n_contacts > 0, n_contacts, 1)
            w_mean_safe = where(w_mean > 0.0, w_mean, 1.0)
            w_normalized = where(contact_mask, w_i / w_mean_safe, 0.0)

            scale_factor = where(
                node_initial_springs > 0, active_counts / node_initial_springs, 0.0
            )
            v_max = dx / dt

        # 1. Irreversible failure updates & internal forces (Spring + Stiffness Damping)
        if backend.BACKEND == "numba" and backend.HAS_NUMBA:
            spring_stiff_damp_forces, _step_fracture_energy, step_stiff_damp_power = (
                numba_step_internal_forces_and_failures(
                    positions,
                    velocities,
                    grid_springs,
                    grid_stiffnesses,
                    grid_rest_lengths,
                    grid_failed,
                    grid_tension_only,
                    node_spring_offsets,
                    node_spring_ids,
                    node_spring_signs,
                    damage_onset_strain,
                    failure_strain,
                    rayleigh_beta,
                )
            )
            damp_dissipated += step_stiff_damp_power * dt
        else:
            # Recompute vectorized values for force updates
            p1 = positions[grid_springs[:, 0]]
            p2 = positions[grid_springs[:, 1]]
            diff = p2 - p1
            lengths = sqrt(sum(diff**2, axis=1))
            strains = (lengths - grid_rest_lengths) / grid_rest_lengths
            newly_failed = (~grid_failed) & (strains > failure_strain)
            grid_failed = grid_failed | newly_failed

            # Spring forces (with progressive damage)
            lengths_safe = where(lengths == 0.0, 1.0, lengths)
            denom = failure_strain - damage_onset_strain
            denom_safe = where(denom == 0.0, 1.0, denom)
            damage = minimum(maximum((strains - damage_onset_strain) / denom_safe, 0.0), 1.0)
            effective_k = grid_stiffnesses * (1.0 - damage)
            f_mag = effective_k * strains * grid_rest_lengths
            f_mag = where(grid_tension_only & (strains < 0.0), 0.0, f_mag)
            f_mag = where(grid_failed, 0.0, f_mag)

            # Stiffness-proportional damping
            v1 = velocities[grid_springs[:, 0]]
            v2 = velocities[grid_springs[:, 1]]
            v_rel = v2 - v1
            unit_axes = diff / lengths_safe[:, np.newaxis]
            v_proj = sum(v_rel * unit_axes, axis=1)

            effective_beta = 0.0 if use_viscous else rayleigh_beta
            stiff_damp_mag = effective_beta * grid_stiffnesses * v_proj
            stiff_damp_mag = where(grid_failed, 0.0, stiff_damp_mag)

            # Dissipation power
            stiff_damp_power = sum(stiff_damp_mag * v_proj)
            damp_dissipated += stiff_damp_power * dt

            total_mag = f_mag + stiff_damp_mag
            force_vecs = total_mag[:, np.newaxis] * unit_axes

            spring_stiff_damp_forces = zeros(positions.shape, dtype=positions.dtype)
            spring_stiff_damp_forces = scatter_add(
                spring_stiff_damp_forces, grid_springs[:, 0], force_vecs
            )
            spring_stiff_damp_forces = scatter_add(
                spring_stiff_damp_forces, grid_springs[:, 1], -force_vecs
            )

        # 2. Projectile Contact Forces (IDW Distribution)
        direction = where(
            strike_direction != 0.0, strike_direction, where(proj_velocity[2] < 0.0, -1.0, 1.0)
        )
        penetration = maximum(0.0, (proj_position[2] - positions[:, 2]) * direction)
        f_i = k_penalty * w_normalized * penetration * scale_factor
        proj_forces = stack_z(f_i * direction)

        # 3. Inter-ply Contact Forces (Checkout Mode)
        interply_forces, _ = compute_interply_contact_forces(
            positions, n_nodes_per_layer, n_plies, t_ply, k_penalty, active_counts
        )

        # 4. Mass-proportional Rayleigh Damping or Legacy Viscous Damping
        if use_viscous:
            f_mass_damp = -rayleigh_alpha * velocities
        else:
            f_mass_damp = -rayleigh_alpha * masses_col * velocities
        p_mass_damp = sum(f_mass_damp * velocities)
        damp_dissipated += -p_mass_damp * dt

        # Net acceleration calculation
        net_forces = (
            spring_stiff_damp_forces
            + proj_forces
            + interply_forces
            + f_mass_damp
            + nodal_external_forces
        )

        # Reset forces/velocities on boundary clamped nodes
        net_forces = clamp_boundary(net_forces, boundary_mask)
        velocities = clamp_boundary(velocities, boundary_mask)

        # Integrate node dynamics (leapfrog Verlet)
        accel = net_forces / masses_col
        velocities = velocities + accel * dt

        # CFL velocity clamping (Part B.4)
        if backend.BACKEND == "numba" and backend.HAS_NUMBA:
            velocities, excess_ke = numba_clamp_velocities(velocities, grid_masses, v_max)
            clamp_dissipated += excess_ke
        else:
            v_mag = sqrt(sum(velocities**2, axis=1))
            scale = where(v_mag > v_max, v_max / maximum(v_mag, 1e-30), 1.0)
            excess_ke = sum(0.5 * grid_masses * (v_mag**2) * (1.0 - scale**2))
            clamp_dissipated += excess_ke
            velocities = velocities * scale[:, np.newaxis]

        velocities = clamp_boundary(velocities, boundary_mask)
        positions = positions + velocities * dt

        # Integrate rigid-body projectile kinematics
        proj_reaction_force = -sum(proj_forces, axis=0)
        proj_accel = proj_reaction_force / proj_mass
        proj_velocity = proj_velocity + proj_accel * dt
        proj_position = proj_position + proj_velocity * dt

        t_sim += dt

        # Update progressive failure dissipated energy continuously
        if backend.BACKEND == "numba" and backend.HAS_NUMBA:
            failure_dissipated = numba_compute_failure_dissipated(
                positions,
                grid_springs,
                grid_stiffnesses,
                grid_rest_lengths,
                grid_failed,
                damage_onset_strain,
                failure_strain,
                fracture_energy_multiplier,
            )
        else:
            p1_post = positions[grid_springs[:, 0]]
            p2_post = positions[grid_springs[:, 1]]
            diff_post = p2_post - p1_post
            lengths_post = sqrt(sum(diff_post**2, axis=1))
            x_val = (lengths_post - grid_rest_lengths) / grid_rest_lengths

            denom = failure_strain - damage_onset_strain
            denom_safe = where(denom == 0.0, 1.0, denom)
            damage_post = minimum(maximum((x_val - damage_onset_strain) / denom_safe, 0.0), 1.0)
            eff_k_post = grid_stiffnesses * (1.0 - damage_post)
            eff_k_post = where(grid_failed, 0.0, eff_k_post)

            w_input = where(
                x_val < damage_onset_strain,
                0.5 * grid_stiffnesses * (x_val * grid_rest_lengths) ** 2,
                (grid_stiffnesses * grid_rest_lengths**2 / 6.0)
                * (x_val**2 + x_val * damage_onset_strain + damage_onset_strain**2),
            )
            se_actual = 0.5 * eff_k_post * (x_val * grid_rest_lengths) ** 2

            spring_dissipated = where(
                x_val >= damage_onset_strain,
                fracture_energy_multiplier * (w_input - se_actual),
                0.0,
            )
            w_failed = (grid_stiffnesses * grid_rest_lengths**2 / 6.0) * (
                failure_strain**2 + failure_strain * damage_onset_strain + damage_onset_strain**2
            )
            spring_dissipated = where(
                grid_failed, fracture_energy_multiplier * w_failed, spring_dissipated
            )
            failure_dissipated = sum(spring_dissipated)

        # Periodically capture frame telemetry
        step_1indexed = step + 1
        if step_1indexed % save_interval == 0:
            frame_idx = (step_1indexed // save_interval) - 1

            # Energy Metrics S6.5.5
            p1_telem = positions[grid_springs[:, 0]]
            p2_telem = positions[grid_springs[:, 1]]
            strains_telem = sqrt(sum((p2_telem - p1_telem) ** 2, axis=1))
            strains_telem = (strains_telem - grid_rest_lengths) / grid_rest_lengths

            ke = compute_kinetic_energy(velocities, grid_masses)

            # Use degraded strain energy S7.14
            denom = failure_strain - damage_onset_strain
            denom_safe = where(denom == 0.0, 1.0, denom)
            damage_telem = minimum(
                maximum((strains_telem - damage_onset_strain) / denom_safe, 0.0), 1.0
            )
            se = compute_strain_energy(
                strains_telem, grid_stiffnesses, grid_rest_lengths, grid_failed, damage_telem
            )

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
        failure_dissipated,
        clamp_dissipated,
        t_sim,
        hist_positions,
        hist_failed,
        hist_proj_pos,
        hist_time,
        hist_ke,
        hist_se,
        hist_proj_ke,
    )
