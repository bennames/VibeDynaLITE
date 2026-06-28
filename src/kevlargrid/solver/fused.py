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
    sum,
    where,
    zeros,
)
from kevlargrid.solver.energy import compute_kinetic_energy, compute_strain_energy
from kevlargrid.solver.failure import scale_failure_strain
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


@backend.jit(fastmath=True, parallel=False)
def numba_q_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


@backend.jit(fastmath=True, parallel=False)
def numba_q_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_v = np.array([0.0, v[0], v[1], v[2]], dtype=np.float64)
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)
    q_rot = numba_q_mul(numba_q_mul(q, q_v), q_conj)
    return q_rot[1:]


@backend.jit(fastmath=True, parallel=False)
def numba_eval_sdf(
    p: np.ndarray,
    shape_code: int,
    R: float,
    L: float,
    R_e: float,
    R_og: float,
    L_body: float,
    L_nose: float,
    z_com: float,
    S: float,
    c_r: float,
    c_t: float,
    twist_deg: float,
    thickness_ratio: float,
    R_tip: float,
    y_com: float,
    w_h: float,
    t_h: float,
) -> float:
    x, y, z = p[0], p[1], p[2]
    val = 0.0
    if shape_code == 1:  # sphere
        val = np.sqrt(x**2 + y**2 + z**2) - R
    elif shape_code == 2:  # cylinder
        d_cyl = np.sqrt(x**2 + y**2) - (R - R_e)
        d_len = np.abs(z) - (L / 2.0 - R_e)
        c_cyl = d_cyl if d_cyl > 0.0 else 0.0
        c_len = d_len if d_len > 0.0 else 0.0
        ext_d = np.sqrt(c_cyl**2 + c_len**2)
        max_d = d_cyl if d_cyl > d_len else d_len
        int_d = 0.0 if max_d > 0.0 else max_d
        val = ext_d + int_d - R_e
    elif shape_code == 3:  # bullet
        z_geom = z + z_com
        r = np.sqrt(x**2 + y**2)
        if z_geom < 0.0:
            d_cyl = r - R
            d_cap = -z_geom - L_body
            val = d_cyl if d_cyl > d_cap else d_cap
        else:
            if z_geom > L_nose:
                val = np.sqrt(r**2 + (z_geom - L_nose) ** 2)
            else:
                r_c = R - R_og
                dist_to_center = np.sqrt((r - r_c) ** 2 + z_geom**2)
                val = dist_to_center - R_og
    elif shape_code == 4:  # propeller
        y_geom = y + y_com
        if y_geom > S - R_tip:
            val = np.sqrt(x**2 + (y_geom - (S - R_tip)) ** 2 + z**2) - R_tip
        elif y_geom < 0.0:
            u = (x + c_r / 2.0) / c_r
            u_temp = u if u > 0.0 else 0.0
            u_clamped = u_temp if u_temp < 1.0 else 1.0
            t = (
                5.0
                * (thickness_ratio / 100.0)
                * (
                    0.2969 * np.sqrt(u_clamped)
                    - 0.1260 * u_clamped
                    - 0.3516 * (u_clamped**2)
                    + 0.2843 * (u_clamped**3)
                    - 0.1015 * (u_clamped**4)
                )
                * c_r
            )
            t_half = t / 2.0
            half_t = t_half if t_half > R_tip else R_tip
            d_slice = np.abs(z) - half_t
            val = -y_geom if -y_geom > d_slice else d_slice
        else:
            c = c_r + (y_geom / S) * (c_t - c_r)
            theta = np.radians(twist_deg) * (y_geom / S)
            xr = x * np.cos(theta) + z * np.sin(theta)
            zr = -x * np.sin(theta) + z * np.cos(theta)

            u = (xr + c / 2.0) / c
            u_temp = u if u > 0.0 else 0.0
            u_clamped = u_temp if u_temp < 1.0 else 1.0
            t = (
                5.0
                * (thickness_ratio / 100.0)
                * (
                    0.2969 * np.sqrt(u_clamped)
                    - 0.1260 * u_clamped
                    - 0.3516 * (u_clamped**2)
                    + 0.2843 * (u_clamped**3)
                    - 0.1015 * (u_clamped**4)
                )
                * c
            )
            t_half = t / 2.0
            half_t = t_half if t_half > R_tip else R_tip

            if xr < -c / 2.0 + R_tip:
                dist_le = np.sqrt((xr - (-c / 2.0 + R_tip)) ** 2 + zr**2)
                val = dist_le - R_tip
            elif xr > c / 2.0 - R_tip:
                dist_te = np.sqrt((xr - (c / 2.0 - R_tip)) ** 2 + zr**2)
                val = dist_te - R_tip
            else:
                val = np.abs(zr) - half_t
    else:  # box / legacy
        x_min = x if x < w_h else w_h
        x_proj = -w_h if -w_h > x_min else x_min
        y_min = y if y < t_h else t_h
        y_proj = -t_h if -t_h > y_min else y_min
        val = np.sqrt((x - x_proj) ** 2 + (y - y_proj) ** 2 + z**2)
    return val


@backend.jit(fastmath=True, parallel=False)
def numba_eval_sdf_normal(
    p: np.ndarray,
    shape_code: int,
    R: float,
    L: float,
    R_e: float,
    R_og: float,
    L_body: float,
    L_nose: float,
    z_com: float,
    S: float,
    c_r: float,
    c_t: float,
    twist_deg: float,
    thickness_ratio: float,
    R_tip: float,
    y_com: float,
    w_h: float,
    t_h: float,
) -> np.ndarray:
    h = 1e-5
    grad = np.zeros(3, dtype=np.float64)
    p_px = p.copy()
    p_px[0] += h
    p_mx = p.copy()
    p_mx[0] -= h
    grad[0] = (
        numba_eval_sdf(
            p_px,
            shape_code,
            R,
            L,
            R_e,
            R_og,
            L_body,
            L_nose,
            z_com,
            S,
            c_r,
            c_t,
            twist_deg,
            thickness_ratio,
            R_tip,
            y_com,
            w_h,
            t_h,
        )
        - numba_eval_sdf(
            p_mx,
            shape_code,
            R,
            L,
            R_e,
            R_og,
            L_body,
            L_nose,
            z_com,
            S,
            c_r,
            c_t,
            twist_deg,
            thickness_ratio,
            R_tip,
            y_com,
            w_h,
            t_h,
        )
    ) / (2.0 * h)

    p_py = p.copy()
    p_py[1] += h
    p_my = p.copy()
    p_my[1] -= h
    grad[1] = (
        numba_eval_sdf(
            p_py,
            shape_code,
            R,
            L,
            R_e,
            R_og,
            L_body,
            L_nose,
            z_com,
            S,
            c_r,
            c_t,
            twist_deg,
            thickness_ratio,
            R_tip,
            y_com,
            w_h,
            t_h,
        )
        - numba_eval_sdf(
            p_my,
            shape_code,
            R,
            L,
            R_e,
            R_og,
            L_body,
            L_nose,
            z_com,
            S,
            c_r,
            c_t,
            twist_deg,
            thickness_ratio,
            R_tip,
            y_com,
            w_h,
            t_h,
        )
    ) / (2.0 * h)

    p_pz = p.copy()
    p_pz[2] += h
    p_mz = p.copy()
    p_mz[2] -= h
    grad[2] = (
        numba_eval_sdf(
            p_pz,
            shape_code,
            R,
            L,
            R_e,
            R_og,
            L_body,
            L_nose,
            z_com,
            S,
            c_r,
            c_t,
            twist_deg,
            thickness_ratio,
            R_tip,
            y_com,
            w_h,
            t_h,
        )
        - numba_eval_sdf(
            p_mz,
            shape_code,
            R,
            L,
            R_e,
            R_og,
            L_body,
            L_nose,
            z_com,
            S,
            c_r,
            c_t,
            twist_deg,
            thickness_ratio,
            R_tip,
            y_com,
            w_h,
            t_h,
        )
    ) / (2.0 * h)

    norm = np.linalg.norm(grad)
    return grad / norm if norm > 1e-8 else np.array([0.0, 0.0, 1.0], dtype=np.float64)


@backend.jit(parallel=False, fastmath=True)
def numba_compute_spring_forces(
    positions: np.ndarray,
    velocities: np.ndarray,
    springs: np.ndarray,
    effective_k: np.ndarray,
    rest_lengths: np.ndarray,
    tension_only: np.ndarray,
    rayleigh_beta: float,
) -> tuple[np.ndarray, float]:
    n_nodes = len(positions)
    n_springs = len(springs)
    forces = np.zeros((n_nodes, 3), dtype=positions.dtype)
    stiff_damp_energy = 0.0

    for i in range(n_springs):
        k = effective_k[i]
        if k == 0.0:
            continue

        n0 = springs[i, 0]
        n1 = springs[i, 1]

        dx = positions[n1, 0] - positions[n0, 0]
        dy = positions[n1, 1] - positions[n0, 1]
        dz = positions[n1, 2] - positions[n0, 2]

        length = np.sqrt(dx * dx + dy * dy + dz * dz)
        length_safe = length if length != 0.0 else 1.0

        strain = (length - rest_lengths[i]) / rest_lengths[i]

        if tension_only[i] and strain < 0.0:
            continue

        f_mag = k * strain * rest_lengths[i]

        damp_mag = 0.0
        if rayleigh_beta > 0.0:
            dvx = velocities[n1, 0] - velocities[n0, 0]
            dvy = velocities[n1, 1] - velocities[n0, 1]
            dvz = velocities[n1, 2] - velocities[n0, 2]
            v_proj = (dvx * dx + dvy * dy + dvz * dz) / length_safe
            damp_mag = rayleigh_beta * k * v_proj
            stiff_damp_energy += damp_mag * v_proj

        total_mag = f_mag + damp_mag
        f_coeff = total_mag / length_safe

        fx = f_coeff * dx
        fy = f_coeff * dy
        fz = f_coeff * dz

        forces[n0, 0] += fx
        forces[n0, 1] += fy
        forces[n0, 2] += fz

        forces[n1, 0] -= fx
        forces[n1, 1] -= fy
        forces[n1, 2] -= fz

    return forces, stiff_damp_energy


@backend.jit(parallel=True, fastmath=True)
def numba_parallel_compute_spring_forces(
    positions: np.ndarray,
    velocities: np.ndarray,
    springs: np.ndarray,
    effective_k: np.ndarray,
    rest_lengths: np.ndarray,
    tension_only: np.ndarray,
    node_spring_offsets: np.ndarray,
    node_spring_ids: np.ndarray,
    node_spring_signs: np.ndarray,
    rayleigh_beta: float,
) -> tuple[np.ndarray, float]:
    n_springs = len(springs)
    n_nodes = len(positions)
    force_vecs = np.zeros((n_springs, 3), dtype=positions.dtype)
    stiff_damp_energy = 0.0

    for i in numba.prange(n_springs):
        k = effective_k[i]
        if k == 0.0:
            continue
        n0 = springs[i, 0]
        n1 = springs[i, 1]
        dx = positions[n1, 0] - positions[n0, 0]
        dy = positions[n1, 1] - positions[n0, 1]
        dz = positions[n1, 2] - positions[n0, 2]
        length = np.sqrt(dx * dx + dy * dy + dz * dz)
        length_safe = length if length != 0.0 else 1.0
        strain = (length - rest_lengths[i]) / rest_lengths[i]
        if tension_only[i] and strain < 0.0:
            continue
        f_mag = k * strain * rest_lengths[i]
        damp_mag = 0.0
        if rayleigh_beta > 0.0:
            dvx = velocities[n1, 0] - velocities[n0, 0]
            dvy = velocities[n1, 1] - velocities[n0, 1]
            dvz = velocities[n1, 2] - velocities[n0, 2]
            v_proj = (dvx * dx + dvy * dy + dvz * dz) / length_safe
            damp_mag = rayleigh_beta * k * v_proj
            stiff_damp_energy += damp_mag * v_proj
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

    return forces, stiff_damp_energy


@backend.jit(parallel=True, fastmath=True)
def numba_compute_effective_k(
    positions: np.ndarray,
    springs: np.ndarray,
    stiffnesses: np.ndarray,
    rest_lengths: np.ndarray,
    failed: np.ndarray,
    damage_onset_strain: float,
    failure_strain: float,
    grid_damage: np.ndarray,
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
        new_damage = damage if damage > grid_damage[i] else grid_damage[i]
        grid_damage[i] = new_damage
        if new_damage >= 1.0:
            failed[i] = True
            effective_k[i] = 0.0
        else:
            effective_k[i] = stiffnesses[i] * (1.0 - new_damage)
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
    springs: np.ndarray,
    stiffnesses: np.ndarray,
    rest_lengths: np.ndarray,
    failed: np.ndarray,
    grid_damage: np.ndarray,
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
        else:
            D = grid_damage[i]
            if D > 0.0:
                x_peak = x_onset + D * (x_fail - x_onset)
                denom = x_fail - x_onset
                denom_safe = denom if denom != 0.0 else 1.0
                w_diss = (k * L0**2 / (6.0 * denom_safe)) * (x_peak**3 - x_onset**3)
                total_diss += fracture_energy_multiplier * w_diss
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
            damp_mag = rayleigh_beta * effective_k * v_proj
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
        "shape_code",
    ),
)
def _fused_leapfrog_loop_jit(
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
    use_viscous: bool,
    cfl_factor: float,
    grid_damage: np.ndarray,
    proj_quat: np.ndarray,
    proj_omega: np.ndarray,
    shape_code: int,
    proj_radius: float,
    proj_length: float,
    proj_edge_radius: float,
    proj_ogive_multiplier: float,
    proj_span: float,
    proj_root_chord: float,
    proj_tip_chord: float,
    proj_twist: float,
    proj_thickness_ratio: float,
    proj_tip_radius: float,
    proj_z_com: float,
    proj_y_com: float,
    proj_c_damping: float,
    proj_inertia_inv_diag: np.ndarray,
    proj_peak_deceleration: np.ndarray,
    hist_proj_quat: np.ndarray,
    w_h: float,
    t_h: float,
    contact_energy_init: float,
    mu_s: float,
    friction_dissipated_init: float,
):
    n_nodes = len(positions)
    n_springs = len(grid_springs)
    m_frames = max(1, n_steps // save_interval)

    # Pre-allocate history structures (compatible with JIT vector allocations)
    hist_positions = zeros((m_frames, n_nodes, 3), dtype=positions.dtype)
    hist_failed = zeros((m_frames, n_springs), dtype=np.bool_)
    hist_proj_pos = zeros((m_frames, 3), dtype=positions.dtype)
    hist_time = zeros(m_frames, dtype=positions.dtype)
    hist_ke = zeros(m_frames, dtype=positions.dtype)
    hist_se = zeros(m_frames, dtype=positions.dtype)
    hist_proj_ke = zeros(m_frames, dtype=positions.dtype)

    proximity_threshold = dx * 2.0

    damp_dissipated = damp_dissipated_init
    failure_dissipated = failure_dissipated_init
    clamp_dissipated = clamp_dissipated_init
    contact_energy = contact_energy_init
    friction_dissipated = friction_dissipated_init
    t_sim = t_sim_init

    masses_col = grid_masses.reshape(-1, 1)

    accel = zeros((n_nodes, 3), dtype=positions.dtype)
    proj_accel = zeros(3, dtype=np.float64)
    omega_dot = zeros(3, dtype=np.float64)
    proj_reaction_force = zeros(3, dtype=np.float64)
    proj_torque = zeros(3, dtype=np.float64)

    # Precompute ogive parameters if bullet
    R0_val = proj_radius
    R_og_val = R0_val * proj_ogive_multiplier
    L_nose_val = np.sqrt(max(0.0, 2.0 * R_og_val * R0_val - R0_val**2))
    L_body_val = max(0.0, proj_length - L_nose_val)

    # Pre-compute effective_k at step 0 so we can compute initial forces
    if backend.BACKEND == "numba" and backend.HAS_NUMBA:
        effective_k = numba_compute_effective_k(
            positions,
            grid_springs,
            grid_stiffnesses,
            grid_rest_lengths,
            grid_failed,
            damage_onset_strain,
            failure_strain,
            grid_damage,
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
        grid_damage[:] = maximum(grid_damage, damage)
        effective_k = grid_stiffnesses * (1.0 - grid_damage)
        effective_k = where(grid_failed, 0.0, effective_k)

    # Initial force calculation to populate `accel`
    if backend.BACKEND == "numba" and backend.HAS_NUMBA:
        spring_stiff_damp_forces, _ = numba_compute_spring_forces(
            positions,
            velocities,
            grid_springs,
            effective_k,
            grid_rest_lengths,
            grid_tension_only,
            rayleigh_beta,
        )
    else:
        p1 = positions[grid_springs[:, 0]]
        p2 = positions[grid_springs[:, 1]]
        v1 = velocities[grid_springs[:, 0]]
        v2 = velocities[grid_springs[:, 1]]
        diff_p = p2 - p1
        diff_v = v2 - v1
        lengths = sqrt(sum(diff_p**2, axis=1))
        lengths_safe = where(lengths == 0.0, 1.0, lengths)
        dirs = diff_p / lengths_safe[:, np.newaxis]
        strains = (lengths - grid_rest_lengths) / grid_rest_lengths
        f_stiff_mags = effective_k * strains * grid_rest_lengths
        if len(grid_tension_only) > 0:
            f_stiff_mags = where(grid_tension_only & (strains < 0), 0.0, f_stiff_mags)
        v_rel_proj = sum(diff_v * dirs, axis=1)
        f_damp_mags = rayleigh_beta * effective_k * v_rel_proj
        f_mags = f_stiff_mags + f_damp_mags
        f_vecs = f_mags[:, np.newaxis] * dirs
        spring_stiff_damp_forces = zeros((n_nodes, 3), dtype=positions.dtype)
        spring_stiff_damp_forces = scatter_add(spring_stiff_damp_forces, grid_springs[:, 0], f_vecs)
        spring_stiff_damp_forces = scatter_add(
            spring_stiff_damp_forces, grid_springs[:, 1], -f_vecs
        )

    interply_forces, contact_e_step, _ = compute_interply_contact_forces(
        positions,
        n_nodes_per_layer,
        n_plies,
        t_ply,
        k_penalty,
        active_counts=zeros(n_nodes, dtype=positions.dtype) + 1.0,
        velocities=velocities,
        mu_s=mu_s,
        dt=dt,
    )
    contact_energy = contact_e_step

    if use_viscous:
        f_mass_damp = -rayleigh_alpha * velocities
    else:
        f_mass_damp = -rayleigh_alpha * masses_col * velocities

    net_forces = spring_stiff_damp_forces + interply_forces + f_mass_damp + nodal_external_forces
    net_forces = clamp_boundary(net_forces, boundary_mask)
    accel = net_forces / masses_col

    for step in range(n_steps):
        # 1. Update v_half and positions (Velocity Verlet Step 1)
        v_half = velocities + 0.5 * accel * dt
        proj_v_half = proj_velocity + 0.5 * proj_accel * dt
        proj_omega_half = proj_omega + 0.5 * omega_dot * dt

        positions = positions + v_half * dt
        proj_position = proj_position + proj_v_half * dt

        if shape_code > 0:
            omega_q = np.array(
                [0.0, proj_omega_half[0], proj_omega_half[1], proj_omega_half[2]], dtype=np.float64
            )
            q_dot = numba_q_mul(omega_q, proj_quat)
            q_new = proj_quat + 0.5 * dt * q_dot
            q_new_norm = np.linalg.norm(q_new)
            if q_new_norm > 1e-8:
                proj_quat[0] = q_new[0] / q_new_norm
                proj_quat[1] = q_new[1] / q_new_norm
                proj_quat[2] = q_new[2] / q_new_norm
                proj_quat[3] = q_new[3] / q_new_norm

        # 2. Compute Nodal Stiffnesses & CFL Timestep
        if backend.BACKEND == "numba" and backend.HAS_NUMBA:
            effective_k = numba_compute_effective_k(
                positions,
                grid_springs,
                grid_stiffnesses,
                grid_rest_lengths,
                grid_failed,
                damage_onset_strain,
                failure_strain,
                grid_damage,
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
            grid_damage[:] = maximum(grid_damage, damage)
            effective_k = grid_stiffnesses * (1.0 - grid_damage)
            effective_k = where(grid_failed, 0.0, effective_k)
            active_springs = where(grid_failed, 0, 1)
            active_counts = zeros(n_nodes, dtype=positions.dtype)
            active_counts = scatter_add(active_counts, grid_springs[:, 0], active_springs)
            active_counts = scatter_add(active_counts, grid_springs[:, 1], active_springs)

        # Compute contact mask and weights (always needed for force calculation, and optionally CFL)
        if shape_code == 0:
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
        else:
            # dummy initializations to prevent UnboundLocalError/warnings in non-box shapes
            contact_mask = zeros(n_nodes, dtype=np.bool_)
            dists = zeros(n_nodes, dtype=positions.dtype)

        if cfl_factor > 0.0:
            if backend.BACKEND == "numba" and backend.HAS_NUMBA:
                nodal_k_springs = numba_sum_nodal_k_springs(
                    effective_k, node_spring_offsets, node_spring_ids
                )
            else:
                nodal_k_springs = zeros(n_nodes, dtype=positions.dtype)
                nodal_k_springs = scatter_add(nodal_k_springs, grid_springs[:, 0], effective_k)
                nodal_k_springs = scatter_add(nodal_k_springs, grid_springs[:, 1], effective_k)

            # Contact mask & weights
            if shape_code == 0:
                nodal_k_contact = where(contact_mask, k_penalty * w_normalized * scale_factor, 0.0)
            else:
                nodal_k_contact = zeros(n_nodes, dtype=positions.dtype)
                q_conj = np.array(
                    [proj_quat[0], -proj_quat[1], -proj_quat[2], -proj_quat[3]], dtype=np.float64
                )
                max_R = max(proj_radius, max(proj_length, proj_span))
                cutoff = max_R + proximity_threshold
                cutoff_sq = cutoff ** 2
                for i in range(n_nodes):
                    dx_p = positions[i, 0] - proj_position[0]
                    dy_p = positions[i, 1] - proj_position[1]
                    dz_p = positions[i, 2] - proj_position[2]
                    if dx_p**2 + dy_p**2 + dz_p**2 > cutoff_sq:
                        continue
                    P_rel = positions[i] - proj_position
                    P_loc = numba_q_rotate(q_conj, P_rel)
                    dist = numba_eval_sdf(
                        P_loc,
                        shape_code,
                        proj_radius,
                        proj_length,
                        proj_edge_radius,
                        R_og_val,
                        L_body_val,
                        L_nose_val,
                        proj_z_com,
                        proj_span,
                        proj_root_chord,
                        proj_tip_chord,
                        proj_twist,
                        proj_thickness_ratio,
                        proj_tip_radius,
                        proj_y_com,
                        w_h,
                        t_h,
                    )
                    if dist <= proximity_threshold:
                        s_factor = 0.0
                        if node_initial_springs[i] > 0:
                            s_factor = float(active_counts[i]) / float(node_initial_springs[i])
                        else:
                            s_factor = 1.0
                        nodal_k_contact[i] = k_penalty * s_factor

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
            dt_crit = min(sqrt(grid_masses / total_nodal_k))
            dt = cfl_factor * dt_crit
            v_max = dx / dt
        else:
            v_max = dx / dt

        # 3. Calculate internal node forces (stiffness + damping)
        if backend.BACKEND == "numba" and backend.HAS_NUMBA:
            spring_stiff_damp_forces, stiff_damp_e = numba_parallel_compute_spring_forces(
                positions,
                v_half,
                grid_springs,
                effective_k,
                grid_rest_lengths,
                grid_tension_only,
                node_spring_offsets,
                node_spring_ids,
                node_spring_signs,
                rayleigh_beta,
            )
            damp_dissipated += stiff_damp_e * dt
        else:
            p1 = positions[grid_springs[:, 0]]
            p2 = positions[grid_springs[:, 1]]
            v1 = v_half[grid_springs[:, 0]]
            v2 = v_half[grid_springs[:, 1]]
            diff_p = p2 - p1
            diff_v = v2 - v1
            lengths = sqrt(sum(diff_p**2, axis=1))
            lengths_safe = where(lengths == 0.0, 1.0, lengths)
            dirs = diff_p / lengths_safe[:, np.newaxis]
            strains = (lengths - grid_rest_lengths) / grid_rest_lengths
            f_stiff_mags = effective_k * strains * grid_rest_lengths
            if len(grid_tension_only) > 0:
                f_stiff_mags = where(grid_tension_only & (strains < 0), 0.0, f_stiff_mags)
            v_rel_proj = sum(diff_v * dirs, axis=1)
            f_damp_mags = rayleigh_beta * effective_k * v_rel_proj
            f_mags = f_stiff_mags + f_damp_mags
            f_vecs = f_mags[:, np.newaxis] * dirs
            spring_stiff_damp_forces = zeros((n_nodes, 3), dtype=positions.dtype)
            spring_stiff_damp_forces = scatter_add(
                spring_stiff_damp_forces, grid_springs[:, 0], f_vecs
            )
            spring_stiff_damp_forces = scatter_add(
                spring_stiff_damp_forces, grid_springs[:, 1], -f_vecs
            )
            if rayleigh_beta > 0.0:
                damp_dissipated += np.sum(f_damp_mags * v_rel_proj) * dt

        # Contact forces
        proj_forces = zeros((n_nodes, 3), dtype=positions.dtype)
        proj_reaction_force = zeros(3, dtype=np.float64)
        proj_torque = zeros(3, dtype=np.float64)
        proj_contact_e_step = 0.0

        if shape_code == 0:
            if sum(contact_mask) > 0:
                d_safe = maximum(dists, 1e-4)
                nx = (positions[:, 0] - x_proj) / d_safe
                ny = (positions[:, 1] - y_proj) / d_safe
                nz = (positions[:, 2] - proj_position[2]) / d_safe
                delta = proximity_threshold - dists
                f_mags = k_penalty * delta * w_normalized * scale_factor * strike_direction
                proj_forces[:, 0] = where(contact_mask, f_mags * nx, 0.0)
                proj_forces[:, 1] = where(contact_mask, f_mags * ny, 0.0)
                proj_forces[:, 2] = where(contact_mask, f_mags * nz, 0.0)
                proj_reaction_force[0] = -sum(proj_forces[:, 0])
                proj_reaction_force[1] = -sum(proj_forces[:, 1])
                proj_reaction_force[2] = -sum(proj_forces[:, 2])
                
                # Projectile contact potential energy
                proj_pe_elements = where(
                    contact_mask,
                    0.5 * k_penalty * w_normalized * delta * delta * scale_factor,
                    0.0,
                )
                proj_contact_e_step += np.sum(proj_pe_elements)
        else:
            q_conj = np.array(
                [proj_quat[0], -proj_quat[1], -proj_quat[2], -proj_quat[3]], dtype=np.float64
            )
            max_R = max(proj_radius, max(proj_length, proj_span))
            cutoff = max_R + proximity_threshold
            cutoff_sq = cutoff ** 2
            for i in range(n_nodes):
                dx_p = positions[i, 0] - proj_position[0]
                dy_p = positions[i, 1] - proj_position[1]
                dz_p = positions[i, 2] - proj_position[2]
                if dx_p**2 + dy_p**2 + dz_p**2 > cutoff_sq:
                    continue
                P_rel = positions[i] - proj_position
                P_loc = numba_q_rotate(q_conj, P_rel)
                dist = numba_eval_sdf(
                    P_loc,
                    shape_code,
                    proj_radius,
                    proj_length,
                    proj_edge_radius,
                    R_og_val,
                    L_body_val,
                    L_nose_val,
                    proj_z_com,
                    proj_span,
                    proj_root_chord,
                    proj_tip_chord,
                    proj_twist,
                    proj_thickness_ratio,
                    proj_tip_radius,
                    proj_y_com,
                    0.0,
                    0.0,
                )
                delta = -dist
                if delta > 0.0:
                    n_loc = numba_eval_sdf_normal(
                        P_loc,
                        shape_code,
                        proj_radius,
                        proj_length,
                        proj_edge_radius,
                        R_og_val,
                        L_body_val,
                        L_nose_val,
                        proj_z_com,
                        proj_span,
                        proj_root_chord,
                        proj_tip_chord,
                        proj_twist,
                        proj_thickness_ratio,
                        proj_tip_radius,
                        proj_y_com,
                        0.0,
                        0.0,
                    )
                    n_world = numba_q_rotate(proj_quat, n_loc)

                    # Compute relative velocity and contact damping
                    v_proj_point = proj_v_half + np.array(
                        [
                            proj_omega_half[1] * P_rel[2] - proj_omega_half[2] * P_rel[1],
                            proj_omega_half[2] * P_rel[0] - proj_omega_half[0] * P_rel[2],
                            proj_omega_half[0] * P_rel[1] - proj_omega_half[1] * P_rel[0],
                        ],
                        dtype=np.float64,
                    )

                    v_rel = v_half[i] - v_proj_point
                    delta_dot = -(
                        v_rel[0] * n_world[0] + v_rel[1] * n_world[1] + v_rel[2] * n_world[2]
                    )

                    f_mag = k_penalty * delta + proj_c_damping * delta_dot
                    if f_mag < 0.0:
                        f_mag = 0.0

                    node_scale_factor = 1.0
                    if node_initial_springs[i] > 0:
                        node_scale_factor = float(active_counts[i]) / float(node_initial_springs[i])

                    F_contact = f_mag * n_world * node_scale_factor
                    proj_forces[i, 0] = F_contact[0]
                    proj_forces[i, 1] = F_contact[1]
                    proj_forces[i, 2] = F_contact[2]
                    
                    proj_contact_e_step += 0.5 * k_penalty * delta * delta * node_scale_factor

                    proj_reaction_force[0] -= F_contact[0]
                    proj_reaction_force[1] -= F_contact[1]
                    proj_reaction_force[2] -= F_contact[2]

                    # Surface projection for torque moment arm
                    P_contact = P_rel - delta * n_world
                    proj_torque[0] += P_contact[1] * (-F_contact[2]) - P_contact[2] * (
                        -F_contact[1]
                    )
                    proj_torque[1] += P_contact[2] * (-F_contact[0]) - P_contact[0] * (
                        -F_contact[2]
                    )
                    proj_torque[2] += P_contact[0] * (-F_contact[1]) - P_contact[1] * (
                        -F_contact[0]
                    )

                    # Projectile 6-DOF contact friction
                    if mu_s > 0.0:
                        v_rel_dot_n = v_rel[0] * n_world[0] + v_rel[1] * n_world[1] + v_rel[2] * n_world[2]
                        v_tang = np.array([
                            v_rel[0] - v_rel_dot_n * n_world[0],
                            v_rel[1] - v_rel_dot_n * n_world[1],
                            v_rel[2] - v_rel_dot_n * n_world[2],
                        ], dtype=np.float64)
                        v_rel_sq = v_tang[0]**2 + v_tang[1]**2 + v_tang[2]**2
                        
                        v0 = 0.01
                        denom = np.sqrt(v_rel_sq + v0**2)
                        
                        f_fric_mag = mu_s * f_mag * node_scale_factor
                        F_friction = -f_fric_mag * (v_tang / denom)
                        
                        proj_forces[i, 0] += F_friction[0]
                        proj_forces[i, 1] += F_friction[1]
                        proj_forces[i, 2] += F_friction[2]
                        
                        proj_reaction_force[0] -= F_friction[0]
                        proj_reaction_force[1] -= F_friction[1]
                        proj_reaction_force[2] -= F_friction[2]
                        
                        proj_torque[0] += P_contact[1] * (-F_friction[2]) - P_contact[2] * (-F_friction[1])
                        proj_torque[1] += P_contact[2] * (-F_friction[0]) - P_contact[0] * (-F_friction[2])
                        proj_torque[2] += P_contact[0] * (-F_friction[1]) - P_contact[1] * (-F_friction[0])
                        
                        friction_dissipated += f_fric_mag * (v_rel_sq / denom) * dt

        # Apply Coulomb friction to legacy 3-DOF box projectile
        if shape_code == 0 and mu_s > 0.0 and sum(contact_mask) > 0:
            for i in range(n_nodes):
                if contact_mask[i]:
                    nx_i = nx[i]
                    ny_i = ny[i]
                    nz_i = nz[i]
                    f_N = abs(f_mags[i])
                    vx_rel = v_half[i, 0] - proj_v_half[0]
                    vy_rel = v_half[i, 1] - proj_v_half[1]
                    vz_rel = v_half[i, 2] - proj_v_half[2]
                    v_rel_dot_n = vx_rel * nx_i + vy_rel * ny_i + vz_rel * nz_i
                    v_tang_x = vx_rel - v_rel_dot_n * nx_i
                    v_tang_y = vy_rel - v_rel_dot_n * ny_i
                    v_tang_z = vz_rel - v_rel_dot_n * nz_i
                    v_rel_sq = v_tang_x**2 + v_tang_y**2 + v_tang_z**2
                    v0 = 0.01
                    denom = np.sqrt(v_rel_sq + v0**2)
                    f_fric_mag = mu_s * f_N
                    f_fric_x = -f_fric_mag * (v_tang_x / denom)
                    f_fric_y = -f_fric_mag * (v_tang_y / denom)
                    f_fric_z = -f_fric_mag * (v_tang_z / denom)
                    
                    proj_forces[i, 0] += f_fric_x
                    proj_forces[i, 1] += f_fric_y
                    proj_forces[i, 2] += f_fric_z
                    proj_reaction_force[0] -= f_fric_x
                    proj_reaction_force[1] -= f_fric_y
                    proj_reaction_force[2] -= f_fric_z
                    
                    friction_dissipated += f_fric_mag * (v_rel_sq / denom) * dt

        interply_forces, contact_e_step, fric_diss_step = compute_interply_contact_forces(
            positions,
            n_nodes_per_layer,
            n_plies,
            t_ply,
            k_penalty,
            active_counts=active_counts,
            velocities=v_half,
            mu_s=mu_s,
            dt=dt,
        )
        contact_energy = contact_e_step + proj_contact_e_step
        friction_dissipated += fric_diss_step

        if use_viscous:
            f_mass_damp = -rayleigh_alpha * v_half
        else:
            f_mass_damp = -rayleigh_alpha * masses_col * v_half

        p_mass_damp = sum(f_mass_damp * v_half)
        damp_dissipated += -p_mass_damp * dt

        net_forces = (
            spring_stiff_damp_forces
            + proj_forces
            + interply_forces
            + f_mass_damp
            + nodal_external_forces
        )
        net_forces = clamp_boundary(net_forces, boundary_mask)

        # 4. Update Accelerations and Finalize v_full (Velocity Verlet Step 2)
        accel = net_forces / masses_col
        proj_accel = proj_reaction_force / proj_mass

        if shape_code > 0:
            omega_dot[0] = proj_inertia_inv_diag[0] * proj_torque[0]
            omega_dot[1] = proj_inertia_inv_diag[1] * proj_torque[1]
            omega_dot[2] = proj_inertia_inv_diag[2] * proj_torque[2]

        velocities = v_half + 0.5 * accel * dt

        # CFL velocity clamping (Part B.4)
        v_full_mag = sqrt(sum(velocities**2, axis=1))
        clamp_mask = (v_full_mag > v_max) & ~boundary_mask
        if sum(clamp_mask) > 0:
            e_before = 0.5 * masses_col * v_full_mag**2
            velocities = where(
                clamp_mask[:, np.newaxis],
                velocities * (v_max / (v_full_mag + 1e-15))[:, np.newaxis],
                velocities,
            )
            v_after = sqrt(sum(velocities**2, axis=1))
            e_after = 0.5 * masses_col * v_after**2
            clamp_dissipated += sum(e_before - e_after)

        proj_velocity = proj_v_half + 0.5 * proj_accel * dt
        if shape_code > 0:
            proj_omega[:] = proj_omega_half + 0.5 * omega_dot * dt

        # 5. Irreversible Continuum Damage Mechanics (CDM)
        if backend.BACKEND == "numba" and backend.HAS_NUMBA:
            effective_k = numba_compute_effective_k(
                positions,
                grid_springs,
                grid_stiffnesses,
                grid_rest_lengths,
                grid_failed,
                damage_onset_strain,
                failure_strain,
                grid_damage,
            )
        else:
            p1_d = positions[grid_springs[:, 0]]
            p2_d = positions[grid_springs[:, 1]]
            lengths_d = sqrt(sum((p2_d - p1_d) ** 2, axis=1))
            strains_d = (lengths_d - grid_rest_lengths) / grid_rest_lengths
            denom = failure_strain - damage_onset_strain
            denom_safe = where(denom == 0.0, 1.0, denom)
            damage_d = minimum(
                maximum((strains_d - damage_onset_strain) / denom_safe, 0.0), 1.0
            )
            grid_damage[:] = maximum(grid_damage, damage_d)
            grid_failed[:] = grid_damage >= 1.0
            effective_k = grid_stiffnesses * (1.0 - grid_damage)
            effective_k = where(grid_failed, 0.0, effective_k)

        accel_mag = (
            np.sqrt(
                proj_reaction_force[0] ** 2
                + proj_reaction_force[1] ** 2
                + proj_reaction_force[2] ** 2
            )
            / proj_mass
        )
        accel_mag_g = accel_mag / 9.80665
        if accel_mag_g > proj_peak_deceleration[0]:
            proj_peak_deceleration[0] = accel_mag_g

        t_sim += dt

        if backend.BACKEND == "numba" and backend.HAS_NUMBA:
            failure_dissipated = numba_compute_failure_dissipated(
                grid_springs,
                grid_stiffnesses,
                grid_rest_lengths,
                grid_failed,
                grid_damage,
                damage_onset_strain,
                failure_strain,
                fracture_energy_multiplier,
            )
        else:
            # Python/NumPy fallback failure energy tracking
            k = grid_stiffnesses
            L0 = grid_rest_lengths
            x_onset = damage_onset_strain
            x_fail = failure_strain
            D = grid_damage
            
            w_failed = (k * L0**2 / 6.0) * (x_fail**2 + x_fail * x_onset + x_onset**2)
            denom_val = x_fail - x_onset
            denom_safe_val = 1.0 if denom_val == 0.0 else denom_val
            x_peak = x_onset + D * (x_fail - x_onset)
            w_diss = (k * L0**2 / (6.0 * denom_safe_val)) * (x_peak**3 - x_onset**3)
            
            diss_arr = np.where(grid_failed, w_failed, np.where(D > 0.0, w_diss, 0.0))
            failure_dissipated = float(np.sum(fracture_energy_multiplier * diss_arr))

        if step % save_interval == 0:
            frame_idx = step // save_interval
            p1_telem = positions[grid_springs[:, 0]]
            p2_telem = positions[grid_springs[:, 1]]
            strains_telem = sqrt(sum((p2_telem - p1_telem) ** 2, axis=1))
            strains_telem = (strains_telem - grid_rest_lengths) / grid_rest_lengths
            ke = compute_kinetic_energy(velocities, grid_masses)
            denom = failure_strain - damage_onset_strain
            denom_safe = where(denom == 0.0, 1.0, denom)
            damage_telem = minimum(
                maximum((strains_telem - damage_onset_strain) / denom_safe, 0.0), 1.0
            )
            se = compute_strain_energy(
                strains_telem,
                grid_stiffnesses,
                grid_rest_lengths,
                grid_failed,
                damage_telem,
                grid_tension_only,
            )

            proj_rot_ke = 0.0
            if shape_code > 0:
                proj_rot_ke = 0.5 * (
                    (1.0 / proj_inertia_inv_diag[0]) * proj_omega[0] ** 2
                    + (1.0 / proj_inertia_inv_diag[1]) * proj_omega[1] ** 2
                    + (1.0 / proj_inertia_inv_diag[2]) * proj_omega[2] ** 2
                )
            proj_ke = 0.5 * proj_mass * sum(proj_velocity**2) + proj_rot_ke

            hist_positions = set_index_3d(hist_positions, frame_idx, positions)
            hist_failed = set_index_2d_bool(hist_failed, frame_idx, grid_failed)
            hist_proj_pos = set_index_2d_float(hist_proj_pos, frame_idx, proj_position)
            hist_time = set_index_1d(hist_time, frame_idx, t_sim)
            hist_ke = set_index_1d(hist_ke, frame_idx, ke)
            hist_se = set_index_1d(
                hist_se, frame_idx, se + contact_energy
            )  # Propagate contact energy to total SE
            hist_proj_ke = set_index_1d(hist_proj_ke, frame_idx, proj_ke)
            hist_proj_quat[frame_idx, 0] = proj_quat[0]
            hist_proj_quat[frame_idx, 1] = proj_quat[1]
            hist_proj_quat[frame_idx, 2] = proj_quat[2]
            hist_proj_quat[frame_idx, 3] = proj_quat[3]

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
        contact_energy,
        friction_dissipated,
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
    grid_damage: np.ndarray | None = None,
    proj_quat: np.ndarray | None = None,
    proj_omega: np.ndarray | None = None,
    proj_shape_type: str = "box",
    proj_radius: float = 0.005,
    proj_length: float = 0.01,
    proj_edge_radius: float = 0.0,
    proj_ogive_multiplier: float = 2.0,
    proj_span: float = 0.05,
    proj_root_chord: float = 0.01,
    proj_tip_chord: float = 0.005,
    proj_twist: float = 15.0,
    proj_thickness_ratio: float = 12.0,
    proj_tip_radius: float = 0.002,
    proj_z_com: float = 0.0,
    proj_y_com: float = 0.0,
    proj_c_damping: float = 0.0,
    proj_inertia_inv: np.ndarray | None = None,
    proj_peak_deceleration: np.ndarray | None = None,
    hist_proj_quat: np.ndarray | None = None,
    contact_energy_init: float = 0.0,
    mu_s: float = 0.0,
    friction_dissipated_init: float = 0.0,
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
    float,  # contact_energy
    float,  # friction_dissipated
]:
    n_springs = len(grid_springs)
    if grid_damage is None:
        grid_damage = np.zeros(n_springs, dtype=np.float64)
    if proj_quat is None:
        proj_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if proj_omega is None:
        proj_omega = np.zeros(3, dtype=np.float64)
    if proj_inertia_inv is None:
        proj_inertia_inv_diag = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    else:
        proj_inertia_inv_diag = (
            np.diagonal(proj_inertia_inv) if proj_inertia_inv.ndim == 2 else proj_inertia_inv
        )
    if proj_peak_deceleration is None:
        proj_peak_deceleration = np.zeros(1, dtype=np.float64)
    if hist_proj_quat is None:
        hist_proj_quat = np.zeros((max(1, n_steps // save_interval), 4), dtype=np.float64)

    # Map shape type to integer code
    shape_map = {"box": 0, "sphere": 1, "cylinder": 2, "bullet": 3, "propeller": 4}
    shape_code = shape_map.get(proj_shape_type.lower(), 0)

    # Bazant Regularization
    failure_strain_eff = scale_failure_strain(failure_strain, dx)
    # Scale damage onset strain proportionally
    if failure_strain > 0.0:
        ratio = damage_onset_strain / failure_strain
        damage_onset_strain_eff = failure_strain_eff * ratio
    else:
        damage_onset_strain_eff = damage_onset_strain

    return _fused_leapfrog_loop_jit(
        positions,
        velocities,
        grid_springs,
        grid_stiffnesses,
        grid_rest_lengths,
        grid_failed,
        grid_masses,
        grid_tension_only,
        boundary_mask,
        nodal_external_forces,
        proj_position,
        proj_velocity,
        proj_mass,
        n_plies,
        n_nodes_per_layer,
        t_ply,
        dx,
        k_penalty,
        rayleigh_alpha,
        rayleigh_beta,
        failure_strain_eff,
        damage_onset_strain_eff,
        fracture_energy_multiplier,
        dt,
        n_steps,
        save_interval,
        damp_dissipated_init,
        failure_dissipated_init,
        clamp_dissipated_init,
        t_sim_init,
        strike_direction,
        node_initial_springs,
        node_spring_offsets,
        node_spring_ids,
        node_spring_signs,
        use_viscous,
        cfl_factor,
        grid_damage,
        proj_quat,
        proj_omega,
        shape_code,
        proj_radius,
        proj_length,
        proj_edge_radius,
        proj_ogive_multiplier,
        proj_span,
        proj_root_chord,
        proj_tip_chord,
        proj_twist,
        proj_thickness_ratio,
        proj_tip_radius,
        proj_z_com,
        proj_y_com,
        proj_c_damping,
        proj_inertia_inv_diag,
        proj_peak_deceleration,
        hist_proj_quat,
        proj_blade_width / 2.0,
        proj_edge_thickness / 2.0,
        contact_energy_init,
        mu_s,
        friction_dissipated_init,
    )
