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
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    vx, vy, vz = v[0], v[1], v[2]

    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)

    res = np.zeros(3, dtype=v.dtype)
    res[0] = vx + qw * tx + (qy * tz - qz * ty)
    res[1] = vy + qw * ty + (qz * tx - qx * tz)
    res[2] = vz + qw * tz + (qx * ty - qy * tx)
    return res


@backend.jit(fastmath=True, parallel=False)
def numba_q_rotate_inplace(q: np.ndarray, v: np.ndarray, out: np.ndarray) -> None:
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    vx, vy, vz = v[0], v[1], v[2]

    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)

    out[0] = vx + qw * tx + (qy * tz - qz * ty)
    out[1] = vy + qw * ty + (qz * tx - qx * tz)
    out[2] = vz + qw * tz + (qx * ty - qy * tx)


@backend.jit(fastmath=True, parallel=False, inline="never")
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
        h_h = L / 2.0 if L > 0.0 else 0.005
        d_x = np.abs(x) - w_h
        d_y = np.abs(y) - t_h
        d_z = np.abs(z) - h_h

        # External distance
        dx_pos = d_x if d_x > 0.0 else 0.0
        dy_pos = d_y if d_y > 0.0 else 0.0
        dz_pos = d_z if d_z > 0.0 else 0.0
        ext_dist = np.sqrt(dx_pos**2 + dy_pos**2 + dz_pos**2)

        # Internal distance
        max_d = d_x if d_x > d_y else d_y
        max_d = max_d if max_d > d_z else d_z
        int_dist = max_d if max_d < 0.0 else 0.0

        val = ext_dist + int_dist
    return val


@backend.jit(fastmath=True, parallel=False, inline="never")
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
    stiffnesses: np.ndarray,
    rest_lengths: np.ndarray,
    tension_only: np.ndarray,
    rayleigh_beta: float,
    failed: np.ndarray,
    spring_failed_step: np.ndarray,
    current_step: int,
    erosion_softening_steps: int,
) -> tuple[np.ndarray, float]:
    n_nodes = len(positions)
    n_springs = len(springs)
    forces = np.zeros((n_nodes, 3), dtype=positions.dtype)
    stiff_damp_energy = 0.0

    for i in range(n_springs):
        k_soft = effective_k[i]
        if k_soft == 0.0:
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

        f_mag = k_soft * strain * rest_lengths[i]

        damp_mag = 0.0
        if rayleigh_beta > 0.0 and not failed[i]:
            dvx = velocities[n1, 0] - velocities[n0, 0]
            dvy = velocities[n1, 1] - velocities[n0, 1]
            dvz = velocities[n1, 2] - velocities[n0, 2]
            v_proj = (dvx * dx + dvy * dy + dvz * dz) / length_safe
            damp_mag = rayleigh_beta * k_soft * v_proj
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
    stiffnesses: np.ndarray,
    rest_lengths: np.ndarray,
    tension_only: np.ndarray,
    node_spring_offsets: np.ndarray,
    node_spring_ids: np.ndarray,
    node_spring_signs: np.ndarray,
    rayleigh_beta: float,
    failed: np.ndarray,
    spring_failed_step: np.ndarray,
    current_step: int,
    erosion_softening_steps: int,
) -> tuple[np.ndarray, float]:
    n_springs = len(springs)
    n_nodes = len(positions)
    force_vecs = np.zeros((n_springs, 3), dtype=positions.dtype)
    stiff_damp_energy = 0.0

    for i in numba.prange(n_springs):
        k_soft = effective_k[i]
        if k_soft == 0.0:
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

        f_mag = k_soft * strain * rest_lengths[i]
        damp_mag = 0.0
        if rayleigh_beta > 0.0 and not failed[i]:
            dvx = velocities[n1, 0] - velocities[n0, 0]
            dvy = velocities[n1, 1] - velocities[n0, 1]
            dvz = velocities[n1, 2] - velocities[n0, 2]
            v_proj = (dvx * dx + dvy * dy + dvz * dz) / length_safe
            damp_mag = rayleigh_beta * k_soft * v_proj
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
    spring_failed_step: np.ndarray,
    current_step: int,
    fracture_energy_multiplier: float,
    erosion_softening_steps: int = 10,
) -> tuple[np.ndarray, float]:
    n_springs = len(springs)
    effective_k = np.zeros(n_springs, dtype=positions.dtype)
    step_fracture_energy = 0.0
    for i in numba.prange(n_springs):
        if failed[i]:
            age = current_step - spring_failed_step[i]
            denom = erosion_softening_steps
            denom_safe = denom if denom != 0 else 1
            ramp = 1.0 - float(age) / float(denom_safe)
            if ramp <= 0.0:
                effective_k[i] = 0.0
            else:
                effective_k[i] = stiffnesses[i] * (1.0 - grid_damage[i]) * ramp
            continue
        n0 = springs[i, 0]
        n1 = springs[i, 1]
        dx = positions[n1, 0] - positions[n0, 0]
        dy = positions[n1, 1] - positions[n0, 1]
        dz = positions[n1, 2] - positions[n0, 2]
        length = np.sqrt(dx * dx + dy * dy + dz * dz)
        if rest_lengths[i] > 0.0:
            strain = (length - rest_lengths[i]) / rest_lengths[i]
        else:
            strain = length

        denom_strain = failure_strain - damage_onset_strain
        denom_strain_safe = denom_strain if denom_strain != 0.0 else 1.0
        val = (strain - damage_onset_strain) / denom_strain_safe
        damage = 0.0
        if val > 0.0:
            damage = val if val < 1.0 else 1.0
        new_damage = damage if damage > grid_damage[i] else grid_damage[i]
        grid_damage[i] = new_damage
        if failure_strain > 0.0 and (new_damage >= 1.0 or strain > failure_strain):
            failed[i] = True
            spring_failed_step[i] = current_step
            denom = erosion_softening_steps
            denom_safe = denom if denom != 0 else 1
            ramp = 1.0
            effective_k[i] = stiffnesses[i] * (1.0 - new_damage) * ramp
            w_fail = 0.5 * stiffnesses[i] * (failure_strain * rest_lengths[i]) ** 2
            step_fracture_energy += fracture_energy_multiplier * w_fail
        else:
            effective_k[i] = stiffnesses[i] * (1.0 - new_damage)
    return effective_k, step_fracture_energy


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
    spring_failed_step: np.ndarray,
    current_step: int,
    erosion_softening_steps: int,
) -> np.ndarray:
    n_nodes = len(node_spring_offsets) - 1
    active_counts = np.zeros(n_nodes, dtype=np.float64)
    for i in numba.prange(n_nodes):
        start = node_spring_offsets[i]
        end = node_spring_offsets[i + 1]
        cnt = 0.0
        for idx in range(start, end):
            sp_id = node_spring_ids[idx]
            if not grid_failed[sp_id]:
                cnt += 1.0
            else:
                age = current_step - spring_failed_step[sp_id]
                denom = erosion_softening_steps
                denom_safe = denom if denom != 0 else 1
                ramp = 1.0 - float(age) / float(denom_safe)
                if ramp > 0.0:
                    cnt += ramp
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


@backend.jit(fastmath=True, parallel=False)
def numba_compute_cfl_contact_distances(
    positions,
    proj_position,
    proj_quat,
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
    proximity_threshold,
):
    n_nodes = len(positions)
    dists = np.zeros(n_nodes, dtype=positions.dtype)
    dists[:] = 999.0
    q_conj = np.array([proj_quat[0], -proj_quat[1], -proj_quat[2], -proj_quat[3]], dtype=np.float64)
    max_R = max(proj_radius, max(proj_length, proj_span))
    cutoff = max_R + proximity_threshold
    cutoff_sq = cutoff**2
    P_loc = np.zeros(3, dtype=positions.dtype)

    for i in range(n_nodes):
        dx_p = positions[i, 0] - proj_position[0]
        dy_p = positions[i, 1] - proj_position[1]
        dz_p = positions[i, 2] - proj_position[2]
        if dx_p**2 + dy_p**2 + dz_p**2 > cutoff_sq:
            continue
        P_rel = positions[i] - proj_position
        numba_q_rotate_inplace(q_conj, P_rel, P_loc)
        dists[i] = numba_eval_sdf(
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
    return dists


@backend.jit(fastmath=True, parallel=False)
def numba_compute_cfl_contact_stiffness(
    positions,
    active_counts,
    node_initial_springs,
    proj_position,
    proj_quat,
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
    proximity_threshold,
    k_penalty,
):
    n_nodes = len(positions)
    nodal_k_contact = np.zeros(n_nodes, dtype=positions.dtype)
    q_conj = np.array([proj_quat[0], -proj_quat[1], -proj_quat[2], -proj_quat[3]], dtype=np.float64)
    max_R = max(proj_radius, max(proj_length, proj_span))
    cutoff = max_R + proximity_threshold
    cutoff_sq = cutoff**2
    P_loc_cfl = np.zeros(3, dtype=positions.dtype)

    for i in range(n_nodes):
        dx_p = positions[i, 0] - proj_position[0]
        dy_p = positions[i, 1] - proj_position[1]
        dz_p = positions[i, 2] - proj_position[2]
        if dx_p**2 + dy_p**2 + dz_p**2 > cutoff_sq:
            continue
        P_rel = positions[i] - proj_position
        numba_q_rotate_inplace(q_conj, P_rel, P_loc_cfl)
        dist = numba_eval_sdf(
            P_loc_cfl,
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

    return nodal_k_contact


@backend.jit(fastmath=True, parallel=False)
def numba_compute_projectile_contact_forces(
    positions,
    v_half,
    active_counts,
    node_initial_springs,
    proj_position,
    proj_quat,
    proj_v_half,
    proj_omega_half,
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
    proximity_threshold,
    k_penalty,
    proj_c_damping,
    dx,
    cap_stiffness,
    yield_strength_gpa,
    thickness,
    mu_s,
    dt,
    allow_sph_debris,
):
    n_nodes = len(positions)
    proj_forces = np.zeros((n_nodes, 3), dtype=positions.dtype)
    proj_reaction_force = np.zeros(3, dtype=np.float64)
    proj_torque = np.zeros(3, dtype=np.float64)
    friction_dissipated = 0.0
    proj_contact_e_step = 0.0

    q_conj = np.array([proj_quat[0], -proj_quat[1], -proj_quat[2], -proj_quat[3]], dtype=np.float64)
    if shape_code == 0:
        max_R = max(w_h, max(t_h, proj_length / 2.0))
    else:
        max_R = max(proj_radius, max(proj_length, proj_span))
    cutoff = max_R + proximity_threshold
    cutoff_sq = cutoff**2
    P_loc_force = np.zeros(3, dtype=positions.dtype)

    for i in range(n_nodes):
        dx_p = positions[i, 0] - proj_position[0]
        dy_p = positions[i, 1] - proj_position[1]
        dz_p = positions[i, 2] - proj_position[2]
        if dx_p**2 + dy_p**2 + dz_p**2 > cutoff_sq:
            continue
        P_rel = positions[i] - proj_position
        numba_q_rotate_inplace(q_conj, P_rel, P_loc_force)
        dist = numba_eval_sdf(
            P_loc_force,
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
        delta = proximity_threshold - dist
        if delta > 0.0:
            if not allow_sph_debris and active_counts[i] == 0.0:
                continue

            # Cap the penalty contact force to structural yield capacity if in metal shell mode
            if yield_strength_gpa > 0.0:
                f_cap = 1.5 * yield_strength_gpa * 1e9 * thickness * dx
            else:
                f_cap = cap_stiffness * dx if cap_stiffness > 0.0 else 1.0e6

            n_loc = numba_eval_sdf_normal(
                P_loc_force,
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
            delta_dot = -(v_rel[0] * n_world[0] + v_rel[1] * n_world[1] + v_rel[2] * n_world[2])

            f_mag = k_penalty * delta + proj_c_damping * delta_dot
            if f_mag < 0.0:
                f_mag = 0.0
            if f_mag > f_cap:
                f_mag = f_cap

            node_scale_factor = 1.0
            if node_initial_springs[i] > 0:
                if active_counts[i] > 0.0:
                    node_scale_factor = float(active_counts[i]) / float(node_initial_springs[i])
                else:
                    node_scale_factor = 0.05

            proj_forces[i, 0] += f_mag * n_world[0] * node_scale_factor
            proj_forces[i, 1] += f_mag * n_world[1] * node_scale_factor
            proj_forces[i, 2] += f_mag * n_world[2] * node_scale_factor

            proj_reaction_force[0] -= f_mag * n_world[0] * node_scale_factor
            proj_reaction_force[1] -= f_mag * n_world[1] * node_scale_factor
            proj_reaction_force[2] -= f_mag * n_world[2] * node_scale_factor

            # Surface projection for torque moment arm
            P_contact = P_rel - delta * n_world
            proj_torque[0] += P_contact[1] * (-f_mag * n_world[2] * node_scale_factor) - P_contact[
                2
            ] * (-f_mag * n_world[1] * node_scale_factor)
            proj_torque[1] += P_contact[2] * (-f_mag * n_world[0] * node_scale_factor) - P_contact[
                0
            ] * (-f_mag * n_world[2] * node_scale_factor)
            proj_torque[2] += P_contact[0] * (-f_mag * n_world[1] * node_scale_factor) - P_contact[
                1
            ] * (-f_mag * n_world[0] * node_scale_factor)

            # Projectile 6-DOF contact friction
            if mu_s > 0.0:
                v_rel_dot_n = v_rel[0] * n_world[0] + v_rel[1] * n_world[1] + v_rel[2] * n_world[2]
                v_tang = np.array(
                    [
                        v_rel[0] - v_rel_dot_n * n_world[0],
                        v_rel[1] - v_rel_dot_n * n_world[1],
                        v_rel[2] - v_rel_dot_n * n_world[2],
                    ],
                    dtype=np.float64,
                )
                v_rel_sq = v_tang[0] ** 2 + v_tang[1] ** 2 + v_tang[2] ** 2

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

            f_elastic = k_penalty * delta
            if f_elastic > f_cap:
                f_elastic = f_cap
            proj_contact_e_step += 0.5 * (f_elastic**2) / k_penalty * node_scale_factor

    return proj_forces, proj_reaction_force, proj_torque, proj_contact_e_step, friction_dissipated


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
    proximity_threshold: float,
    spring_failed_step: np.ndarray,
    erosion_softening_steps: int,
    velocity_clamping_multiplier: float,
    youngs_modulus_gpa: float,
    density_kgm3: float,
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

    c_p = np.sqrt(youngs_modulus_gpa * 1e9 / density_kgm3)
    v_max_phys = velocity_clamping_multiplier * c_p

    # Pre-compute effective_k at step 0 so we can compute initial forces
    if backend.BACKEND == "numba" and backend.HAS_NUMBA:
        effective_k, step_fe = numba_compute_effective_k(
            positions,
            grid_springs,
            grid_stiffnesses,
            grid_rest_lengths,
            grid_failed,
            damage_onset_strain,
            failure_strain,
            grid_damage,
            spring_failed_step,
            0,
            fracture_energy_multiplier,
            erosion_softening_steps,
        )
        failure_dissipated += step_fe
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
            grid_stiffnesses,
            grid_rest_lengths,
            grid_tension_only,
            rayleigh_beta,
            grid_failed,
            spring_failed_step,
            0,
            erosion_softening_steps,
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

        if shape_code >= 0:
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
            effective_k, step_fe = numba_compute_effective_k(
                positions,
                grid_springs,
                grid_stiffnesses,
                grid_rest_lengths,
                grid_failed,
                damage_onset_strain,
                failure_strain,
                grid_damage,
                spring_failed_step,
                step,
                fracture_energy_multiplier,
                erosion_softening_steps,
            )
            failure_dissipated += step_fe
            active_counts = numba_gather_active_counts(
                grid_failed,
                node_spring_offsets,
                node_spring_ids,
                spring_failed_step,
                step,
                erosion_softening_steps,
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
        # Compute contact mask and weights (always needed for force calculation, and optionally CFL)
        dists = numba_compute_cfl_contact_distances(
            positions,
            proj_position,
            proj_quat,
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
            proximity_threshold,
        )
        contact_mask = dists <= proximity_threshold
        w_i = where(contact_mask, 1.0 / maximum(np.abs(dists), 1e-4), 0.0)
        n_contacts = sum(contact_mask)
        w_sum = sum(w_i)
        w_mean = w_sum / where(n_contacts > 0, n_contacts, 1)
        w_mean_safe = where(w_mean > 0.0, w_mean, 1.0)
        w_normalized = where(contact_mask, w_i / w_mean_safe, 0.0)
        scale_factor = where(node_initial_springs > 0, active_counts / node_initial_springs, 0.0)

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
                nodal_k_contact = numba_compute_cfl_contact_stiffness(
                    positions,
                    active_counts,
                    node_initial_springs,
                    proj_position,
                    proj_quat,
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
                    proximity_threshold,
                    k_penalty,
                )

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
            if dx / dt < v_max_phys:
                v_max = dx / dt
            else:
                v_max = v_max_phys
        else:
            if dx / dt < v_max_phys:
                v_max = dx / dt
            else:
                v_max = v_max_phys

        # 3. Calculate internal node forces (stiffness + damping)
        if backend.BACKEND == "numba" and backend.HAS_NUMBA:
            spring_stiff_damp_forces, stiff_damp_e = numba_parallel_compute_spring_forces(
                positions,
                v_half,
                grid_springs,
                effective_k,
                grid_stiffnesses,
                grid_rest_lengths,
                grid_tension_only,
                node_spring_offsets,
                node_spring_ids,
                node_spring_signs,
                rayleigh_beta,
                grid_failed,
                spring_failed_step,
                step,
                erosion_softening_steps,
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

        cap_stiffness = grid_stiffnesses[0] if len(grid_stiffnesses) > 0 else 0.0
        proj_forces, proj_reaction_force, proj_torque, proj_contact_e_step, friction_diss_step = (
            numba_compute_projectile_contact_forces(
                positions,
                v_half,
                active_counts,
                node_initial_springs,
                proj_position,
                proj_quat,
                proj_v_half,
                proj_omega_half,
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
                proximity_threshold,
                k_penalty,
                proj_c_damping,
                dx,
                cap_stiffness,
                0.0,  # yield_strength_gpa (not used in fabric mode)
                0.0,  # thickness (not used in fabric mode)
                mu_s,
                dt,
                False,  # allow_sph_debris
            )
        )
        friction_dissipated += friction_diss_step

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

        if shape_code >= 0:
            q_conj = np.array(
                [proj_quat[0], -proj_quat[1], -proj_quat[2], -proj_quat[3]], dtype=np.float64
            )
            torque_body = numba_q_rotate(q_conj, proj_torque)
            omega_body = numba_q_rotate(q_conj, proj_omega)

            I_xx = 1.0 / proj_inertia_inv_diag[0] if proj_inertia_inv_diag[0] > 0.0 else 0.0
            I_yy = 1.0 / proj_inertia_inv_diag[1] if proj_inertia_inv_diag[1] > 0.0 else 0.0
            I_zz = 1.0 / proj_inertia_inv_diag[2] if proj_inertia_inv_diag[2] > 0.0 else 0.0

            h_x = I_xx * omega_body[0]
            h_y = I_yy * omega_body[1]
            h_z = I_zz * omega_body[2]

            coriolis_x = omega_body[1] * h_z - omega_body[2] * h_y
            coriolis_y = omega_body[2] * h_x - omega_body[0] * h_z
            coriolis_z = omega_body[0] * h_y - omega_body[1] * h_x

            omega_dot_body = np.array(
                [
                    proj_inertia_inv_diag[0] * (torque_body[0] - coriolis_x),
                    proj_inertia_inv_diag[1] * (torque_body[1] - coriolis_y),
                    proj_inertia_inv_diag[2] * (torque_body[2] - coriolis_z),
                ],
                dtype=np.float64,
            )

            rot_val = numba_q_rotate(proj_quat, omega_dot_body)
            omega_dot[0] = rot_val[0]
            omega_dot[1] = rot_val[1]
            omega_dot[2] = rot_val[2]

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
        if shape_code >= 0:
            proj_omega[:] = proj_omega_half + 0.5 * omega_dot * dt

        # 5. Irreversible Continuum Damage Mechanics (CDM) - already updated at step start
        if backend.BACKEND == "numba" and backend.HAS_NUMBA:
            pass
        else:
            # Fallback numpy CDM already updated at step start
            pass

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
            # failure_dissipated already incrementally accumulated
            pass
        else:
            # Python/NumPy fallback incremental energy tracking
            p1_telem = positions[grid_springs[:, 0]]
            p2_telem = positions[grid_springs[:, 1]]
            lengths_telem = sqrt(sum((p2_telem - p1_telem) ** 2, axis=1))
            strains_telem = (lengths_telem - grid_rest_lengths) / grid_rest_lengths
            new_failed_mask = (strains_telem > failure_strain) & ~grid_failed
            if np.any(new_failed_mask):
                w_fail = 0.5 * grid_stiffnesses * (failure_strain * grid_rest_lengths) ** 2
                failure_dissipated += float(
                    np.sum(fracture_energy_multiplier * w_fail[new_failed_mask])
                )
                grid_failed[new_failed_mask] = True

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
            if shape_code >= 0:
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


@backend.jit(parallel=False, fastmath=True)
def numba_step_shell_forces_and_failures(
    positions,
    X_ref,
    velocities,
    ang_positions,
    ang_velocities,
    elements,
    thickness,
    E,
    nu,
    yield_strength,
    hardening_modulus,
    ultimate_strain,
    tensile_strength,
    element_strains,
    element_stress,
    element_peeq,
    element_damage,
    element_failed,
    dx,
    rayleigh_beta,
    dt,
    density_kgm3,
    element_failed_step,
    current_step,
    erosion_softening_steps,
    element_peeq_rate,
    rate_parameter_c,
    rate_parameter_p,
    fracture_energy_jm2,
):
    n_nodes = len(positions)
    n_elements = len(elements)
    forces = np.zeros((n_nodes, 3), dtype=positions.dtype)
    torques = np.zeros((n_nodes, 3), dtype=positions.dtype)

    G = E / (2.0 * (1.0 + nu))
    kappa_s = 5.0 / 6.0  # shear correction factor
    G_s = G * kappa_s

    # Ramberg-Osgood precomputations
    eps_reg = 1e-5
    if ultimate_strain > 0.0 and tensile_strength > yield_strength:
        denom_ro = (ultimate_strain + eps_reg) ** 0.2 - eps_reg**0.2
        K_ro = (tensile_strength - yield_strength) / denom_ro
        sig_y_u = tensile_strength
        H_soft = 1e7  # 10 MPa (extremely soft)
    else:
        K_ro = 0.0
        sig_y_u = yield_strength
        H_soft = hardening_modulus

    step_fracture_energy = 0.0
    step_stiff_damp_power = 0.0

    for e in range(n_elements):
        n0 = elements[e, 0]
        n1 = elements[e, 1]
        n2 = elements[e, 2]
        n3 = elements[e, 3]

        is_failed = element_failed[e] == 1
        is_softening = element_failed[e] == 2
        ramp = 1.0

        if is_failed:
            continue

        if is_softening:
            element_failed_step[e] -= 1
            ramp = float(element_failed_step[e]) / float(erosion_softening_steps)
            if element_failed_step[e] <= 0:
                element_failed[e] = 1
                continue

        # Current nodal displacements
        u0 = positions[n0, 0] - X_ref[n0, 0]
        v0 = positions[n0, 1] - X_ref[n0, 1]
        w0 = positions[n0, 2] - X_ref[n0, 2]

        u1 = positions[n1, 0] - X_ref[n1, 0]
        v1 = positions[n1, 1] - X_ref[n1, 1]
        w1 = positions[n1, 2] - X_ref[n1, 2]

        u2 = positions[n2, 0] - X_ref[n2, 0]
        v2 = positions[n2, 1] - X_ref[n2, 1]
        w2 = positions[n2, 2] - X_ref[n2, 2]

        u3 = positions[n3, 0] - X_ref[n3, 0]
        v3 = positions[n3, 1] - X_ref[n3, 1]
        w3 = positions[n3, 2] - X_ref[n3, 2]

        # Nodal rotations (theta_x, theta_y)
        tx0, ty0 = ang_positions[n0, 0], ang_positions[n0, 1]
        tx1, ty1 = ang_positions[n1, 0], ang_positions[n1, 1]
        tx2, ty2 = ang_positions[n2, 0], ang_positions[n2, 1]
        tx3, ty3 = ang_positions[n3, 0], ang_positions[n3, 1]

        # Compute out-of-plane deflection gradients
        dw_dx = ((w1 - w0) + (w2 - w3)) / (2.0 * dx)
        dw_dy = ((w3 - w0) + (w2 - w1)) / (2.0 * dx)

        # Compute strains (with Von Karman non-linear membrane strain terms)
        eps_xx = ((u1 - u0) + (u2 - u3)) / (2.0 * dx) + 0.5 * dw_dx**2
        eps_yy = ((v3 - v0) + (v2 - v1)) / (2.0 * dx) + 0.5 * dw_dy**2
        gam_xy = (
            ((u3 - u0) + (u2 - u1)) / (2.0 * dx)
            + ((v1 - v0) + (v2 - v3)) / (2.0 * dx)
            + dw_dx * dw_dy
        )

        kappa_xx = ((ty1 - ty0) + (ty2 - ty3)) / (2.0 * dx)
        kappa_yy = -((tx3 - tx0) + (tx2 - tx1)) / (2.0 * dx)
        kappa_xy = ((ty3 - ty0) + (ty2 - ty1)) / (2.0 * dx) - ((tx1 - tx0) + (tx2 - tx3)) / (
            2.0 * dx
        )

        gam_xz = dw_dx + (ty0 + ty1 + ty2 + ty3) / 4.0
        gam_yz = dw_dy - (tx0 + tx1 + tx2 + tx3) / 4.0

        # Calculate strain increments
        d_eps_xx = eps_xx - element_strains[e, 0]
        d_eps_yy = eps_yy - element_strains[e, 1]
        d_gam_xy = gam_xy - element_strains[e, 2]

        d_kappa_xx = kappa_xx - element_strains[e, 3]
        d_kappa_yy = kappa_yy - element_strains[e, 4]
        d_kappa_xy = kappa_xy - element_strains[e, 5]
        d_gam_xz = gam_xz - element_strains[e, 6]
        d_gam_yz = gam_yz - element_strains[e, 7]

        # (strain store moved to after stress integration)

        # Incompressibility-based dynamic thickness update
        lambda_z = 1.0 - (eps_xx + eps_yy)
        if lambda_z < 0.1:
            lambda_z = 0.1
        t_curr = thickness * lambda_z

        zk0 = -0.5 * t_curr
        zk1 = -0.25 * t_curr
        zk2 = 0.0
        zk3 = 0.25 * t_curr
        zk4 = 0.5 * t_curr

        wk0 = t_curr / 12.0
        wk1 = 4.0 * t_curr / 12.0
        wk2 = 2.0 * t_curr / 12.0
        wk3 = 4.0 * t_curr / 12.0
        wk4 = t_curr / 12.0

        N_xx, N_yy, N_xy = 0.0, 0.0, 0.0
        M_xx, M_yy, M_xy = 0.0, 0.0, 0.0

        # Thickness integration
        for k in range(5):
            if k == 0:
                zk = zk0
                wk = wk0
            elif k == 1:
                zk = zk1
                wk = wk1
            elif k == 2:
                zk = zk2
                wk = wk2
            elif k == 3:
                zk = zk3
                wk = wk3
            else:
                zk = zk4
                wk = wk4

            d_exx_k = d_eps_xx + zk * d_kappa_xx
            d_eyy_k = d_eps_yy + zk * d_kappa_yy
            d_gxy_k = d_gam_xy + zk * d_kappa_xy

            # Elastic trial stress
            sig_xx_old = element_stress[e, k, 0]
            sig_yy_old = element_stress[e, k, 1]
            tau_xy_old = element_stress[e, k, 2]

            C_mat = E / (1.0 - nu * nu)
            sig_xx_trial = sig_xx_old + C_mat * (d_exx_k + nu * d_eyy_k)
            sig_yy_trial = sig_yy_old + C_mat * (d_eyy_k + nu * d_exx_k)
            tau_xy_trial = tau_xy_old + G * d_gxy_k

            # Yield check (von Mises yield criterion)
            sig_vm_trial = np.sqrt(
                sig_xx_trial**2
                + sig_yy_trial**2
                - sig_xx_trial * sig_yy_trial
                + 3.0 * tau_xy_trial**2
            )

            peeq_old = element_peeq[e, k]
            peeq_rate_old = element_peeq_rate[e, k]

            # Cowper-Symonds rate scaling factor (constant during iteration)
            if rate_parameter_c > 0.0 and rate_parameter_p > 0.0:
                beta = 1.0 + (max(0.0, peeq_rate_old) / rate_parameter_c) ** (
                    1.0 / rate_parameter_p
                )
            else:
                beta = 1.0

            if ultimate_strain > 0.0 and tensile_strength > yield_strength:
                if peeq_old <= ultimate_strain:
                    yield_val = yield_strength + K_ro * ((peeq_old + eps_reg) ** 0.2 - eps_reg**0.2)
                else:
                    yield_val = sig_y_u + H_soft * (peeq_old - ultimate_strain)
            else:
                yield_val = yield_strength + hardening_modulus * peeq_old

            yield_val_dynamic = yield_val * beta
            f_yield = sig_vm_trial - yield_val_dynamic

            sig_xx_new = sig_xx_trial
            sig_yy_new = sig_yy_trial
            tau_xy_new = tau_xy_trial
            peeq_new = peeq_old
            d_peeq = 0.0

            if f_yield > 0.0 and yield_strength > 0.0:
                if ultimate_strain > 0.0 and tensile_strength > yield_strength:
                    # Nonlinear Newton-Raphson radial return
                    d_peeq = f_yield / (3.0 * G + 100.0)
                    for _ in range(8):
                        peeq_temp = peeq_old + d_peeq
                        if peeq_temp <= ultimate_strain:
                            sig_y_val = yield_strength + K_ro * (
                                (peeq_temp + eps_reg) ** 0.2 - eps_reg**0.2
                            )
                            H_tang = 0.2 * K_ro * (peeq_temp + eps_reg) ** (-0.8)
                        else:
                            sig_y_val = sig_y_u + H_soft * (peeq_temp - ultimate_strain)
                            H_tang = H_soft

                        f_val = sig_vm_trial - 3.0 * G * d_peeq - sig_y_val * beta
                        df_val = -3.0 * G - H_tang * beta
                        diff = f_val / df_val
                        d_peeq -= diff
                        if abs(diff) < 1e-6 * yield_strength:
                            break
                    if d_peeq < 0.0:
                        d_peeq = 0.0
                    peeq_new = peeq_old + d_peeq

                    # Re-evaluate final yield value for plastic energy tracking
                    if peeq_new <= ultimate_strain:
                        yield_val = yield_strength + K_ro * (
                            (peeq_new + eps_reg) ** 0.2 - eps_reg**0.2
                        )
                    else:
                        yield_val = sig_y_u + H_soft * (peeq_new - ultimate_strain)
                else:
                    # Linear radial return (analytical solution)
                    d_peeq = f_yield / (3.0 * G + hardening_modulus * beta)
                    if d_peeq < 0.0:
                        d_peeq = 0.0
                    peeq_new = peeq_old + d_peeq
                    yield_val = yield_strength + hardening_modulus * peeq_new

                # Scale stress components (strictly dissipative, scale <= 1.0)
                scale = 1.0 - (3.0 * G * d_peeq) / (sig_vm_trial if sig_vm_trial != 0.0 else 1.0)
                if scale > 1.0:
                    scale = 1.0
                elif scale < 0.0:
                    scale = 0.0

                sig_xx_new = sig_xx_trial * scale
                sig_yy_new = sig_yy_trial * scale
                tau_xy_new = tau_xy_trial * scale

            # Update filtered plastic strain rate
            peeq_rate_inst = d_peeq / dt if dt > 0.0 else 0.0
            peeq_rate_new = 0.9 * peeq_rate_old + 0.1 * peeq_rate_inst
            element_peeq_rate[e, k] = peeq_rate_new

            # Stress Triaxiality & Continuous Damage Mechanics
            sig_mean = (sig_xx_new + sig_yy_new) / 3.0
            sig_vm_new = np.sqrt(
                sig_xx_new**2 + sig_yy_new**2 - sig_xx_new * sig_yy_new + 3.0 * tau_xy_new**2
            )
            eta = sig_mean / sig_vm_new if sig_vm_new > 0.0 else 0.0

            # Triaxiality-dependent failure strain (continuous at eta = 1/3)
            if eta >= (1.0 / 3.0):
                eps_f = ultimate_strain * np.exp(-1.5 * (eta - 1.0 / 3.0))
            else:
                eps_f = ultimate_strain * np.exp(-0.5 * (eta - 1.0 / 3.0))

            if eps_f < 0.005:
                eps_f = 0.005

            # Damage evolution
            if ultimate_strain > 0.0 and peeq_new > ultimate_strain:
                if fracture_energy_jm2 > 0.0:
                    d_dmg = (yield_val * dx * d_peeq) / (2.0 * fracture_energy_jm2)
                else:
                    d_dmg = d_peeq / eps_f
                # Viscous regularization to prevent unphysical high-frequency shock waves
                mu_visc = 1.0e-6  # 1 microsecond relaxation time
                dmg_val = element_damage[e, k] + (dt / (dt + mu_visc)) * d_dmg
                if dmg_val > 1.0:
                    dmg_val = 1.0
                element_damage[e, k] = dmg_val
            else:
                element_damage[e, k] = 0.0

            d_factor = 1.0 - element_damage[e, k]

            # Plastic dissipation energy (using damaged stress)
            step_fracture_energy += d_factor * yield_val * beta * d_peeq * wk * (dx * dx)

            # Store updated plastic strain and stresses
            element_peeq[e, k] = peeq_new
            element_stress[e, k, 0] = sig_xx_new
            element_stress[e, k, 1] = sig_yy_new
            element_stress[e, k, 2] = tau_xy_new

        # Store strains after stress integration so stored state matches what
        # the radial-return used, not the raw kinematic increment.
        element_strains[e, 0] = eps_xx
        element_strains[e, 1] = eps_yy
        element_strains[e, 2] = gam_xy
        element_strains[e, 3] = kappa_xx
        element_strains[e, 4] = kappa_yy
        element_strains[e, 5] = kappa_xy
        element_strains[e, 6] = gam_xz
        element_strains[e, 7] = gam_yz

        # Check for failure initiation
        if element_failed[e] == 0 and ultimate_strain > 0.0:
            mean_dmg = (
                element_damage[e, 0]
                + element_damage[e, 1]
                + element_damage[e, 2]
                + element_damage[e, 3]
                + element_damage[e, 4]
            ) / 5.0
            if (
                mean_dmg >= 0.70
                or element_damage[e, 2] >= 0.90
                or (element_damage[e, 0] >= 0.95 and element_damage[e, 4] >= 0.95)
            ):
                element_failed[e] = 2  # softening
                element_failed_step[e] = erosion_softening_steps
                ramp = 1.0

        # Calculate average damage factor for transverse shear scaling
        mean_damage = (
            element_damage[e, 0]
            + element_damage[e, 1]
            + element_damage[e, 2]
            + element_damage[e, 3]
            + element_damage[e, 4]
        ) / 5.0
        d_factor_mean = 1.0 - mean_damage if ultimate_strain > 0.0 else 1.0

        # Recompute forces and moments integrals
        for k in range(5):
            if k == 0:
                zk = zk0
                wk = wk0
            elif k == 1:
                zk = zk1
                wk = wk1
            elif k == 2:
                zk = zk2
                wk = wk2
            elif k == 3:
                zk = zk3
                wk = wk3
            else:
                zk = zk4
                wk = wk4

            sig_xx = element_stress[e, k, 0]
            sig_yy = element_stress[e, k, 1]
            tau_xy = element_stress[e, k, 2]

            # Damped stresses
            e_dot_xx_k = (d_eps_xx + zk * d_kappa_xx) / dt if dt > 0.0 else 0.0
            e_dot_yy_k = (d_eps_yy + zk * d_kappa_yy) / dt if dt > 0.0 else 0.0
            g_dot_xy_k = (d_gam_xy + zk * d_kappa_xy) / dt if dt > 0.0 else 0.0

            sig_xx_damp = rayleigh_beta * C_mat * (e_dot_xx_k + nu * e_dot_yy_k)
            sig_yy_damp = rayleigh_beta * C_mat * (e_dot_yy_k + nu * e_dot_xx_k)
            tau_xy_damp = rayleigh_beta * G * g_dot_xy_k

            # Volumetric Bulk Viscosity
            eps_vol_dot = e_dot_xx_k + e_dot_yy_k
            q_bulk = 0.0
            if eps_vol_dot < 0.0:
                c_sound = sqrt(E / density_kgm3)
                q_bulk = (
                    density_kgm3
                    * dx
                    * (0.06 * c_sound * abs(eps_vol_dot) + 1.5 * dx * (eps_vol_dot**2))
                )

            d_factor = 1.0 - element_damage[e, k] if ultimate_strain > 0.0 else 1.0

            sig_xx_total = (sig_xx + sig_xx_damp - q_bulk) * d_factor
            sig_yy_total = (sig_yy + sig_yy_damp - q_bulk) * d_factor
            tau_xy_total = (tau_xy + tau_xy_damp) * d_factor

            N_xx += wk * sig_xx_total
            N_yy += wk * sig_yy_total
            N_xy += wk * tau_xy_total

            M_xx += wk * sig_xx_total * zk
            M_yy += wk * sig_yy_total * zk
            M_xy += wk * tau_xy_total * zk

            step_stiff_damp_power += (
                (
                    (sig_xx_damp - q_bulk) * e_dot_xx_k
                    + (sig_yy_damp - q_bulk) * e_dot_yy_k
                    + tau_xy_damp * g_dot_xy_k
                )
                * wk
                * (dx * dx)
            )

        # Transverse shear forces (scaling with dynamic thickness and damage mean)
        gam_dot_xz = d_gam_xz / dt if dt > 0.0 else 0.0
        gam_dot_yz = d_gam_yz / dt if dt > 0.0 else 0.0

        q_damp_x = rayleigh_beta * G_s * t_curr * gam_dot_xz
        q_damp_y = rayleigh_beta * G_s * t_curr * gam_dot_yz

        Q_x = (G_s * t_curr * gam_xz + q_damp_x) * d_factor_mean
        Q_y = (G_s * t_curr * gam_yz + q_damp_y) * d_factor_mean

        step_stiff_damp_power += (
            (q_damp_x * gam_dot_xz + q_damp_y * gam_dot_yz) * d_factor_mean * (dx * dx)
        )

        # Apply softened erosion scaling to forces/moments
        N_xx *= ramp
        N_yy *= ramp
        N_xy *= ramp
        M_xx *= ramp
        M_yy *= ramp
        M_xy *= ramp
        Q_x *= ramp
        Q_y *= ramp

        # Calculate nodal internal forces and moments
        half_dx = 0.5 * dx

        # Project membrane tensions onto out-of-plane gradients (Von Karman membrane stiffness)
        Q_x_eff = Q_x + N_xx * dw_dx + N_xy * dw_dy
        Q_y_eff = Q_y + N_yy * dw_dy + N_xy * dw_dx

        # Node 0
        forces[n0, 0] += N_xx * half_dx + N_xy * half_dx
        forces[n0, 1] += N_yy * half_dx + N_xy * half_dx
        forces[n0, 2] += Q_x_eff * half_dx + Q_y_eff * half_dx
        torques[n0, 0] += -M_yy * half_dx - M_xy * half_dx + 0.25 * dx * dx * Q_y
        torques[n0, 1] += M_xx * half_dx + M_xy * half_dx - 0.25 * dx * dx * Q_x

        # Node 1
        forces[n1, 0] += -N_xx * half_dx + N_xy * half_dx
        forces[n1, 1] += N_yy * half_dx - N_xy * half_dx
        forces[n1, 2] += -Q_x_eff * half_dx + Q_y_eff * half_dx
        torques[n1, 0] += -M_yy * half_dx + M_xy * half_dx + 0.25 * dx * dx * Q_y
        torques[n1, 1] += -M_xx * half_dx + M_xy * half_dx - 0.25 * dx * dx * Q_x

        # Node 2
        forces[n2, 0] += -N_xx * half_dx - N_xy * half_dx
        forces[n2, 1] += -N_yy * half_dx - N_xy * half_dx
        forces[n2, 2] += -Q_x_eff * half_dx - Q_y_eff * half_dx
        torques[n2, 0] += M_yy * half_dx + M_xy * half_dx + 0.25 * dx * dx * Q_y
        torques[n2, 1] += -M_xx * half_dx - M_xy * half_dx - 0.25 * dx * dx * Q_x

        # Node 3
        forces[n3, 0] += N_xx * half_dx - N_xy * half_dx
        forces[n3, 1] += -N_yy * half_dx + N_xy * half_dx
        forces[n3, 2] += Q_x_eff * half_dx - Q_y_eff * half_dx
        torques[n3, 0] += M_yy * half_dx - M_xy * half_dx + 0.25 * dx * dx * Q_y
        torques[n3, 1] += M_xx * half_dx - M_xy * half_dx - 0.25 * dx * dx * Q_x

        # Flanagan-Belytschko hourglass stabilization for 4-node quadrilateral shell element
        # 1. Project nodal velocities onto the zero-energy hourglass mode (gamma = [1, -1, 1, -1])
        q_vx = velocities[n0, 0] - velocities[n1, 0] + velocities[n2, 0] - velocities[n3, 0]
        q_vy = velocities[n0, 1] - velocities[n1, 1] + velocities[n2, 1] - velocities[n3, 1]
        q_vz = velocities[n0, 2] - velocities[n1, 2] + velocities[n2, 2] - velocities[n3, 2]

        q_wx = (
            ang_velocities[n0, 0]
            - ang_velocities[n1, 0]
            + ang_velocities[n2, 0]
            - ang_velocities[n3, 0]
        )
        q_wy = (
            ang_velocities[n0, 1]
            - ang_velocities[n1, 1]
            + ang_velocities[n2, 1]
            - ang_velocities[n3, 1]
        )
        q_wz = (
            ang_velocities[n0, 2]
            - ang_velocities[n1, 2]
            + ang_velocities[n2, 2]
            - ang_velocities[n3, 2]
        )

        # 2. Compute physical stabilization damping coefficients using wave-impedance
        C_damp = 0.015 * sqrt(E * density_kgm3) * t_curr * dx * ramp
        C_rot_damp = 0.015 * sqrt(E * density_kgm3) * (t_curr**3) * dx * ramp
        shear_damping = 0.015 * sqrt(G * density_kgm3) * t_curr * (dx * dx * dx) * ramp
        C_rot_total = C_rot_damp + shear_damping

        # 3. Distribute viscous resisting forces/torques using orthogonalized operator (gamma_I)
        # Scaling by 0.25 accounts for the inner product norm ||gamma||^2 = 4.
        # Node 0 (gamma_0 = 1.0)
        forces[n0, 0] -= 0.25 * C_damp * q_vx
        forces[n0, 1] -= 0.25 * C_damp * q_vy
        forces[n0, 2] -= 0.25 * C_damp * q_vz
        torques[n0, 0] -= 0.25 * C_rot_total * q_wx
        torques[n0, 1] -= 0.25 * C_rot_total * q_wy
        torques[n0, 2] -= 0.25 * C_rot_total * q_wz

        # Node 1 (gamma_1 = -1.0)
        forces[n1, 0] += 0.25 * C_damp * q_vx
        forces[n1, 1] += 0.25 * C_damp * q_vy
        forces[n1, 2] += 0.25 * C_damp * q_vz
        torques[n1, 0] += 0.25 * C_rot_total * q_wx
        torques[n1, 1] += 0.25 * C_rot_total * q_wy
        torques[n1, 2] += 0.25 * C_rot_total * q_wz

        # Node 2 (gamma_2 = 1.0)
        forces[n2, 0] -= 0.25 * C_damp * q_vx
        forces[n2, 1] -= 0.25 * C_damp * q_vy
        forces[n2, 2] -= 0.25 * C_damp * q_vz
        torques[n2, 0] -= 0.25 * C_rot_total * q_wx
        torques[n2, 1] -= 0.25 * C_rot_total * q_wy
        torques[n2, 2] -= 0.25 * C_rot_total * q_wz

        # Node 3 (gamma_3 = -1.0)
        forces[n3, 0] += 0.25 * C_damp * q_vx
        forces[n3, 1] += 0.25 * C_damp * q_vy
        forces[n3, 2] += 0.25 * C_damp * q_vz
        torques[n3, 0] += 0.25 * C_rot_total * q_wx
        torques[n3, 1] += 0.25 * C_rot_total * q_wy
        torques[n3, 2] += 0.25 * C_rot_total * q_wz

    return forces, torques, step_fracture_energy, step_stiff_damp_power


@backend.jit(
    parallel=False,
    static_argnames=(
        "n_plies",
        "use_viscous",
        "cfl_factor",
        "shape_code",
        "use_czm",
    ),
)
def _fused_shell_loop_jit(
    positions,
    X_ref,
    velocities,
    grid_masses,
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
    dt,
    n_steps,
    save_interval,
    damp_dissipated_init,
    failure_dissipated_init,
    clamp_dissipated_init,
    t_sim_init,
    strike_direction,
    use_viscous,
    cfl_factor,
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
    proj_half_width,
    proj_half_thickness,
    contact_energy_init,
    mu_s,
    friction_dissipated_init,
    elements,
    element_stress,
    element_peeq,
    element_damage,
    element_failed,
    yield_strength_gpa,
    hardening_modulus_gpa,
    ultimate_strain,
    poisson_ratio,
    tensile_strength_gpa,
    thickness,
    youngs_modulus_gpa,
    density_kgm3,
    proximity_threshold,
    springs,
    spring_failed,
    spring_damage,
    is_tiebreak,
    cohesive_strength_gpa,
    fracture_energy_jm2,
    use_czm,
    coincident_nodes,
    node_czm_spring_ids,
    element_strains,
    ang_positions,
    ang_velocities,
    ang_accel,
    element_failed_step,
    erosion_softening_steps,
    velocity_clamping_multiplier,
    element_peeq_rate,
    rate_parameter_c,
    rate_parameter_p,
):
    n_nodes = len(positions)
    n_elements = len(elements)
    m_frames = max(1, n_steps // save_interval)
    mass_min = np.min(grid_masses)

    # Pre-allocate history structures (compatible with JIT vector allocations)
    hist_positions = zeros((m_frames, n_nodes, 3), dtype=positions.dtype)
    hist_failed = zeros((m_frames, n_elements), dtype=np.bool_)
    hist_proj_pos = zeros((m_frames, 3), dtype=positions.dtype)
    hist_time = zeros(m_frames, dtype=positions.dtype)
    hist_ke = zeros(m_frames, dtype=positions.dtype)
    hist_se = zeros(m_frames, dtype=positions.dtype)
    hist_proj_ke = zeros(m_frames, dtype=positions.dtype)

    damp_dissipated = damp_dissipated_init
    failure_dissipated = failure_dissipated_init
    clamp_dissipated = clamp_dissipated_init
    t_sim = t_sim_init
    contact_energy = contact_energy_init
    friction_dissipated = friction_dissipated_init

    # Projectile dynamic parameters
    proj_reaction_force = zeros(3, dtype=np.float64)
    proj_torque = zeros(3, dtype=np.float64)
    omega_dot = zeros(3, dtype=np.float64)

    rot_inertia = (1.0 / 12.0) * grid_masses * (dx * dx)
    rot_inertia = maximum(rot_inertia, 1e-12)  # safeguard against division by zero
    rot_inertia_col = rot_inertia[:, np.newaxis]
    masses_col = grid_masses[:, np.newaxis]

    E = youngs_modulus_gpa * 1e9
    yield_strength = yield_strength_gpa * 1e9
    hardening_modulus = hardening_modulus_gpa * 1e9
    tensile_strength = (
        tensile_strength_gpa * 1e9
        if tensile_strength_gpa > 0.0
        else (yield_strength + hardening_modulus * ultimate_strain)
    )

    if use_czm:
        cohesive_strength = cohesive_strength_gpa * 1e9
        A_trib = 0.5 * dx * thickness
        F_max = cohesive_strength * A_trib
        delta_c = 2.0 * fracture_energy_jm2 / cohesive_strength
        delta_0 = 0.01 * delta_c
        k_0 = F_max / delta_0
    else:
        cohesive_strength = 0.0
        A_trib = 0.0
        F_max = 0.0
        delta_c = 0.0
        delta_0 = 0.0
        k_0 = 0.0

    c_p = sqrt(E / (density_kgm3 * (1.0 - poisson_ratio * poisson_ratio)))
    if cfl_factor > 0.0:
        omega_shell = 2.0 * c_p / dx
        k_total = mass_min * (omega_shell**2)
        if use_czm:
            k_total += 4.0 * k_0
        if k_penalty > 0.0:
            k_total += 4.0 * k_penalty
        omega_max = sqrt(k_total / mass_min)
        dt_crit = sqrt(rayleigh_beta**2 + 4.0 / (omega_max**2)) - rayleigh_beta
        dt = cfl_factor * dt_crit

    # Dynamically scale softening steps to match the physical wave crossing time of the element.
    # This prevents instantaneous stress release (shock fronts) that drive unzipping cascades.
    t_cross = dx / c_p
    softening_steps_eff = int(t_cross / dt)
    if erosion_softening_steps > softening_steps_eff:
        softening_steps_eff = erosion_softening_steps
    if softening_steps_eff < 10:
        softening_steps_eff = 10
    erosion_softening_steps = softening_steps_eff

    accel = zeros((n_nodes, 3), dtype=positions.dtype)

    # Re-evaluate dimensions for bullet/cylinder shape logic
    R_og_val = 0.0
    L_body_val = 0.0
    L_nose_val = 0.0
    if shape_code == 3:  # Bullet
        R_og_val = proj_radius * proj_ogive_multiplier
        if R_og_val > proj_radius:
            L_nose_val = sqrt(2.0 * R_og_val * proj_radius - proj_radius**2)
        else:
            L_nose_val = 0.0
        L_body_val = max(0.0, proj_length - L_nose_val)

    for step in range(n_steps):
        # 1. Half-step Velocity Verlet updates (Part B.1)
        v_half = velocities + 0.5 * accel * dt
        omega_half = ang_velocities + 0.5 * ang_accel * dt
        positions = positions + v_half * dt
        ang_positions = ang_positions + omega_half * dt

        proj_v_half = proj_velocity + 0.5 * (proj_reaction_force / proj_mass) * dt
        proj_position = proj_position + proj_v_half * dt

        proj_omega_half = proj_omega + 0.5 * omega_dot * dt

        # Projectile quat rotation integration
        if shape_code >= 0:
            omega_q = np.array(
                [0.0, proj_omega_half[0], proj_omega_half[1], proj_omega_half[2]], dtype=np.float64
            )
            q_dot = numba_q_mul(omega_q, proj_quat)
            q_new = proj_quat + 0.5 * dt * q_dot
            q_new_norm = sqrt(q_new[0] ** 2 + q_new[1] ** 2 + q_new[2] ** 2 + q_new[3] ** 2)
            if q_new_norm > 1e-8:
                proj_quat[0] = q_new[0] / q_new_norm
                proj_quat[1] = q_new[1] / q_new_norm
                proj_quat[2] = q_new[2] / q_new_norm
                proj_quat[3] = q_new[3] / q_new_norm

        t_sim += dt

        # 2. CFL step scaling/safety check
        # We can dynamically compute node scale factor based on active elements
        active_counts = zeros(n_nodes, dtype=positions.dtype)
        node_initial_elements = zeros(n_nodes, dtype=np.int32)
        for e in range(n_elements):
            n0, n1, n2, n3 = (
                elements[e, 0],
                elements[e, 1],
                elements[e, 2],
                elements[e, 3],
            )
            node_initial_elements[n0] += 1
            node_initial_elements[n1] += 1
            node_initial_elements[n2] += 1
            node_initial_elements[n3] += 1
            if element_failed[e] == 0:
                active_counts[n0] += 1.0
                active_counts[n1] += 1.0
                active_counts[n2] += 1.0
                active_counts[n3] += 1.0
            elif element_failed[e] == 2:
                rem_steps = element_failed_step[e]
                ramp = (
                    float(rem_steps) / float(erosion_softening_steps)
                    if erosion_softening_steps > 0
                    else 0.0
                )
                if ramp < 0.0:
                    ramp = 0.0
                elif ramp > 1.0:
                    ramp = 1.0
                active_counts[n0] += ramp
                active_counts[n1] += ramp
                active_counts[n2] += ramp
                active_counts[n3] += ramp

        # Compute physical velocity cap based on longitudinal wave speed.
        # Removing the non-physical v_max_limit cap prevents artificial momentum destruction
        # and unzipping cascades caused by clipping physical snap-back and Poisson reflection waves.
        c_p = sqrt(E / (density_kgm3 * (1.0 - poisson_ratio * poisson_ratio)))
        v_max_phys = velocity_clamping_multiplier * c_p
        if dx / dt < v_max_phys:
            v_max = dx / dt
        else:
            v_max = v_max_phys

        # 3. Calculate internal forces and moments
        shell_forces, shell_torques, step_fracture_energy, step_stiff_damp_power = (
            numba_step_shell_forces_and_failures(
                positions,
                X_ref,
                velocities,
                ang_positions,
                ang_velocities,
                elements,
                thickness,
                E,
                poisson_ratio,
                yield_strength,
                hardening_modulus,
                ultimate_strain,
                tensile_strength,
                element_strains,
                element_stress,
                element_peeq,
                element_damage,
                element_failed,
                dx,
                rayleigh_beta,
                dt,
                density_kgm3,
                element_failed_step,
                step,
                erosion_softening_steps,
                element_peeq_rate,
                rate_parameter_c,
                rate_parameter_p,
                fracture_energy_jm2,
            )
        )
        failure_dissipated += step_fracture_energy
        damp_dissipated += step_stiff_damp_power * dt

        # 4. Contact forces
        proj_forces, proj_reaction_force, proj_torque, proj_contact_e_step, friction_diss_step = (
            numba_compute_projectile_contact_forces(
                positions,
                v_half,
                active_counts,
                node_initial_elements,
                proj_position,
                proj_quat,
                proj_v_half,
                proj_omega_half,
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
                proj_half_width,
                proj_half_thickness,
                proximity_threshold,
                k_penalty,
                proj_c_damping,
                dx,
                yield_strength * thickness,
                yield_strength_gpa,
                thickness,
                mu_s,
                dt,
                True,  # allow_sph_debris
            )
        )
        friction_dissipated += friction_diss_step

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
            shell_forces + proj_forces + interply_forces + f_mass_damp + nodal_external_forces
        )
        net_forces = clamp_boundary(net_forces, boundary_mask)

        # 4. Update Accelerations and Finalize velocities (Velocity Verlet Step 2)
        accel = net_forces / masses_col
        proj_accel = proj_reaction_force / proj_mass

        if shape_code >= 0:
            q_conj = np.array(
                [proj_quat[0], -proj_quat[1], -proj_quat[2], -proj_quat[3]], dtype=np.float64
            )
            torque_body = numba_q_rotate(q_conj, proj_torque)
            omega_body = numba_q_rotate(q_conj, proj_omega)

            I_xx = 1.0 / proj_inertia_inv_diag[0] if proj_inertia_inv_diag[0] > 0.0 else 0.0
            I_yy = 1.0 / proj_inertia_inv_diag[1] if proj_inertia_inv_diag[1] > 0.0 else 0.0
            I_zz = 1.0 / proj_inertia_inv_diag[2] if proj_inertia_inv_diag[2] > 0.0 else 0.0

            h_x = I_xx * omega_body[0]
            h_y = I_yy * omega_body[1]
            h_z = I_zz * omega_body[2]

            coriolis_x = omega_body[1] * h_z - omega_body[2] * h_y
            coriolis_y = omega_body[2] * h_x - omega_body[0] * h_z
            coriolis_z = omega_body[0] * h_y - omega_body[1] * h_x

            omega_dot_body = np.array(
                [
                    proj_inertia_inv_diag[0] * (torque_body[0] - coriolis_x),
                    proj_inertia_inv_diag[1] * (torque_body[1] - coriolis_y),
                    proj_inertia_inv_diag[2] * (torque_body[2] - coriolis_z),
                ],
                dtype=np.float64,
            )

            rot_val = numba_q_rotate(proj_quat, omega_dot_body)
            omega_dot[0] = rot_val[0]
            omega_dot[1] = rot_val[1]
            omega_dot[2] = rot_val[2]

        velocities = v_half + 0.5 * accel * dt

        # Rotational velocities updates
        net_torques = shell_torques
        net_torques = clamp_boundary(net_torques, boundary_mask)
        ang_accel = net_torques / rot_inertia_col
        ang_velocities = omega_half + 0.5 * ang_accel * dt
        for i in range(n_nodes):
            if active_counts[i] == 0:
                velocities[i, 0] = 0.0
                velocities[i, 1] = 0.0
                velocities[i, 2] = 0.0
                accel[i, 0] = 0.0
                accel[i, 1] = 0.0
                accel[i, 2] = 0.0
                ang_velocities[i, 0] = 0.0
                ang_velocities[i, 1] = 0.0
                ang_velocities[i, 2] = 0.0
                ang_accel[i, 0] = 0.0
                ang_accel[i, 1] = 0.0
                ang_accel[i, 2] = 0.0

        # CFL velocity clamping
        for i in range(n_nodes):
            vx = velocities[i, 0]
            vy = velocities[i, 1]
            vz = velocities[i, 2]
            v_mag = sqrt(vx * vx + vy * vy + vz * vz)
            if v_mag > v_max:
                scale = v_max / v_mag
                velocities[i, 0] = vx * scale
                velocities[i, 1] = vy * scale
                velocities[i, 2] = vz * scale
                clamp_dissipated += 0.5 * grid_masses[i] * (v_mag * v_mag - v_max * v_max)

        proj_velocity = proj_v_half + 0.5 * proj_accel * dt
        if shape_code >= 0:
            proj_omega[:] = proj_omega_half + 0.5 * omega_dot * dt

        # 5. Populate history arrays at specified intervals
        if save_interval > 0 and step % save_interval == 0:
            frame_idx = step // save_interval
            if frame_idx < m_frames:
                trans_ke = 0.5 * sum(grid_masses * sum(velocities**2, axis=1))
                rot_ke_sheet = 0.5 * sum(rot_inertia * sum(ang_velocities**2, axis=1))
                ke = trans_ke + rot_ke_sheet

                # Strain energy se: estimate from element elastic stresses
                se = 0.0
                w_pts_se = np.array(
                    [
                        thickness / 12.0,
                        4.0 * thickness / 12.0,
                        2.0 * thickness / 12.0,
                        4.0 * thickness / 12.0,
                        thickness / 12.0,
                    ]
                )
                for e in range(n_elements):
                    if element_failed[e] == 0:
                        el_se = 0.0
                        for k in range(5):
                            wk = w_pts_se[k]
                            s_xx = element_stress[e, k, 0]
                            s_yy = element_stress[e, k, 1]
                            t_xy = element_stress[e, k, 2]
                            d_factor = 1.0 - element_damage[e, k] if ultimate_strain > 0.0 else 1.0
                            u0 = (
                                (0.5 / E)
                                * (
                                    s_xx**2
                                    + s_yy**2
                                    - 2.0 * poisson_ratio * s_xx * s_yy
                                    + 2.0 * (1.0 + poisson_ratio) * t_xy**2
                                )
                                * d_factor
                            )
                            el_se += u0 * wk
                        se += el_se * (dx * dx)

                proj_rot_ke = 0.0
                if shape_code >= 0:
                    proj_rot_ke = 0.5 * (
                        (1.0 / proj_inertia_inv_diag[0]) * proj_omega[0] ** 2
                        + (1.0 / proj_inertia_inv_diag[1]) * proj_omega[1] ** 2
                        + (1.0 / proj_inertia_inv_diag[2]) * proj_omega[2] ** 2
                    )
                proj_ke = 0.5 * proj_mass * sum(proj_velocity**2) + proj_rot_ke

                hist_positions = set_index_3d(hist_positions, frame_idx, positions)
                hist_failed = set_index_2d_bool(
                    hist_failed, frame_idx, element_failed.astype(np.bool_)
                )
                hist_proj_pos = set_index_2d_float(hist_proj_pos, frame_idx, proj_position)
                hist_time = set_index_1d(hist_time, frame_idx, t_sim)
                hist_ke = set_index_1d(hist_ke, frame_idx, ke)
                hist_se = set_index_1d(hist_se, frame_idx, se + contact_energy)
                hist_proj_ke = set_index_1d(hist_proj_ke, frame_idx, proj_ke)
                hist_proj_quat[frame_idx, 0] = proj_quat[0]
                hist_proj_quat[frame_idx, 1] = proj_quat[1]
                hist_proj_quat[frame_idx, 2] = proj_quat[2]
                hist_proj_quat[frame_idx, 3] = proj_quat[3]

    return (
        positions,
        velocities,
        element_failed.astype(np.bool_),
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
    structure_type: str = "fabric",
    material_model: str = "linear",
    yield_strength_gpa: float = 0.0,
    hardening_modulus_gpa: float = 0.0,
    ultimate_strain: float = 0.0,
    poisson_ratio: float = 0.3,
    tensile_strength_gpa: float = 0.0,
    elements: np.ndarray | None = None,
    youngs_modulus_gpa: float = 71.0,
    thickness: float = 0.002,
    X_ref: np.ndarray | None = None,
    density_kgm3: float = 7800.0,
    proximity_threshold: float = -1.0,
    element_stress: np.ndarray | None = None,
    element_peeq: np.ndarray | None = None,
    element_damage: np.ndarray | None = None,
    element_failed: np.ndarray | None = None,
    is_tiebreak: np.ndarray | None = None,
    cohesive_strength_gpa: float = 0.485,
    fracture_energy_jm2: float = 50000.0,
    use_czm: bool = False,
    coincident_nodes: np.ndarray | None = None,
    node_czm_spring_ids: np.ndarray | None = None,
    spring_failed_step: np.ndarray | None = None,
    element_failed_step: np.ndarray | None = None,
    element_strains: np.ndarray | None = None,
    ang_positions: np.ndarray | None = None,
    ang_velocities: np.ndarray | None = None,
    ang_accel: np.ndarray | None = None,
    erosion_softening_steps: int = 10,
    velocity_clamping_multiplier: float = 2.0,
    element_peeq_rate: np.ndarray | None = None,
    rate_parameter_c: float = 40.0,
    rate_parameter_p: float = 5.0,
    softening_steps: int | None = None,
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
    n_nodes = len(positions)
    n_springs = len(grid_springs)
    if softening_steps is not None:
        erosion_softening_steps = softening_steps
    if grid_damage is None:
        grid_damage = np.zeros(n_springs, dtype=np.float64)
    if spring_failed_step is None:
        spring_failed_step = np.zeros(n_springs, dtype=np.int32) - 1
    if ang_positions is None:
        ang_positions = np.zeros((n_nodes, 3), dtype=np.float64)
    if ang_velocities is None:
        ang_velocities = np.zeros((n_nodes, 3), dtype=np.float64)
    if ang_accel is None:
        ang_accel = np.zeros((n_nodes, 3), dtype=np.float64)
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

    proximity_threshold_val = dx * 2.0 if proximity_threshold < 0.0 else proximity_threshold

    if X_ref is None:
        X_ref = positions.copy()

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

    if structure_type == "metallic_sheet":
        if elements is None:
            elements = np.zeros((0, 4), dtype=np.int32)
        n_elems = len(elements)
        if element_stress is None:
            element_stress = np.zeros((n_elems, 5, 3), dtype=positions.dtype)
        if element_peeq is None:
            element_peeq = np.zeros((n_elems, 5), dtype=positions.dtype)
        if element_peeq_rate is None:
            element_peeq_rate = np.zeros((n_elems, 5), dtype=positions.dtype)
        if element_damage is None:
            element_damage = np.zeros((n_elems, 5), dtype=positions.dtype)

        # Recover or allocate element_failed
        if element_failed is None or element_failed.shape[0] != n_elems:
            element_failed = np.zeros(n_elems, dtype=np.int32)
        else:
            element_failed = element_failed.astype(np.int32)

        if element_failed_step is None or element_failed_step.shape[0] != n_elems:
            element_failed_step = np.zeros(n_elems, dtype=np.int32) - 1
        else:
            element_failed_step = element_failed_step.astype(np.int32)

        if element_strains is None or element_strains.shape[0] != n_elems:
            element_strains = np.zeros((n_elems, 8), dtype=positions.dtype)
        if ang_positions is None or ang_positions.shape[0] != n_nodes:
            ang_positions = np.zeros((n_nodes, 3), dtype=positions.dtype)
        if ang_velocities is None or ang_velocities.shape[0] != n_nodes:
            ang_velocities = np.zeros((n_nodes, 3), dtype=positions.dtype)
        if ang_accel is None or ang_accel.shape[0] != n_nodes:
            ang_accel = np.zeros((n_nodes, 3), dtype=positions.dtype)

        if coincident_nodes is None:
            coincident_nodes = np.zeros((n_nodes, 4), dtype=np.int32) - 1
            coincident_nodes[:, 0] = np.arange(n_nodes, dtype=np.int32)
        if node_czm_spring_ids is None:
            node_czm_spring_ids = np.zeros((n_nodes, 2), dtype=np.int32) - 1

        return _fused_shell_loop_jit(
            positions,
            X_ref,
            velocities,
            grid_masses,
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
            dt,
            n_steps,
            save_interval,
            damp_dissipated_init,
            failure_dissipated_init,
            clamp_dissipated_init,
            t_sim_init,
            strike_direction,
            use_viscous,
            cfl_factor,
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
            elements,
            element_stress,
            element_peeq,
            element_damage,
            element_failed,
            yield_strength_gpa,
            hardening_modulus_gpa,
            ultimate_strain,
            poisson_ratio,
            tensile_strength_gpa,
            thickness,
            youngs_modulus_gpa,
            density_kgm3,
            proximity_threshold_val,
            grid_springs,
            grid_failed,
            grid_damage,
            is_tiebreak if is_tiebreak is not None else np.zeros(n_springs, dtype=bool),
            cohesive_strength_gpa,
            fracture_energy_jm2,
            use_czm,
            coincident_nodes,
            node_czm_spring_ids,
            element_strains,
            ang_positions,
            ang_velocities,
            ang_accel,
            element_failed_step,
            erosion_softening_steps,
            velocity_clamping_multiplier,
            element_peeq_rate,
            rate_parameter_c,
            rate_parameter_p,
        )

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
        proximity_threshold_val,
        spring_failed_step,
        erosion_softening_steps,
        velocity_clamping_multiplier,
        youngs_modulus_gpa,
        density_kgm3,
    )
