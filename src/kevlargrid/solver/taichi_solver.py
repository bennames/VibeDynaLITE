"""Taichi Lang GPU Mass-Spring Solver compute kernels.

Implements Verlet explicit time integration, spring forces, interply contact,
and projectile contact on macOS Metal or Windows CUDA/Vulkan GPUs.
"""

from typing import Any

import numpy as np

try:
    import taichi as ti
except ImportError:
    ti = None


class PhysicsViolationError(ValueError):
    """Exception raised when physical guardrails (e.g. mass-scaling energy limits) are violated."""

    pass


# Initialize Taichi on GPU (Metal, CUDA, or Vulkan)
if ti is not None:
    import contextlib

    try:
        ti.init(arch=ti.gpu, default_fp=ti.f32)
    except Exception:
        # Fallback to CPU if GPU initialization fails
        with contextlib.suppress(Exception):
            ti.init(arch=ti.cpu, default_fp=ti.f32)


@ti.data_oriented
class TaichiSolver:
    """Class containing Taichi GPU mass-spring simulation kernels."""

    def __init__(
        self,
        n_nodes: int,
        n_springs: int,
        positions_init: np.ndarray,
        velocities_init: np.ndarray,
        springs_init: np.ndarray,
        stiffnesses_init: np.ndarray,
        rest_lengths_init: np.ndarray,
        failed_init: np.ndarray,
        masses_init: np.ndarray,
        tension_only_init: np.ndarray,
        boundary_mask_init: np.ndarray,
        nodal_external_forces_init: np.ndarray,
        proj_position_init: np.ndarray,
        proj_velocity_init: np.ndarray,
        proj_mass_init: float,
        strike_direction_init: float,
        node_initial_springs_init: np.ndarray,
        masses_physical_init: np.ndarray | None = None,
        n_plies: int = 1,
        n_nodes_per_layer: int = 0,
        center_idx_init: int = 0,
        center_springs_init: np.ndarray | None = None,
    ) -> None:
        self.n_nodes = n_nodes
        self.n_springs = n_springs
        self.n_plies_val = n_plies
        self.n_nodes_per_layer_val = n_nodes_per_layer
        self._stiffnesses_id: int | None = None

        if center_springs_init is None:
            center_springs_init = np.zeros((n_plies, 4), dtype=np.int32)

        real_type = ti.f32

        # Node fields (declared without shape for custom SNode layout)
        self.positions = ti.Vector.field(3, dtype=real_type)
        self.velocities = ti.Vector.field(3, dtype=real_type)
        self.forces = ti.Vector.field(3, dtype=real_type)
        self.accel = ti.Vector.field(3, dtype=real_type)
        self.nodal_external_forces = ti.Vector.field(3, dtype=real_type)
        self.masses = ti.field(dtype=real_type)
        self.masses_physical = ti.field(dtype=real_type)
        self.boundary_mask = ti.field(dtype=ti.i32)
        self.node_initial_springs = ti.field(dtype=ti.i32)
        self.node_active_counts = ti.field(dtype=ti.i32)
        self.node_stiffness = ti.field(dtype=real_type)
        self.node_w = ti.field(dtype=real_type)
        self.dt_crit = ti.field(dtype=real_type, shape=())

        # Spring fields (declared without shape for custom SNode layout)
        self.springs = ti.Vector.field(2, dtype=ti.i32)
        self.stiffnesses = ti.field(dtype=real_type)
        self.rest_lengths = ti.field(dtype=real_type)
        self.spring_failed = ti.field(dtype=ti.i32)
        self.spring_damage = ti.field(dtype=real_type)
        self.tension_only = ti.field(dtype=ti.i32)

        # Place fields in Struct-of-Arrays (SoA) layout in memory
        node_block = ti.root.dense(ti.i, n_nodes)
        for i in range(3):
            node_block.place(self.positions.get_scalar_field(i, 0))
        for i in range(3):
            node_block.place(self.velocities.get_scalar_field(i, 0))
        for i in range(3):
            node_block.place(self.forces.get_scalar_field(i, 0))
        for i in range(3):
            node_block.place(self.accel.get_scalar_field(i, 0))
        for i in range(3):
            node_block.place(self.nodal_external_forces.get_scalar_field(i, 0))
        node_block.place(self.masses)
        node_block.place(self.masses_physical)
        node_block.place(self.boundary_mask)
        node_block.place(self.node_initial_springs)
        node_block.place(self.node_active_counts)
        node_block.place(self.node_stiffness)
        node_block.place(self.node_w)

        spring_block = ti.root.dense(ti.i, n_springs)
        for i in range(2):
            spring_block.place(self.springs.get_scalar_field(i, 0))
        spring_block.place(self.stiffnesses)
        spring_block.place(self.rest_lengths)
        spring_block.place(self.spring_failed)
        spring_block.place(self.spring_damage)
        spring_block.place(self.tension_only)

        # Projectile fields
        self.proj_position = ti.Vector.field(3, dtype=real_type, shape=())
        self.proj_velocity = ti.Vector.field(3, dtype=real_type, shape=())
        self.proj_reaction_force = ti.Vector.field(3, dtype=real_type, shape=())
        self.proj_mass = ti.field(dtype=real_type, shape=())
        self.strike_direction = ti.field(dtype=real_type, shape=())
        self.center_idx = ti.field(dtype=ti.i32, shape=())
        self.center_springs = ti.field(dtype=ti.i32, shape=(n_plies, 4))

        # 6-DOF state fields
        self.proj_quat = ti.Vector.field(4, dtype=real_type, shape=())
        self.proj_omega = ti.Vector.field(3, dtype=real_type, shape=())
        self.proj_torque = ti.Vector.field(3, dtype=real_type, shape=())
        self.proj_inertia_inv = ti.Vector.field(3, dtype=real_type, shape=())
        self.proj_shape_type = ti.field(dtype=ti.i32, shape=())
        self.proj_accel = ti.Vector.field(3, dtype=real_type, shape=())
        self.proj_omega_dot = ti.Vector.field(3, dtype=real_type, shape=())

        # Shape parameter fields
        self.radius_field = ti.field(dtype=real_type, shape=())
        self.length_field = ti.field(dtype=real_type, shape=())
        self.edge_radius_field = ti.field(dtype=real_type, shape=())
        self.ogive_multiplier_field = ti.field(dtype=real_type, shape=())
        self.span_field = ti.field(dtype=real_type, shape=())
        self.root_chord_field = ti.field(dtype=real_type, shape=())
        self.tip_chord_field = ti.field(dtype=real_type, shape=())
        self.twist_field = ti.field(dtype=real_type, shape=())
        self.thickness_ratio_field = ti.field(dtype=real_type, shape=())
        self.tip_radius_field = ti.field(dtype=real_type, shape=())
        self.z_com_field = ti.field(dtype=real_type, shape=())
        self.y_com_field = ti.field(dtype=real_type, shape=())
        self.c_damping = ti.field(dtype=real_type, shape=())
        self.peak_deceleration_g = ti.field(dtype=real_type, shape=())

        # Dissipated energy fields
        self.damp_dissipated = ti.field(dtype=real_type, shape=())
        self.failure_dissipated = ti.field(dtype=real_type, shape=())
        self.clamp_dissipated = ti.field(dtype=real_type, shape=())
        self.t_sim = ti.field(dtype=real_type, shape=())
        self.E_artificial_kinetic = ti.field(dtype=real_type, shape=())
        self.physics_violated = ti.field(dtype=ti.i32, shape=())
        self.contact_energy = ti.field(dtype=real_type, shape=())

        # GPU-side telemetry scalars
        self.telem_ke = ti.field(dtype=real_type, shape=())
        self.telem_se = ti.field(dtype=real_type, shape=())
        self.telem_peak_strain = ti.field(dtype=real_type, shape=())
        self.telem_proj_ke = ti.field(dtype=real_type, shape=())
        self.telem_failed_count = ti.field(dtype=ti.i32, shape=())

        # Graph execution scalar fields
        self.dt = ti.field(dtype=real_type, shape=())
        self.v_max = ti.field(dtype=real_type, shape=())
        self.rayleigh_beta = ti.field(dtype=real_type, shape=())
        self.damage_onset_strain = ti.field(dtype=real_type, shape=())
        self.failure_strain = ti.field(dtype=real_type, shape=())
        self.n_plies = ti.field(dtype=ti.i32, shape=())
        self.n_nodes_per_layer = ti.field(dtype=ti.i32, shape=())
        self.t_ply = ti.field(dtype=real_type, shape=())
        self.k_penalty = ti.field(dtype=real_type, shape=())
        self.w_h = ti.field(dtype=real_type, shape=())
        self.t_h = ti.field(dtype=real_type, shape=())
        self.proximity_threshold = ti.field(dtype=real_type, shape=())
        self.rayleigh_alpha = ti.field(dtype=real_type, shape=())
        self.use_viscous = ti.field(dtype=ti.i32, shape=())
        self.cfl_factor = ti.field(dtype=real_type, shape=())
        self.dx = ti.field(dtype=real_type, shape=())
        self.fracture_energy_multiplier = ti.field(dtype=real_type, shape=())

        # Helper reduction fields for projectile contact
        self.w_sum = ti.field(dtype=real_type, shape=())
        self.n_contacts = ti.field(dtype=ti.i32, shape=())

        # Initialize GPU buffers from numpy inputs
        self.positions.from_numpy(positions_init.astype(np.float32))
        self.velocities.from_numpy(velocities_init.astype(np.float32))
        self.masses.from_numpy(masses_init.astype(np.float32))
        if masses_physical_init is not None:
            self.masses_physical.from_numpy(masses_physical_init.astype(np.float32))
        else:
            self.masses_physical.from_numpy(masses_init.astype(np.float32))
        self.boundary_mask.from_numpy(boundary_mask_init.astype(np.int32))
        self.nodal_external_forces.from_numpy(nodal_external_forces_init.astype(np.float32))
        self.node_initial_springs.from_numpy(node_initial_springs_init.astype(np.int32))

        self.springs.from_numpy(springs_init.astype(np.int32))
        self.stiffnesses.from_numpy(stiffnesses_init.astype(np.float32))
        self.rest_lengths.from_numpy(rest_lengths_init.astype(np.float32))
        self.spring_failed.from_numpy(failed_init.astype(np.int32))
        self.spring_damage.from_numpy(failed_init.astype(np.float32))
        self.tension_only.from_numpy(tension_only_init.astype(np.int32))

        self.proj_position[None] = proj_position_init.astype(np.float32)
        self.proj_velocity[None] = proj_velocity_init.astype(np.float32)
        self.proj_mass[None] = float(proj_mass_init)
        self.strike_direction[None] = float(strike_direction_init)
        self.center_idx[None] = center_idx_init
        self.center_springs.from_numpy(center_springs_init.astype(np.int32))

        self.proj_reaction_force[None] = [0.0, 0.0, 0.0]
        self.proj_quat[None] = [1.0, 0.0, 0.0, 0.0]
        self.proj_omega[None] = [0.0, 0.0, 0.0]
        self.proj_torque[None] = [0.0, 0.0, 0.0]
        self.proj_inertia_inv[None] = [1.0, 1.0, 1.0]
        self.proj_shape_type[None] = 0
        self.proj_accel[None] = [0.0, 0.0, 0.0]
        self.proj_omega_dot[None] = [0.0, 0.0, 0.0]
        self.radius_field[None] = 0.005
        self.length_field[None] = 0.01
        self.edge_radius_field[None] = 0.0
        self.ogive_multiplier_field[None] = 2.0
        self.span_field[None] = 0.05
        self.root_chord_field[None] = 0.01
        self.tip_chord_field[None] = 0.005
        self.twist_field[None] = 15.0
        self.thickness_ratio_field[None] = 12.0
        self.tip_radius_field[None] = 0.002
        self.z_com_field[None] = 0.0
        self.y_com_field[None] = 0.0
        self.c_damping[None] = 0.0
        self.peak_deceleration_g[None] = 0.0
        self.damp_dissipated[None] = 0.0
        self.failure_dissipated[None] = 0.0
        self.clamp_dissipated[None] = 0.0
        self.t_sim[None] = 0.0
        self.E_artificial_kinetic[None] = 0.0
        self.physics_violated[None] = 0

        self.telem_ke[None] = 0.0
        self.telem_se[None] = 0.0
        self.telem_peak_strain[None] = 0.0
        self.telem_proj_ke[None] = 0.0
        self.telem_failed_count[None] = 0

        # Initialize Graph execution scalar fields
        self.dt[None] = 0.0
        self.v_max[None] = 0.0
        self.rayleigh_beta[None] = 0.0
        self.damage_onset_strain[None] = 0.0
        self.failure_strain[None] = 0.0
        self.n_plies[None] = n_plies
        self.n_nodes_per_layer[None] = n_nodes_per_layer
        self.t_ply[None] = 0.0
        self.k_penalty[None] = 0.0
        self.w_h[None] = 0.0
        self.t_h[None] = 0.0
        self.proximity_threshold[None] = 0.0
        self.rayleigh_alpha[None] = 0.0
        self.use_viscous[None] = 0
        self.cfl_factor[None] = -1.0
        self.dx[None] = 0.0
        self.fracture_energy_multiplier[None] = 1.0

        # Define kernels inside __init__ to capture self and access fields
        @ti.kernel
        def k_reset_forces_g():
            for i in range(self.n_nodes):
                self.forces[i] = ti.Vector([0.0, 0.0, 0.0])
            self.proj_reaction_force[None] = ti.Vector([0.0, 0.0, 0.0])
            self.proj_torque[None] = ti.Vector([0.0, 0.0, 0.0])
            self.contact_energy[None] = 0.0

        self.k_reset_forces_graph = k_reset_forces_g

        @ti.kernel
        def k_compute_active_counts_g():
            for i in range(self.n_nodes):
                self.node_active_counts[i] = 0
            for j in range(self.n_springs):
                if self.spring_failed[j] == 0:
                    u, v = self.springs[j][0], self.springs[j][1]
                    ti.atomic_add(self.node_active_counts[u], 1)
                    ti.atomic_add(self.node_active_counts[v], 1)

        self.k_compute_active_counts_graph = k_compute_active_counts_g

        @ti.kernel
        def k_update_cfl_g():
            # Clear nodal stiffnesses
            for i in range(self.n_nodes):
                self.node_stiffness[i] = 0.0

            # Sum spring stiffnesses
            for j in range(self.n_springs):
                if self.spring_failed[j] == 0:
                    u, v = self.springs[j][0], self.springs[j][1]
                    diff = self.positions[v] - self.positions[u]
                    length = diff.norm()
                    strain = (length - self.rest_lengths[j]) / self.rest_lengths[j]

                    denom = self.failure_strain[None] - self.damage_onset_strain[None]
                    denom_safe = denom if denom != 0.0 else 1.0
                    val = (strain - self.damage_onset_strain[None]) / denom_safe
                    d_val = 0.0
                    if val > 0.0:
                        d_val = val if val < 1.0 else 1.0

                    damage_effective = ti.max(self.spring_damage[j], d_val)
                    effective_k = self.stiffnesses[j] * (1.0 - damage_effective)

                    ti.atomic_add(self.node_stiffness[u], effective_k)
                    ti.atomic_add(self.node_stiffness[v], effective_k)

            # Projectile contact stiffness (SDF or IDW)
            k_penalty = self.k_penalty[None]
            proximity_threshold = self.proximity_threshold[None]
            if self.proj_shape_type[None] == 0:
                self.w_sum[None] = 0.0
                self.n_contacts[None] = 0
                proj_pos = self.proj_position[None]
                w_h = self.w_h[None]
                t_h = self.t_h[None]

                for i in range(self.n_nodes):
                    self.node_w[i] = 0.0
                    px, py, pz = self.positions[i].x, self.positions[i].y, self.positions[i].z
                    if (
                        px >= proj_pos.x - w_h - proximity_threshold
                        and px <= proj_pos.x + w_h + proximity_threshold
                        and py >= proj_pos.y - t_h - proximity_threshold
                        and py <= proj_pos.y + t_h + proximity_threshold
                        and ti.abs(pz - proj_pos.z) <= proximity_threshold
                    ):
                        x_proj = ti.max(proj_pos.x - w_h, ti.min(px, proj_pos.x + w_h))
                        y_proj = ti.max(proj_pos.y - t_h, ti.min(py, proj_pos.y + t_h))
                        dist = ti.sqrt(
                            (px - x_proj) ** 2 + (py - y_proj) ** 2 + (pz - proj_pos.z) ** 2
                        )

                        if dist <= proximity_threshold:
                            w = 1.0 / ti.max(dist, 1e-4)
                            self.node_w[i] = w
                            ti.atomic_add(self.w_sum[None], w)
                            ti.atomic_add(self.n_contacts[None], 1)

                w_mean = 0.0
                if self.n_contacts[None] > 0:
                    w_mean = self.w_sum[None] / float(self.n_contacts[None])
                for i in range(self.n_nodes):
                    if self.node_w[i] > 0.0 and w_mean > 0.0:
                        w_normalized = self.node_w[i] / w_mean
                        scale_factor = 0.0
                        if self.node_initial_springs[i] > 0:
                            scale_factor = float(self.node_active_counts[i]) / float(
                                self.node_initial_springs[i]
                            )
                        ti.atomic_add(
                            self.node_stiffness[i], k_penalty * w_normalized * scale_factor
                        )
            else:
                # SDF bounding volume contact stiffness check
                proj_pos = self.proj_position[None]
                max_R = ti.max(
                    self.radius_field[None], ti.max(self.length_field[None], self.span_field[None])
                )
                cutoff = max_R + proximity_threshold
                for i in range(self.n_nodes):
                    dist = (self.positions[i] - proj_pos).norm()
                    if dist <= cutoff:
                        ti.atomic_add(self.node_stiffness[i], k_penalty)

            # Inter-ply contact stiffness using absolute gap formulation
            if self.n_plies_val > 1:
                for u in range((self.n_plies_val - 1) * self.n_nodes_per_layer_val):
                    v = u + self.n_nodes_per_layer_val
                    gap = ti.abs(self.positions[u].z - self.positions[v].z)
                    penetration = self.t_ply[None] - gap
                    if penetration > 0.0:
                        if self.node_active_counts[u] > 0 and self.node_active_counts[v] > 0:
                            ti.atomic_add(self.node_stiffness[u], k_penalty)
                            ti.atomic_add(self.node_stiffness[v], k_penalty)

            # Compute minimum dt
            self.dt_crit[None] = 1e10
            for i in range(self.n_nodes):
                k_i = ti.max(self.node_stiffness[i], 1e-4)
                dt_i = ti.sqrt(self.masses[i] / k_i)
                ti.atomic_min(self.dt_crit[None], dt_i)

            dt_crit = self.cfl_factor[None] * self.dt_crit[None]
            self.dt[None] = dt_crit
            self.v_max[None] = self.dx[None] / dt_crit

        self.k_update_cfl_graph = k_update_cfl_g

        @ti.kernel
        def k_fused_spring_pass_g():
            ti.block_local(self.positions)
            ti.block_local(self.forces)

            # Phase A: zero active counts and forces (fused reset forces)
            for i in range(self.n_nodes):
                self.node_active_counts[i] = 0
                self.forces[i] = ti.Vector([0.0, 0.0, 0.0])
            self.proj_reaction_force[None] = ti.Vector([0.0, 0.0, 0.0])
            self.contact_energy[None] = 0.0

            # Phase C: fused spring traversal
            for j in range(self.n_springs):
                if self.spring_failed[j] == 1:
                    continue

                u, v = self.springs[j][0], self.springs[j][1]

                # --- Shared geometry (computed ONCE) ---
                diff = self.positions[v] - self.positions[u]
                length = diff.norm()
                length_safe = length if length > 0.0 else 1.0
                strain = (length - self.rest_lengths[j]) / self.rest_lengths[j]

                # --- Active count ---
                ti.atomic_add(self.node_active_counts[u], 1)
                ti.atomic_add(self.node_active_counts[v], 1)

                # --- Damage evolution ---
                denom = self.failure_strain[None] - self.damage_onset_strain[None]
                denom_safe = denom if denom != 0.0 else 1.0
                val = (strain - self.damage_onset_strain[None]) / denom_safe
                d_val = 0.0
                if val > 0.0:
                    d_val = val if val < 1.0 else 1.0
                if d_val > self.spring_damage[j]:
                    d_old = self.spring_damage[j]
                    d_new = d_val
                    self.spring_damage[j] = d_new

                    # Calculate energy increment
                    k = self.stiffnesses[j]
                    L0 = self.rest_lengths[j]
                    x_onset = self.damage_onset_strain[None]
                    x_fail = self.failure_strain[None]

                    h = x_fail - x_onset
                    h_safe = h if h != 0.0 else 1.0

                    w_old = 0.0
                    if d_old > 0.0:
                        x_peak_old = x_onset + d_old * h
                        w_old = (k * L0**2 / (6.0 * h_safe)) * (x_peak_old**3 - x_onset**3)

                    w_new = 0.0
                    if d_new >= 1.0:
                        w_new = (k * L0**2 / 6.0) * (x_fail**2 + x_fail * x_onset + x_onset**2)
                        self.spring_failed[j] = 1
                    else:
                        x_peak_new = x_onset + d_new * h
                        w_new = (k * L0**2 / (6.0 * h_safe)) * (x_peak_new**3 - x_onset**3)

                    dw = w_new - w_old
                    ti.atomic_add(
                        self.failure_dissipated[None], self.fracture_energy_multiplier[None] * dw
                    )

                if self.spring_failed[j] == 1:
                    continue

                # --- Force computation ---
                effective_k = self.stiffnesses[j] * (1.0 - self.spring_damage[j])
                f_mag = effective_k * strain * self.rest_lengths[j]
                if self.tension_only[j] == 1 and strain < 0.0:
                    f_mag = 0.0
                f_vec = (f_mag / length_safe) * diff
                ti.atomic_add(self.forces[u], f_vec)
                ti.atomic_add(self.forces[v], -f_vec)

                # Stiffness-proportional damping
                rayleigh_beta = self.rayleigh_beta[None]
                if self.use_viscous[None] == 1:
                    rayleigh_beta = 0.0
                if rayleigh_beta > 0.0:
                    dv = self.velocities[v] - self.velocities[u]
                    v_proj = dv.dot(diff / length_safe)
                    damp_mag = rayleigh_beta * effective_k * v_proj
                    damp_vec = (damp_mag / length_safe) * diff
                    ti.atomic_add(self.forces[u], damp_vec)
                    ti.atomic_add(self.forces[v], -damp_vec)
                    ti.atomic_add(self.damp_dissipated[None], damp_mag * v_proj * self.dt[None])

        self.k_fused_spring_pass_graph = k_fused_spring_pass_g

        @ti.kernel
        def k_compute_projectile_forces_g():
            self.compute_projectile_forces(
                self.w_h[None],
                self.t_h[None],
                self.k_penalty[None],
                self.proximity_threshold[None],
            )

        self.k_compute_projectile_forces_graph = k_compute_projectile_forces_g

        @ti.kernel
        def k_compute_interply_forces_g():
            t_ply = self.t_ply[None]
            k_penalty = self.k_penalty[None]
            if self.n_plies_val > 1:
                for u in range((self.n_plies_val - 1) * self.n_nodes_per_layer_val):
                    v = u + self.n_nodes_per_layer_val
                    zu = self.positions[u].z
                    zv = self.positions[v].z
                    gap = ti.abs(zu - zv)
                    penetration = t_ply - gap
                    if penetration > 0.0:
                        if self.node_active_counts[u] > 0 and self.node_active_counts[v] > 0:
                            f_mag = k_penalty * penetration
                            direction = 1.0 if zu > zv else -1.0
                            ti.atomic_add(self.forces[u].z, f_mag * direction)
                            ti.atomic_add(self.forces[v].z, -f_mag * direction)
                            ti.atomic_add(
                                self.contact_energy[None],
                                0.5 * k_penalty * penetration * penetration,
                            )

        self.k_compute_interply_forces_graph = k_compute_interply_forces_g

        @ti.kernel
        def k_apply_impedance_boundary_g():
            for i in range(self.n_nodes):
                if self.boundary_mask[i] == 2:
                    self.node_stiffness[i] = 0.0

            for j in range(self.n_springs):
                if self.spring_failed[j] == 0:
                    u, v = self.springs[j][0], self.springs[j][1]
                    effective_k = self.stiffnesses[j] * (1.0 - self.spring_damage[j])
                    if self.boundary_mask[u] == 2:
                        ti.atomic_add(self.node_stiffness[u], effective_k)
                    if self.boundary_mask[v] == 2:
                        ti.atomic_add(self.node_stiffness[v], effective_k)

            for i in range(self.n_nodes):
                if self.boundary_mask[i] == 2:
                    k_eff = ti.max(self.node_stiffness[i], 1e-4)
                    C_i = ti.sqrt(self.masses[i] * k_eff)
                    f_boundary = -C_i * self.velocities[i]
                    self.forces[i] += f_boundary

        self.k_apply_impedance_boundary_graph = k_apply_impedance_boundary_g

        @ti.kernel
        def k_compute_initial_accelerations_g():
            rayleigh_alpha = self.rayleigh_alpha[None]
            use_viscous = self.use_viscous[None]
            for i in range(self.n_nodes):
                if self.boundary_mask[i] == 1:
                    self.forces[i] = ti.Vector([0.0, 0.0, 0.0])
                    self.velocities[i] = ti.Vector([0.0, 0.0, 0.0])
                    self.accel[i] = ti.Vector([0.0, 0.0, 0.0])
                    continue
                damp_f = ti.Vector([0.0, 0.0, 0.0])
                if use_viscous == 1:
                    damp_f = -rayleigh_alpha * self.velocities[i]
                else:
                    damp_f = -rayleigh_alpha * self.masses[i] * self.velocities[i]
                net_f = self.forces[i] + damp_f + self.nodal_external_forces[i]
                self.accel[i] = net_f / self.masses[i]

            self.proj_accel[None] = self.proj_reaction_force[None] / self.proj_mass[None]
            if self.proj_shape_type[None] > 0:
                self.proj_omega_dot[None] = self.proj_torque[None] * self.proj_inertia_inv[None]

        self.k_compute_initial_accelerations_graph = k_compute_initial_accelerations_g

        @ti.kernel
        def k_integrate_nodes_half_step_g():
            dt = self.dt[None]
            for i in range(self.n_nodes):
                if self.boundary_mask[i] == 1:
                    continue
                self.velocities[i] += 0.5 * self.accel[i] * dt
                self.positions[i] += self.velocities[i] * dt

            self.proj_velocity[None] += 0.5 * self.proj_accel[None] * dt
            self.proj_position[None] += self.proj_velocity[None] * dt

            if self.proj_shape_type[None] > 0:
                self.proj_omega[None] += 0.5 * self.proj_omega_dot[None] * dt

                omega = self.proj_omega[None]
                q = self.proj_quat[None]
                qw, qx, qy, qz = q[0], q[1], q[2], q[3]
                ox, oy, oz = omega[0], omega[1], omega[2]

                dq_w = -ox * qx - oy * qy - oz * qz
                dq_x = ox * qw + oy * qz - oz * qy
                dq_y = -ox * qz + oy * qw + oz * qx
                dq_z = ox * qy - oy * qx + oz * qw

                q_new = ti.Vector([qw, qx, qy, qz]) + 0.5 * dt * ti.Vector([dq_w, dq_x, dq_y, dq_z])
                q_norm = q_new.norm()
                if q_norm > 1e-8:
                    self.proj_quat[None] = q_new / q_norm

        self.k_integrate_nodes_half_step_graph = k_integrate_nodes_half_step_g

        @ti.kernel
        def k_integrate_nodes_full_step_g():
            self.E_artificial_kinetic[None] = 0.0

            dt = self.dt[None]
            rayleigh_alpha = self.rayleigh_alpha[None]
            v_max = self.v_max[None]
            use_viscous = self.use_viscous[None]

            for i in range(self.n_nodes):
                if self.boundary_mask[i] == 1:
                    self.forces[i] = ti.Vector([0.0, 0.0, 0.0])
                    self.velocities[i] = ti.Vector([0.0, 0.0, 0.0])
                    self.accel[i] = ti.Vector([0.0, 0.0, 0.0])
                    continue

                damp_f = ti.Vector([0.0, 0.0, 0.0])
                if use_viscous == 1:
                    damp_f = -rayleigh_alpha * self.velocities[i]
                else:
                    damp_f = -rayleigh_alpha * self.masses[i] * self.velocities[i]
                p_damp = damp_f.dot(self.velocities[i])
                ti.atomic_add(self.damp_dissipated[None], -p_damp * dt)

                net_f = self.forces[i] + damp_f + self.nodal_external_forces[i]
                self.accel[i] = net_f / self.masses[i]

                self.velocities[i] += 0.5 * self.accel[i] * dt

                v_mag = self.velocities[i].norm()
                v_sq = self.velocities[i].norm_sqr()
                if v_mag > v_max:
                    scale = v_max / v_mag
                    excess_ke = 0.5 * self.masses[i] * (v_sq - v_max * v_max)
                    ti.atomic_add(self.clamp_dissipated[None], excess_ke)
                    self.velocities[i] *= scale
                    v_sq = v_max * v_max

                # Compute artificial KE inline
                m_scaled = self.masses[i]
                m_phys = self.masses_physical[i]
                ti.atomic_add(self.E_artificial_kinetic[None], 0.5 * (m_scaled - m_phys) * v_sq)

            # Projectile integration (scalar)
            self.proj_accel[None] = self.proj_reaction_force[None] / self.proj_mass[None]
            self.proj_velocity[None] += 0.5 * self.proj_accel[None] * dt
            self.t_sim[None] += dt

            if self.proj_shape_type[None] > 0:
                self.proj_omega_dot[None] = self.proj_torque[None] * self.proj_inertia_inv[None]
                self.proj_omega[None] += 0.5 * self.proj_omega_dot[None] * dt

            # Telemetry peak deceleration
            accel_mag = self.proj_accel[None].norm()
            accel_mag_g = accel_mag / 9.80665
            if accel_mag_g > self.peak_deceleration_g[None]:
                self.peak_deceleration_g[None] = accel_mag_g

        self.k_integrate_nodes_full_step_graph = k_integrate_nodes_full_step_g

        @ti.kernel
        def k_guardrail_check_g():
            e_int = 0.0
            for j in range(self.n_springs):
                if self.spring_failed[j] == 0:
                    u, v = self.springs[j][0], self.springs[j][1]
                    diff = self.positions[v] - self.positions[u]
                    length = diff.norm()
                    strain = (length - self.rest_lengths[j]) / self.rest_lengths[j]
                    effective_k = self.stiffnesses[j] * (1.0 - self.spring_damage[j])
                    ti.atomic_add(e_int, 0.5 * effective_k * (strain * self.rest_lengths[j]) ** 2)

            # sync barrier
            e_total_int = (
                e_int
                + self.contact_energy[None]
                + self.failure_dissipated[None]
                + self.damp_dissipated[None]
                + self.clamp_dissipated[None]
            )
            if e_total_int > 0.0:
                if self.E_artificial_kinetic[None] > 0.02 * e_total_int:
                    self.physics_violated[None] = 1

        self.k_guardrail_check_graph = k_guardrail_check_g
        self.compiled_graphs: dict[tuple[int, bool, bool, int], Any] = {}

    @ti.func
    def ti_q_mul(self, q1, q2):
        w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
        w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
        return ti.Vector(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]
        )

    @ti.func
    def ti_q_rotate(self, q, v):
        q_v = ti.Vector([0.0, v.x, v.y, v.z])
        q_conj = ti.Vector([q[0], -q[1], -q[2], -q[3]])
        q_rot = self.ti_q_mul(self.ti_q_mul(q, q_v), q_conj)
        return ti.Vector([q_rot[1], q_rot[2], q_rot[3]])

    @ti.func
    def sdf_sphere(self, p, R):
        return p.norm() - R

    @ti.func
    def sdf_cylinder(self, p, R, L, R_e):
        d_cyl = ti.sqrt(p.x**2 + p.y**2) - (R - R_e)
        d_len = ti.abs(p.z) - (L / 2.0 - R_e)
        ext_d = ti.sqrt(ti.max(0.0, d_cyl) ** 2 + ti.max(0.0, d_len) ** 2)
        int_d = ti.min(0.0, ti.max(d_cyl, d_len))
        return ext_d + int_d - R_e

    @ti.func
    def sdf_bullet(self, p, R0, R_og, L_body, L_nose, z_com):
        z_geom = p.z + z_com
        r = ti.sqrt(p.x**2 + p.y**2)
        val = 0.0
        if z_geom < 0.0:
            d_cyl = r - R0
            d_cap = -z_geom - L_body
            val = ti.max(d_cyl, d_cap)
        else:
            if z_geom > L_nose:
                val = ti.sqrt(r**2 + (z_geom - L_nose) ** 2)
            else:
                r_c = R0 - R_og
                dist_to_center = ti.sqrt((r - r_c) ** 2 + z_geom**2)
                val = dist_to_center - R_og
        return val

    @ti.func
    def sdf_propeller_slice(self, x_prime, z_prime, c, tau, R_tip):
        u = (x_prime + c / 2.0) / c
        u_clamped = ti.min(ti.max(u, 0.0), 1.0)
        t = (
            5.0
            * tau
            * (
                0.2969 * ti.sqrt(u_clamped)
                - 0.1260 * u_clamped
                - 0.3516 * (u_clamped**2)
                + 0.2843 * (u_clamped**3)
                - 0.1015 * (u_clamped**4)
            )
            * c
        )
        half_t = ti.max(t / 2.0, R_tip)
        val = 0.0
        if x_prime < -c / 2.0 + R_tip:
            dist_le = ti.sqrt((x_prime - (-c / 2.0 + R_tip)) ** 2 + z_prime**2)
            val = dist_le - R_tip
        elif x_prime > c / 2.0 - R_tip:
            dist_te = ti.sqrt((x_prime - (c / 2.0 - R_tip)) ** 2 + z_prime**2)
            val = dist_te - R_tip
        else:
            val = ti.abs(z_prime) - half_t
        return val

    @ti.func
    def sdf_propeller(self, p, S, c_r, c_t, twist_deg, thickness_ratio, R_tip, y_com):
        y_geom = p.y + y_com
        val = 0.0
        if y_geom > S - R_tip:
            val = ti.sqrt(p.x**2 + (y_geom - (S - R_tip)) ** 2 + p.z**2) - R_tip
        elif y_geom < 0.0:
            u = (p.x + c_r / 2.0) / c_r
            u_clamped = ti.min(ti.max(u, 0.0), 1.0)
            t = (
                5.0
                * (thickness_ratio / 100.0)
                * (
                    0.2969 * ti.sqrt(u_clamped)
                    - 0.1260 * u_clamped
                    - 0.3516 * (u_clamped**2)
                    + 0.2843 * (u_clamped**3)
                    - 0.1015 * (u_clamped**4)
                )
                * c_r
            )
            half_t = ti.max(t / 2.0, R_tip)
            d_slice = ti.abs(p.z) - half_t
            val = ti.max(-y_geom, d_slice)
        else:
            c = c_r + (y_geom / S) * (c_t - c_r)
            theta = (twist_deg * 3.141592653589793 / 180.0) * (y_geom / S)
            xr = p.x * ti.cos(theta) + p.z * ti.sin(theta)
            zr = -p.x * ti.sin(theta) + p.z * ti.cos(theta)
            val = self.sdf_propeller_slice(xr, zr, c, thickness_ratio / 100.0, R_tip)
        return val

    @ti.func
    def eval_sdf(self, p):
        val = 0.0
        shape = self.proj_shape_type[None]
        if shape == 1:
            val = self.sdf_sphere(p, self.radius_field[None])
        elif shape == 2:
            val = self.sdf_cylinder(
                p, self.radius_field[None], self.length_field[None], self.edge_radius_field[None]
            )
        elif shape == 3:
            R0 = self.radius_field[None]
            R_og = R0 * self.ogive_multiplier_field[None]
            L_nose = ti.sqrt(2.0 * R_og * R0 - R0**2)
            L_body = ti.max(0.0, self.length_field[None] - L_nose)
            val = self.sdf_bullet(p, R0, R_og, L_body, L_nose, self.z_com_field[None])
        elif shape == 4:
            val = self.sdf_propeller(
                p,
                self.span_field[None],
                self.root_chord_field[None],
                self.tip_chord_field[None],
                self.twist_field[None],
                self.thickness_ratio_field[None],
                self.tip_radius_field[None],
                self.y_com_field[None],
            )
        else:
            w_h = self.w_h[None]
            t_h = self.t_h[None]
            x_proj = ti.max(-w_h, ti.min(p.x, w_h))
            y_proj = ti.max(-t_h, ti.min(p.y, t_h))
            val = ti.sqrt((p.x - x_proj) ** 2 + (p.y - y_proj) ** 2 + p.z**2)
        return val

    @ti.func
    def eval_sdf_normal(self, p):
        h = 1e-5
        grad = ti.Vector([0.0, 0.0, 0.0])
        grad.x = (
            self.eval_sdf(p + ti.Vector([h, 0.0, 0.0]))
            - self.eval_sdf(p - ti.Vector([h, 0.0, 0.0]))
        ) / (2.0 * h)
        grad.y = (
            self.eval_sdf(p + ti.Vector([0.0, h, 0.0]))
            - self.eval_sdf(p - ti.Vector([0.0, h, 0.0]))
        ) / (2.0 * h)
        grad.z = (
            self.eval_sdf(p + ti.Vector([0.0, 0.0, h]))
            - self.eval_sdf(p - ti.Vector([0.0, 0.0, h]))
        ) / (2.0 * h)
        norm = grad.norm()
        return grad / norm if norm > 1e-8 else ti.Vector([0.0, 0.0, 1.0])

    @ti.func
    def reset_forces(self):
        """Clear dynamic nodal forces and projectile reaction forces."""
        for i in range(self.n_nodes):
            self.forces[i] = ti.Vector([0.0, 0.0, 0.0])
        self.proj_reaction_force[None] = ti.Vector([0.0, 0.0, 0.0])
        self.proj_torque[None] = ti.Vector([0.0, 0.0, 0.0])
        self.contact_energy[None] = 0.0

    @ti.func
    def compute_spring_forces(
        self, rayleigh_beta: ti.f32, damage_onset_strain: ti.f32, failure_strain: ti.f32, dt: ti.f32
    ):
        """Compute structural mass-spring forces and stiffness-proportional damping forces."""
        for j in range(self.n_springs):
            if self.spring_failed[j] == 1:
                continue

            u, v = self.springs[j][0], self.springs[j][1]
            diff = self.positions[v] - self.positions[u]
            length = diff.norm()
            length_safe = length if length > 0.0 else 1.0
            strain = (length - self.rest_lengths[j]) / self.rest_lengths[j]

            effective_k = self.stiffnesses[j] * (1.0 - self.spring_damage[j])

            f_mag = effective_k * strain * self.rest_lengths[j]

            # Orthogonal/tension-only spring logic
            if self.tension_only[j] == 1 and strain < 0.0:
                f_mag = 0.0

            f_vec = (f_mag / length_safe) * diff

            # Atomic force accumulations
            ti.atomic_add(self.forces[u], f_vec)
            ti.atomic_add(self.forces[v], -f_vec)

            # Stiffness-proportional damping
            if rayleigh_beta > 0.0:
                dv = self.velocities[v] - self.velocities[u]
                v_proj = dv.dot(diff / length_safe)
                damp_mag = rayleigh_beta * effective_k * v_proj
                damp_vec = (damp_mag / length_safe) * diff
                ti.atomic_add(self.forces[u], damp_vec)
                ti.atomic_add(self.forces[v], -damp_vec)
                ti.atomic_add(self.damp_dissipated[None], damp_mag * v_proj * dt)

    @ti.func
    def compute_interply_forces(
        self, n_nodes_per_layer: ti.i32, n_plies: ti.i32, t_ply: ti.f32, k_penalty: ti.f32
    ):
        """Compute contact force penalty values preventing interply penetration."""
        for u in range((n_plies - 1) * n_nodes_per_layer):
            v = u + n_nodes_per_layer
            gap = ti.abs(self.positions[u].z - self.positions[v].z)
            penetration = t_ply - gap
            if penetration > 0.0:
                if self.node_active_counts[u] > 0 and self.node_active_counts[v] > 0:
                    f_mag = k_penalty * penetration
                    direction = 1.0 if self.positions[u].z > self.positions[v].z else -1.0
                    ti.atomic_add(self.forces[u].z, f_mag * direction)
                    ti.atomic_add(self.forces[v].z, -f_mag * direction)
                    ti.atomic_add(self.contact_energy[None], 0.5 * k_penalty * penetration**2)

    @ti.func
    def compute_active_counts(self):
        """Compute the count of active (non-failed) springs connected to each node."""
        for i in range(self.n_nodes):
            self.node_active_counts[i] = 0
        for j in range(self.n_springs):
            if self.spring_failed[j] == 0:
                u, v = self.springs[j][0], self.springs[j][1]
                ti.atomic_add(self.node_active_counts[u], 1)
                ti.atomic_add(self.node_active_counts[v], 1)

    @ti.func
    def compute_projectile_forces(
        self, w_h: ti.f32, t_h: ti.f32, k_penalty: ti.f32, proximity_threshold: ti.f32
    ):
        """Compute blade-to-mesh contact interface force distribution."""
        if self.proj_shape_type[None] == 0:
            # Legacy box/blade IDW contact force
            self.w_sum[None] = 0.0
            self.n_contacts[None] = 0

            direction = self.strike_direction[None]
            if direction == 0.0:
                direction = 1.0
                if self.proj_velocity[None].z < 0.0:
                    direction = -1.0

            proj_pos = self.proj_position[None]

            # Phase 1: Determine contact nodes and compute weights
            for i in range(self.n_nodes):
                self.node_w[i] = 0.0
                px, py, pz = self.positions[i].x, self.positions[i].y, self.positions[i].z
                if (
                    px >= proj_pos.x - w_h - proximity_threshold
                    and px <= proj_pos.x + w_h + proximity_threshold
                    and py >= proj_pos.y - t_h - proximity_threshold
                    and py <= proj_pos.y + t_h + proximity_threshold
                    and ti.abs(pz - proj_pos.z) <= proximity_threshold
                ):
                    x_proj = ti.max(proj_pos.x - w_h, ti.min(px, proj_pos.x + w_h))
                    y_proj = ti.max(proj_pos.y - t_h, ti.min(py, proj_pos.y + t_h))
                    dist = ti.sqrt((px - x_proj) ** 2 + (py - y_proj) ** 2 + (pz - proj_pos.z) ** 2)

                    if dist <= proximity_threshold:
                        w = 1.0 / ti.max(dist, 1e-4)
                        self.node_w[i] = w
                        ti.atomic_add(self.w_sum[None], w)
                        ti.atomic_add(self.n_contacts[None], 1)

            # Phase 2: Distribute force based on IDW weights
            if self.n_contacts[None] > 0:
                w_mean = self.w_sum[None] / float(self.n_contacts[None])
                for i in range(self.n_nodes):
                    if self.node_w[i] > 0.0:
                        w_normalized = self.node_w[i] / w_mean if w_mean > 0.0 else self.node_w[i]
                        n_nodes_layer = (
                            self.n_nodes_per_layer_val
                            if self.n_nodes_per_layer_val > 0
                            else self.n_nodes
                        )
                        layer_idx = i // n_nodes_layer
                        is_ruptured = (
                            self.spring_failed[self.center_springs[layer_idx, 0]] == 1
                            and self.spring_failed[self.center_springs[layer_idx, 1]] == 1
                            and self.spring_failed[self.center_springs[layer_idx, 2]] == 1
                            and self.spring_failed[self.center_springs[layer_idx, 3]] == 1
                        )
                        penetration = 0.0
                        if not is_ruptured:
                            penetration = ti.max(
                                0.0, (proj_pos.z - self.positions[i].z) * direction
                            )

                        scale_factor = 0.0
                        if self.node_initial_springs[i] > 0:
                            scale_factor = float(self.node_active_counts[i]) / float(
                                self.node_initial_springs[i]
                            )

                        f_val = k_penalty * w_normalized * penetration * scale_factor
                        f_z = f_val * direction

                        self.forces[i].z += f_z
                        ti.atomic_add(self.proj_reaction_force[None].z, -f_z)
        else:
            # 6-DOF SDF Node-to-Surface penalty contact
            proj_pos = self.proj_position[None]
            q = self.proj_quat[None]
            q_conj = ti.Vector([q[0], -q[1], -q[2], -q[3]])
            omega = self.proj_omega[None]
            v_proj = self.proj_velocity[None]

            for i in range(self.n_nodes):
                P_rel = self.positions[i] - proj_pos
                P_loc = self.ti_q_rotate(q_conj, P_rel)
                delta = -self.eval_sdf(P_loc)
                if delta > 0.0:
                    n_loc = self.eval_sdf_normal(P_loc)
                    n_world = self.ti_q_rotate(q, n_loc)

                    v_proj_point = v_proj + omega.cross(P_rel)
                    v_rel = self.velocities[i] - v_proj_point

                    delta_dot = -v_rel.dot(n_world)
                    f_mag = k_penalty * delta + self.c_damping[None] * delta_dot
                    f_mag = ti.max(0.0, f_mag)

                    scale_factor = 1.0
                    if self.node_initial_springs[i] > 0:
                        scale_factor = float(self.node_active_counts[i]) / float(
                            self.node_initial_springs[i]
                        )

                    F_contact = f_mag * n_world * scale_factor
                    self.forces[i] += F_contact
                    ti.atomic_add(self.proj_reaction_force[None], -F_contact)
                    P_contact = P_rel - delta * n_world
                    ti.atomic_add(self.proj_torque[None], P_contact.cross(-F_contact))

    @ti.func
    def apply_impedance_boundary(self, dt: ti.f32):
        """Apply dynamic boundary dashpots matching local acoustic impedance."""
        for i in range(self.n_nodes):
            if self.boundary_mask[i] == 2:
                self.node_stiffness[i] = 0.0

        for j in range(self.n_springs):
            if self.spring_failed[j] == 0:
                u, v = self.springs[j][0], self.springs[j][1]
                effective_k = self.stiffnesses[j] * (1.0 - self.spring_damage[j])
                if self.boundary_mask[u] == 2:
                    ti.atomic_add(self.node_stiffness[u], effective_k)
                if self.boundary_mask[v] == 2:
                    ti.atomic_add(self.node_stiffness[v], effective_k)

        for i in range(self.n_nodes):
            if self.boundary_mask[i] == 2:
                k_eff = ti.max(self.node_stiffness[i], 1e-4)
                C_i = ti.sqrt(self.masses[i] * k_eff)
                f_boundary = -C_i * self.velocities[i]
                self.forces[i] += f_boundary

    @ti.func
    def integrate_nodes(
        self, dt: ti.f32, rayleigh_alpha: ti.f32, v_max: ti.f32, use_viscous: ti.i32
    ):
        """Update node velocities and coordinates using leapfrog time integration."""
        for i in range(self.n_nodes):
            if self.boundary_mask[i] == 1:
                self.forces[i] = ti.Vector([0.0, 0.0, 0.0])
                self.velocities[i] = ti.Vector([0.0, 0.0, 0.0])
                continue

            # Mass-proportional Rayleigh or Legacy Viscous damping force
            damp_f = ti.Vector([0.0, 0.0, 0.0])
            if use_viscous == 1:
                damp_f = -rayleigh_alpha * self.velocities[i]
            else:
                damp_f = -rayleigh_alpha * self.masses[i] * self.velocities[i]
            p_damp = damp_f.dot(self.velocities[i])
            ti.atomic_add(self.damp_dissipated[None], -p_damp * dt)

            net_f = self.forces[i] + damp_f + self.nodal_external_forces[i]
            accel = net_f / self.masses[i]
            self.velocities[i] += accel * dt

            # CFL velocity clamping
            v_mag = self.velocities[i].norm()
            if v_mag > v_max:
                scale = v_max / v_mag
                excess_ke = 0.5 * self.masses[i] * (v_mag * v_mag - v_max * v_max)
                ti.atomic_add(self.clamp_dissipated[None], excess_ke)
                self.velocities[i] *= scale

            self.positions[i] += self.velocities[i] * dt

    @ti.func
    def integrate_projectile(self, dt: ti.f32):
        """Update rigid body kinetics representing striking projectile."""
        accel = self.proj_reaction_force[None] / self.proj_mass[None]
        accel_mag_g = accel.norm() / 9.80665
        ti.atomic_max(self.peak_deceleration_g[None], accel_mag_g)

        self.proj_velocity[None] += accel * dt
        self.proj_position[None] += self.proj_velocity[None] * dt

        if self.proj_shape_type[None] > 0:
            omega_dot = self.proj_inertia_inv[None] * self.proj_torque[None]
            self.proj_omega[None] += omega_dot * dt

            omega = self.proj_omega[None]
            q_old = self.proj_quat[None]
            q_dot = self.ti_q_mul(ti.Vector([0.0, omega.x, omega.y, omega.z]), q_old)
            q_new = q_old + 0.5 * dt * q_dot
            self.proj_quat[None] = q_new.normalized()

    @ti.func
    def evolve_failures(self, damage_onset_strain: ti.f32, failure_strain: ti.f32):
        """Evolve spring damage and rupture flags irreversibly."""
        for j in range(self.n_springs):
            if self.spring_failed[j] == 0:
                u, v = self.springs[j][0], self.springs[j][1]
                diff = self.positions[v] - self.positions[u]
                length = diff.norm()
                strain = (length - self.rest_lengths[j]) / self.rest_lengths[j]

                denom = failure_strain - damage_onset_strain
                denom_safe = denom if denom != 0.0 else 1.0
                val = (strain - damage_onset_strain) / denom_safe
                d_val = 0.0
                if val > 0.0:
                    d_val = val if val < 1.0 else 1.0

                if d_val > self.spring_damage[j]:
                    self.spring_damage[j] = d_val

                if self.spring_damage[j] >= 1.0:
                    self.spring_failed[j] = 1

    @ti.func
    def compute_failure_dissipated(
        self,
        failure_strain: ti.f32,
        damage_onset_strain: ti.f32,
        fracture_energy_multiplier: ti.f32,
    ):
        """Compute the total progressive damage failure dissipated energy continuously."""
        self.failure_dissipated[None] = 0.0
        for j in range(self.n_springs):
            k = self.stiffnesses[j]
            L0 = self.rest_lengths[j]
            x_onset = damage_onset_strain
            x_fail = failure_strain

            if self.spring_failed[j] == 1:
                w_failed = (k * L0**2 / 6.0) * (x_fail**2 + x_fail * x_onset + x_onset**2)
                ti.atomic_add(self.failure_dissipated[None], fracture_energy_multiplier * w_failed)
            else:
                D = self.spring_damage[j]
                if D > 0.0:
                    x_peak = x_onset + D * (x_fail - x_onset)
                    denom = x_fail - x_onset
                    denom_safe = denom if denom != 0.0 else 1.0
                    w_diss = (k * L0**2 / (6.0 * denom_safe)) * (x_peak**3 - x_onset**3)
                    ti.atomic_add(
                        self.failure_dissipated[None], fracture_energy_multiplier * w_diss
                    )

    @ti.func
    def compute_internal_energy(self) -> ti.f32:
        se = 0.0
        for j in range(self.n_springs):
            if self.spring_failed[j] == 0:
                u, v = self.springs[j][0], self.springs[j][1]
                diff = self.positions[v] - self.positions[u]
                length = diff.norm()
                strain = (length - self.rest_lengths[j]) / self.rest_lengths[j]
                effective_k = self.stiffnesses[j] * (1.0 - self.spring_damage[j])
                # Strain energy: 0.5 * k_eff * (strain * L0)^2
                se += 0.5 * effective_k * (strain * self.rest_lengths[j]) ** 2
        return (
            se
            + self.failure_dissipated[None]
            + self.damp_dissipated[None]
            + self.clamp_dissipated[None]
        )

    @ti.func
    def compute_artificial_kinetic_energy(self) -> ti.f32:
        e_art = 0.0
        for i in range(self.n_nodes):
            m_scaled = self.masses[i]
            m_phys = self.masses_physical[i]
            v_sq = self.velocities[i].norm_sqr()
            e_art += 0.5 * (m_scaled - m_phys) * v_sq
        return e_art

    @ti.func
    def compute_dynamic_dt_func(
        self,
        failure_strain: ti.f32,
        damage_onset_strain: ti.f32,
        w_h: ti.f32,
        t_h: ti.f32,
        k_penalty: ti.f32,
        proximity_threshold: ti.f32,
        n_nodes_per_layer: ti.i32,
        n_plies: ti.i32,
        t_ply: ti.f32,
        cfl_factor: ti.f32,
    ) -> ti.f32:
        # Clear nodal stiffnesses
        for i in range(self.n_nodes):
            self.node_stiffness[i] = 0.0

        # Sum spring stiffnesses
        for j in range(self.n_springs):
            if self.spring_failed[j] == 0:
                u, v = self.springs[j][0], self.springs[j][1]
                diff = self.positions[v] - self.positions[u]
                length = diff.norm()
                strain = (length - self.rest_lengths[j]) / self.rest_lengths[j]

                denom = failure_strain - damage_onset_strain
                denom_safe = denom if denom != 0.0 else 1.0
                val = (strain - damage_onset_strain) / denom_safe
                d_val = 0.0
                if val > 0.0:
                    d_val = val if val < 1.0 else 1.0

                damage_effective = ti.max(self.spring_damage[j], d_val)
                effective_k = self.stiffnesses[j] * (1.0 - damage_effective)

                ti.atomic_add(self.node_stiffness[u], effective_k)
                ti.atomic_add(self.node_stiffness[v], effective_k)

        # Add projectile contact stiffness
        direction = self.strike_direction[None]
        if direction == 0.0:
            direction = 1.0
            if self.proj_velocity[None].z < 0.0:
                direction = -1.0
        proj_pos = self.proj_position[None]

        # Calculate contact weights
        self.w_sum[None] = 0.0
        self.n_contacts[None] = 0
        for i in range(self.n_nodes):
            self.node_w[i] = 0.0
            px, py, pz = self.positions[i].x, self.positions[i].y, self.positions[i].z
            if (
                px >= proj_pos.x - w_h - proximity_threshold
                and px <= proj_pos.x + w_h + proximity_threshold
                and py >= proj_pos.y - t_h - proximity_threshold
                and py <= proj_pos.y + t_h + proximity_threshold
                and ti.abs(pz - proj_pos.z) <= proximity_threshold
            ):
                x_proj = ti.max(proj_pos.x - w_h, ti.min(px, proj_pos.x + w_h))
                y_proj = ti.max(proj_pos.y - t_h, ti.min(py, proj_pos.y + t_h))
                dist = ti.sqrt((px - x_proj) ** 2 + (py - y_proj) ** 2 + (pz - proj_pos.z) ** 2)

                if dist <= proximity_threshold:
                    w = 1.0 / ti.max(dist, 1e-4)
                    self.node_w[i] = w
                    ti.atomic_add(self.w_sum[None], w)
                    ti.atomic_add(self.n_contacts[None], 1)

        if self.n_contacts[None] > 0:
            w_mean = self.w_sum[None] / float(self.n_contacts[None])
            for i in range(self.n_nodes):
                if self.node_w[i] > 0.0:
                    w_normalized = self.node_w[i] / w_mean if w_mean > 0.0 else self.node_w[i]
                    scale_factor = 0.0
                    if self.node_initial_springs[i] > 0:
                        scale_factor = float(self.node_active_counts[i]) / float(
                            self.node_initial_springs[i]
                        )

                    ti.atomic_add(self.node_stiffness[i], k_penalty * w_normalized * scale_factor)

        # Add inter-ply contact stiffness
        if n_plies > 1:
            for u in range((n_plies - 1) * n_nodes_per_layer):
                v = u + n_nodes_per_layer
                delta = self.positions[u].z - self.positions[v].z + t_ply
                if delta > 0.0:
                    if self.node_active_counts[u] > 0 and self.node_active_counts[v] > 0:
                        ti.atomic_add(self.node_stiffness[u], k_penalty)
                        ti.atomic_add(self.node_stiffness[v], k_penalty)

        # Compute minimum dt
        self.dt_crit[None] = 1e10
        for i in range(self.n_nodes):
            k_i = ti.max(self.node_stiffness[i], 1e-4)
            dt_i = ti.sqrt(self.masses[i] / k_i)
            ti.atomic_min(self.dt_crit[None], dt_i)

        return cfl_factor * self.dt_crit[None]

    @ti.func
    def update_cfl_func(self):
        dt_crit = self.compute_dynamic_dt_func(
            self.failure_strain[None],
            self.damage_onset_strain[None],
            self.w_h[None],
            self.t_h[None],
            self.k_penalty[None],
            self.proximity_threshold[None],
            self.n_nodes_per_layer[None],
            self.n_plies[None],
            self.t_ply[None],
            self.cfl_factor[None],
        )
        self.dt[None] = dt_crit
        self.v_max[None] = self.dx[None] / dt_crit

    @ti.func
    def guardrail_check_func(self):
        e_art = self.compute_artificial_kinetic_energy()
        self.E_artificial_kinetic[None] = e_art
        e_int = self.compute_internal_energy()
        if e_int > 0.0:
            if e_art > 0.02 * e_int:
                self.physics_violated[None] = 1

    @ti.kernel
    def compute_dynamic_dt(
        self,
        failure_strain: ti.f32,
        damage_onset_strain: ti.f32,
        w_h: ti.f32,
        t_h: ti.f32,
        k_penalty: ti.f32,
        proximity_threshold: ti.f32,
        n_nodes_per_layer: ti.i32,
        n_plies: ti.i32,
        t_ply: ti.f32,
        cfl_factor: ti.f32,
    ) -> ti.f32:
        return self.compute_dynamic_dt_func(
            failure_strain,
            damage_onset_strain,
            w_h,
            t_h,
            k_penalty,
            proximity_threshold,
            n_nodes_per_layer,
            n_plies,
            t_ply,
            cfl_factor,
        )

    @ti.kernel
    def k_compute_active_counts(self):
        """Parallel kernel: count active springs per node."""
        self.compute_active_counts()

    @ti.kernel
    def k_evolve_failures(self, damage_onset_strain: ti.f32, failure_strain: ti.f32):
        self.evolve_failures(damage_onset_strain, failure_strain)

    @ti.kernel
    def k_reset_forces(self):
        self.reset_forces()

    @ti.kernel
    def k_compute_projectile_forces(
        self, w_h: ti.f32, t_h: ti.f32, k_penalty: ti.f32, proximity_threshold: ti.f32
    ):
        self.compute_projectile_forces(w_h, t_h, k_penalty, proximity_threshold)

    @ti.kernel
    def k_compute_interply_forces(
        self, n_nodes_per_layer: ti.i32, n_plies: ti.i32, t_ply: ti.f32, k_penalty: ti.f32
    ):
        self.compute_interply_forces(n_nodes_per_layer, n_plies, t_ply, k_penalty)

    @ti.kernel
    def k_compute_spring_forces(
        self, rayleigh_beta: ti.f32, damage_onset_strain: ti.f32, failure_strain: ti.f32, dt: ti.f32
    ):
        self.compute_spring_forces(rayleigh_beta, damage_onset_strain, failure_strain, dt)

    @ti.kernel
    def k_apply_impedance_boundary(self, dt: ti.f32):
        self.apply_impedance_boundary(dt)

    @ti.kernel
    def k_integrate_nodes(
        self, dt: ti.f32, rayleigh_alpha: ti.f32, v_max: ti.f32, use_viscous: ti.i32
    ):
        self.integrate_nodes(dt, rayleigh_alpha, v_max, use_viscous)

    @ti.kernel
    def k_integrate_projectile(self, dt: ti.f32):
        self.integrate_projectile(dt)

    @ti.kernel
    def k_compute_failure_dissipated(
        self,
        failure_strain: ti.f32,
        damage_onset_strain: ti.f32,
        fracture_energy_multiplier: ti.f32,
    ):
        self.compute_failure_dissipated(
            failure_strain, damage_onset_strain, fracture_energy_multiplier
        )

    @ti.kernel
    def k_compute_guardrail(self) -> ti.i32:
        """Compute artificial KE and internal energy, return 1 if violated."""
        self.guardrail_check_func()
        return self.physics_violated[None]

    @ti.kernel
    def k_advance_time(self, dt: ti.f32):
        self.t_sim[None] += dt

    @ti.kernel
    def k_compute_internal_energy(self) -> ti.f32:
        return self.compute_internal_energy()

    @ti.func
    def fused_spring_pass_func(
        self,
        rayleigh_beta: ti.f32,
        damage_onset_strain: ti.f32,
        failure_strain: ti.f32,
        dt: ti.f32,
        fracture_energy_multiplier: ti.f32,
    ):
        # Phase A: zero active counts
        for i in range(self.n_nodes):
            self.node_active_counts[i] = 0
        # Taichi inserts automatic sync between top-level for loops

        # Phase B: zero failure dissipated
        self.failure_dissipated[None] = 0.0

        # Phase C: fused spring traversal
        for j in range(self.n_springs):
            u, v = self.springs[j][0], self.springs[j][1]

            # --- Shared geometry (computed ONCE) ---
            diff = self.positions[v] - self.positions[u]
            length = diff.norm()
            length_safe = length if length > 0.0 else 1.0
            strain = (length - self.rest_lengths[j]) / self.rest_lengths[j]

            if self.spring_failed[j] == 1:
                # Failed spring: only dissipation accounting
                k = self.stiffnesses[j]
                L0 = self.rest_lengths[j]
                x_onset = damage_onset_strain
                x_fail = failure_strain
                w_failed = (k * L0**2 / 6.0) * (x_fail**2 + x_fail * x_onset + x_onset**2)
                ti.atomic_add(self.failure_dissipated[None], fracture_energy_multiplier * w_failed)
                continue

            # --- Active count ---
            ti.atomic_add(self.node_active_counts[u], 1)
            ti.atomic_add(self.node_active_counts[v], 1)

            # --- Damage evolution ---
            denom = failure_strain - damage_onset_strain
            denom_safe = denom if denom != 0.0 else 1.0
            val = (strain - damage_onset_strain) / denom_safe
            d_val = 0.0
            if val > 0.0:
                d_val = val if val < 1.0 else 1.0
            if d_val > self.spring_damage[j]:
                self.spring_damage[j] = d_val
            if self.spring_damage[j] >= 1.0:
                self.spring_failed[j] = 1

            # --- Force computation ---
            effective_k = self.stiffnesses[j] * (1.0 - self.spring_damage[j])
            f_mag = effective_k * strain * self.rest_lengths[j]
            if self.tension_only[j] == 1 and strain < 0.0:
                f_mag = 0.0
            f_vec = (f_mag / length_safe) * diff
            ti.atomic_add(self.forces[u], f_vec)
            ti.atomic_add(self.forces[v], -f_vec)

            # Stiffness-proportional damping
            if rayleigh_beta > 0.0:
                dv = self.velocities[v] - self.velocities[u]
                v_proj = dv.dot(diff / length_safe)
                damp_mag = rayleigh_beta * effective_k * v_proj
                damp_vec = (damp_mag / length_safe) * diff
                ti.atomic_add(self.forces[u], damp_vec)
                ti.atomic_add(self.forces[v], -damp_vec)
                ti.atomic_add(self.damp_dissipated[None], damp_mag * v_proj * dt)

            # --- Partial damage dissipation ---
            D = self.spring_damage[j]
            if D > 0.0:
                k = self.stiffnesses[j]
                L0 = self.rest_lengths[j]
                x_onset = damage_onset_strain
                x_fail = failure_strain
                x_peak = x_onset + D * (x_fail - x_onset)
                denom2 = x_fail - x_onset
                denom2_safe = denom2 if denom2 != 0.0 else 1.0
                w_diss = (k * L0**2 / (6.0 * denom2_safe)) * (x_peak**3 - x_onset**3)
                ti.atomic_add(self.failure_dissipated[None], fracture_energy_multiplier * w_diss)

    @ti.kernel
    def k_fused_spring_pass(
        self,
        rayleigh_beta: ti.f32,
        damage_onset_strain: ti.f32,
        failure_strain: ti.f32,
        dt: ti.f32,
        fracture_energy_multiplier: ti.f32,
    ):
        ti.block_local(self.positions)
        ti.block_local(self.forces)
        self.fused_spring_pass_func(
            rayleigh_beta, damage_onset_strain, failure_strain, dt, fracture_energy_multiplier
        )

    @ti.func
    def fused_node_pass_func(
        self,
        dt: ti.f32,
        rayleigh_alpha: ti.f32,
        v_max: ti.f32,
        use_viscous: ti.i32,
    ) -> ti.f32:
        e_art = 0.0
        for i in range(self.n_nodes):
            if self.boundary_mask[i] == 1:
                self.forces[i] = ti.Vector([0.0, 0.0, 0.0])
                self.velocities[i] = ti.Vector([0.0, 0.0, 0.0])
                continue

            # Mass-proportional Rayleigh or Legacy Viscous damping force
            damp_f = ti.Vector([0.0, 0.0, 0.0])
            if use_viscous == 1:
                damp_f = -rayleigh_alpha * self.velocities[i]
            else:
                damp_f = -rayleigh_alpha * self.masses[i] * self.velocities[i]
            p_damp = damp_f.dot(self.velocities[i])
            ti.atomic_add(self.damp_dissipated[None], -p_damp * dt)

            net_f = self.forces[i] + damp_f + self.nodal_external_forces[i]
            accel = net_f / self.masses[i]
            self.velocities[i] += accel * dt

            # CFL velocity clamping
            v_mag = self.velocities[i].norm()
            v_sq = self.velocities[i].norm_sqr()
            if v_mag > v_max:
                scale = v_max / v_mag
                excess_ke = 0.5 * self.masses[i] * (v_sq - v_max * v_max)
                ti.atomic_add(self.clamp_dissipated[None], excess_ke)
                self.velocities[i] *= scale
                v_sq = v_max * v_max

            self.positions[i] += self.velocities[i] * dt

            # Compute artificial KE inline
            m_scaled = self.masses[i]
            m_phys = self.masses_physical[i]
            e_art += 0.5 * (m_scaled - m_phys) * v_sq

        # Projectile integration (single-element, no parallelism needed)
        self.integrate_projectile(dt)
        self.t_sim[None] += dt
        return e_art

    @ti.kernel
    def k_fused_node_pass(
        self,
        dt: ti.f32,
        rayleigh_alpha: ti.f32,
        v_max: ti.f32,
        use_viscous: ti.i32,
    ) -> ti.f32:
        return self.fused_node_pass_func(dt, rayleigh_alpha, v_max, use_viscous)

    @ti.kernel
    def k_compute_telemetry(self):
        """Compute all telemetry scalars on GPU — avoids full array downloads."""
        ke = 0.0
        for i in range(self.n_nodes):
            ke += 0.5 * self.masses[i] * self.velocities[i].norm_sqr()
        self.telem_ke[None] = ke

        se = 0.0
        peak = 0.0
        n_failed = 0
        for j in range(self.n_springs):
            if self.spring_failed[j] == 1:
                n_failed += 1
                continue
            u, v = self.springs[j][0], self.springs[j][1]
            diff = self.positions[v] - self.positions[u]
            length = diff.norm()
            strain = (length - self.rest_lengths[j]) / self.rest_lengths[j]
            effective_k = self.stiffnesses[j] * (1.0 - self.spring_damage[j])
            se += 0.5 * effective_k * (strain * self.rest_lengths[j]) ** 2
            if strain > peak:
                ti.atomic_max(peak, strain)  # parallel-safe max
        self.telem_se[None] = se + self.contact_energy[None]
        self.telem_peak_strain[None] = peak
        self.telem_failed_count[None] = n_failed

        # Projectile KE (translational + rotational)
        rot_ke = 0.0
        if self.proj_shape_type[None] > 0:
            omega = self.proj_omega[None]
            inertia_inv = self.proj_inertia_inv[None]
            rot_ke = 0.5 * (
                (1.0 / ti.max(inertia_inv[0], 1e-15)) * omega[0] ** 2
                + (1.0 / ti.max(inertia_inv[1], 1e-15)) * omega[1] ** 2
                + (1.0 / ti.max(inertia_inv[2], 1e-15)) * omega[2] ** 2
            )
        self.telem_proj_ke[None] = (
            0.5 * self.proj_mass[None] * self.proj_velocity[None].norm_sqr() + rot_ke
        )

    def get_telemetry(self) -> dict:
        """Read GPU-computed telemetry scalars (a few float transfers, no arrays)."""
        self.k_compute_telemetry()
        return {
            "ke": float(self.telem_ke[None]),
            "se": float(self.telem_se[None]),
            "peak_strain": float(self.telem_peak_strain[None]),
            "proj_ke": float(self.telem_proj_ke[None]),
            "failed_count": int(self.telem_failed_count[None]),
            "peak_deceleration_g": float(self.peak_deceleration_g[None]),
        }

    def build_simulation_graph(self, num_substeps, use_cfl, use_interply, cfl_recompute_interval):
        builder = ti.graph.GraphBuilder()
        for step in range(num_substeps):
            # Step 1: half step velocity and position update
            builder.dispatch(self.k_integrate_nodes_half_step_graph)

            if use_cfl and step % cfl_recompute_interval == 0:
                builder.dispatch(self.k_compute_active_counts_graph)
                builder.dispatch(self.k_update_cfl_graph)

            # Step 2: compute forces at updated positions/velocities
            builder.dispatch(self.k_fused_spring_pass_graph)
            builder.dispatch(self.k_compute_projectile_forces_graph)
            if use_interply:
                builder.dispatch(self.k_compute_interply_forces_graph)
            builder.dispatch(self.k_apply_impedance_boundary_graph)

            # Step 3: full step velocity update
            builder.dispatch(self.k_integrate_nodes_full_step_graph)

            # Amortized guardrail check (only every 20 steps, and at the end of the chunk)
            if step % cfl_recompute_interval == 0 or step == num_substeps - 1:
                builder.dispatch(self.k_guardrail_check_graph)

        return builder.compile()

    def get_or_compile_graph(self, num_substeps, use_cfl, use_interply, cfl_recompute_interval):
        key = (num_substeps, use_cfl, use_interply, cfl_recompute_interval)
        if key not in self.compiled_graphs:
            self.compiled_graphs[key] = self.build_simulation_graph(
                num_substeps, use_cfl, use_interply, cfl_recompute_interval
            )
        return self.compiled_graphs[key]

    def run_substeps(
        self,
        num_substeps: int,
        dt_init: float,
        rayleigh_beta: float,
        damage_onset_strain: float,
        failure_strain: float,
        n_plies: int,
        n_nodes_per_layer: int,
        t_ply: float,
        k_penalty: float,
        w_h: float,
        t_h: float,
        proximity_threshold: float,
        rayleigh_alpha: float,
        use_viscous: int,
        cfl_factor: float,
        dx: float,
        fracture_energy_multiplier: float,
        cfl_recompute_interval: int = 20,
    ) -> float:
        """Host-side substep loop with parallel GPU kernels compiled into a static graph."""
        if self.physics_violated[None] == 1:
            return dt_init

        # Update scalar parameters on GPU
        self.dt[None] = dt_init
        if cfl_factor > 0.0:
            self.v_max[None] = dx / dt_init
        else:
            self.v_max[None] = 1e20
        self.rayleigh_beta[None] = rayleigh_beta
        self.damage_onset_strain[None] = damage_onset_strain
        self.failure_strain[None] = failure_strain
        self.n_plies[None] = n_plies
        self.n_nodes_per_layer[None] = n_nodes_per_layer
        self.t_ply[None] = t_ply
        self.k_penalty[None] = k_penalty
        self.w_h[None] = w_h
        self.t_h[None] = t_h
        self.proximity_threshold[None] = proximity_threshold
        self.rayleigh_alpha[None] = rayleigh_alpha
        self.use_viscous[None] = use_viscous
        self.cfl_factor[None] = cfl_factor
        self.dx[None] = dx
        self.fracture_energy_multiplier[None] = fracture_energy_multiplier

        # Compile or retrieve graph
        use_cfl = cfl_factor > 0.0
        use_interply = n_plies > 1

        if self.t_sim[None] == 0.0:
            self.k_compute_active_counts_graph()
            self.k_fused_spring_pass_graph()
            self.k_compute_projectile_forces_graph()
            if use_interply:
                self.k_compute_interply_forces_graph()
            self.k_apply_impedance_boundary_graph()
            self.k_compute_initial_accelerations_graph()

        graph = self.get_or_compile_graph(
            num_substeps, use_cfl, use_interply, cfl_recompute_interval
        )

        # Run graph in exactly 1 call
        graph.run({})

        return float(self.dt[None])

    @ti.kernel
    def advance_substeps(
        self,
        num_substeps: ti.i32,
        dt_init: ti.f32,
        rayleigh_beta: ti.f32,
        damage_onset_strain: ti.f32,
        failure_strain: ti.f32,
        n_plies: ti.i32,
        n_nodes_per_layer: ti.i32,
        t_ply: ti.f32,
        k_penalty: ti.f32,
        w_h: ti.f32,
        t_h: ti.f32,
        proximity_threshold: ti.f32,
        rayleigh_alpha: ti.f32,
        use_viscous: ti.i32,
        cfl_factor: ti.f32,
        dx: ti.f32,
        fracture_energy_multiplier: ti.f32,
    ) -> ti.f32:
        ti.loop_config(serialize=True)
        dt = dt_init
        for _ in range(num_substeps):
            if self.physics_violated[None] == 1:
                break

            # 0. Compute active counts (needed for contact scale factor and stiffness)
            self.compute_active_counts()

            # 1. Compute dynamic dt and v_max
            v_max = 1e20
            if cfl_factor > 0.0:
                dt = self.compute_dynamic_dt_func(
                    failure_strain,
                    damage_onset_strain,
                    w_h,
                    t_h,
                    k_penalty,
                    proximity_threshold,
                    n_nodes_per_layer,
                    n_plies,
                    t_ply,
                    cfl_factor,
                )
                v_max = dx / dt

            # 2. Evolve Failures
            self.evolve_failures(damage_onset_strain, failure_strain)

            self.reset_forces()

            # 3. Projectile Contact
            self.compute_projectile_forces(w_h, t_h, k_penalty, proximity_threshold)

            # 4. Inter-ply Contact
            if n_plies > 1:
                self.compute_interply_forces(n_nodes_per_layer, n_plies, t_ply, k_penalty)

            # 5. Internal Springs & stiffness damping
            beta_val = 0.0
            if use_viscous == 0:
                beta_val = rayleigh_beta
            self.compute_spring_forces(beta_val, damage_onset_strain, failure_strain, dt)

            # 5.5 Dynamic acoustic impedance boundary damping
            self.apply_impedance_boundary(dt)

            # 6. Integrate (nodes and projectile)
            self.integrate_nodes(dt, rayleigh_alpha, v_max, use_viscous)
            self.integrate_projectile(dt)

            self.t_sim[None] += dt

            # 7. Compute progressive damage dissipated energy continuously
            self.compute_failure_dissipated(
                failure_strain, damage_onset_strain, fracture_energy_multiplier
            )

            # 8. Mass-scaling safety guardrail
            e_art = self.compute_artificial_kinetic_energy()
            self.E_artificial_kinetic[None] = e_art
            e_int = self.compute_internal_energy()
            if e_int > 0.0:
                if e_art > 0.02 * e_int:
                    self.physics_violated[None] = 1

        return dt


_SOLVER_CACHE = None


def taichi_leapfrog_loop(
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
    node_spring_offsets: np.ndarray | None = None,
    node_spring_ids: np.ndarray | None = None,
    node_spring_signs: np.ndarray | None = None,
    use_viscous: bool = False,
    cfl_factor: float = -1.0,
    grid_damage: np.ndarray | None = None,
    grid_masses_physical: np.ndarray | None = None,
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
    hist_proj_quat: np.ndarray | None = None,
    contact_energy_init: float = 0.0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    """Execute a block chunk of explicit time steps entirely in GPU compilation."""
    if ti is None:
        raise ImportError("Taichi compiler package not available.")

    global _SOLVER_CACHE
    n_nodes = len(positions)
    n_springs = len(grid_springs)

    base_nodes = positions[:n_nodes_per_layer]
    dists_in_plane = (base_nodes[:, 0] - proj_position[0]) ** 2 + (
        base_nodes[:, 1] - proj_position[1]
    ) ** 2
    center_idx = int(np.argmin(dists_in_plane))

    # Build node_spring_offsets and node_spring_ids if they are None
    offsets = node_spring_offsets
    ids = node_spring_ids
    if offsets is None or ids is None:
        node_counts = np.zeros(n_nodes, dtype=np.int32)
        np.add.at(node_counts, grid_springs[:, 0], 1)
        np.add.at(node_counts, grid_springs[:, 1], 1)
        offsets = np.zeros(n_nodes + 1, dtype=np.int32)
        offsets[1:] = np.cumsum(node_counts)
        current_offset = offsets[:-1].copy()
        ids = np.zeros(2 * n_springs, dtype=np.int32)
        for j in range(n_springs):
            n0 = grid_springs[j, 0]
            n1 = grid_springs[j, 1]
            ids[current_offset[n0]] = j
            current_offset[n0] += 1
            ids[current_offset[n1]] = j
            current_offset[n1] += 1

    center_springs_list = []
    for ply in range(n_plies):
        c = center_idx + ply * n_nodes_per_layer
        start = offsets[c]
        end = offsets[c + 1]
        layer_c_springs = []
        for idx in range(start, end):
            sp_id = ids[idx]
            if grid_tension_only[sp_id]:
                layer_c_springs.append(sp_id)
        while len(layer_c_springs) < 4:
            layer_c_springs.append(layer_c_springs[0] if len(layer_c_springs) > 0 else 0)
        center_springs_list.append(layer_c_springs[:4])
    center_springs_arr = np.array(center_springs_list, dtype=np.int32)

    if (
        _SOLVER_CACHE is None
        or _SOLVER_CACHE.n_nodes != n_nodes
        or _SOLVER_CACHE.n_springs != n_springs
        or getattr(_SOLVER_CACHE, "n_plies_val", None) != n_plies
        or getattr(_SOLVER_CACHE, "n_nodes_per_layer_val", None) != n_nodes_per_layer
        or getattr(_SOLVER_CACHE, "_stiffnesses_id", None) != id(grid_stiffnesses)
    ):
        _SOLVER_CACHE = TaichiSolver(
            n_nodes=n_nodes,
            n_springs=n_springs,
            positions_init=positions,
            velocities_init=velocities,
            springs_init=grid_springs,
            stiffnesses_init=grid_stiffnesses,
            rest_lengths_init=grid_rest_lengths,
            failed_init=grid_failed,
            masses_init=grid_masses,
            tension_only_init=grid_tension_only,
            boundary_mask_init=boundary_mask,
            nodal_external_forces_init=nodal_external_forces,
            proj_position_init=proj_position,
            proj_velocity_init=proj_velocity,
            proj_mass_init=proj_mass,
            strike_direction_init=strike_direction,
            node_initial_springs_init=node_initial_springs,
            masses_physical_init=grid_masses_physical,
            n_plies=n_plies,
            n_nodes_per_layer=n_nodes_per_layer,
            center_idx_init=center_idx,
            center_springs_init=center_springs_arr,
        )
        _SOLVER_CACHE._stiffnesses_id = id(grid_stiffnesses)
        if grid_damage is not None:
            _SOLVER_CACHE.spring_damage.from_numpy(grid_damage.astype(np.float32))
    else:
        # Re-use existing allocated GPU fields, updating only dynamic variables in-place
        _SOLVER_CACHE.positions.from_numpy(positions.astype(np.float32))
        _SOLVER_CACHE.velocities.from_numpy(velocities.astype(np.float32))
        _SOLVER_CACHE.spring_failed.from_numpy(grid_failed.astype(np.int32))
        if grid_damage is not None:
            _SOLVER_CACHE.spring_damage.from_numpy(grid_damage.astype(np.float32))
        elif t_sim_init == 0.0:
            _SOLVER_CACHE.spring_damage.from_numpy(grid_failed.astype(np.float32))
        _SOLVER_CACHE.proj_position[None] = proj_position.astype(np.float32)
        _SOLVER_CACHE.proj_velocity[None] = proj_velocity.astype(np.float32)
        _SOLVER_CACHE.proj_mass[None] = float(proj_mass)
        _SOLVER_CACHE.strike_direction[None] = float(strike_direction)

        if t_sim_init == 0.0:
            _SOLVER_CACHE.masses.from_numpy(grid_masses.astype(np.float32))
            if grid_masses_physical is not None:
                _SOLVER_CACHE.masses_physical.from_numpy(grid_masses_physical.astype(np.float32))
            else:
                _SOLVER_CACHE.masses_physical.from_numpy(grid_masses.astype(np.float32))
            _SOLVER_CACHE.boundary_mask.from_numpy(boundary_mask.astype(np.int32))
            _SOLVER_CACHE.nodal_external_forces.from_numpy(nodal_external_forces.astype(np.float32))
            _SOLVER_CACHE.node_initial_springs.from_numpy(node_initial_springs.astype(np.int32))
            _SOLVER_CACHE.springs.from_numpy(grid_springs.astype(np.int32))
            _SOLVER_CACHE.stiffnesses.from_numpy(grid_stiffnesses.astype(np.float32))
            _SOLVER_CACHE.rest_lengths.from_numpy(grid_rest_lengths.astype(np.float32))
            _SOLVER_CACHE.tension_only.from_numpy(grid_tension_only.astype(np.int32))
            _SOLVER_CACHE.physics_violated[None] = 0

    solver = _SOLVER_CACHE
    shape_map = {"box": 0, "sphere": 1, "cylinder": 2, "bullet": 3, "propeller": 4}
    shape_code = shape_map.get(proj_shape_type.lower(), 0)
    solver.proj_shape_type[None] = shape_code

    if proj_quat is not None:
        solver.proj_quat[None] = proj_quat.astype(np.float32)
    if proj_omega is not None:
        solver.proj_omega[None] = proj_omega.astype(np.float32)
    if proj_inertia_inv is not None:
        diag = np.diagonal(proj_inertia_inv) if proj_inertia_inv.ndim == 2 else proj_inertia_inv
        solver.proj_inertia_inv[None] = diag.astype(np.float32)
    solver.radius_field[None] = float(proj_radius)
    solver.length_field[None] = float(proj_length)
    solver.edge_radius_field[None] = float(proj_edge_radius)
    solver.ogive_multiplier_field[None] = float(proj_ogive_multiplier)
    solver.span_field[None] = float(proj_span)
    solver.root_chord_field[None] = float(proj_root_chord)
    solver.tip_chord_field[None] = float(proj_tip_chord)
    solver.twist_field[None] = float(proj_twist)
    solver.thickness_ratio_field[None] = float(proj_thickness_ratio)
    solver.tip_radius_field[None] = float(proj_tip_radius)
    solver.z_com_field[None] = float(proj_z_com)
    solver.y_com_field[None] = float(proj_y_com)
    solver.c_damping[None] = float(proj_c_damping)

    if t_sim_init == 0.0:
        solver.peak_deceleration_g[None] = 0.0
    solver.damp_dissipated[None] = damp_dissipated_init
    solver.failure_dissipated[None] = failure_dissipated_init
    solver.clamp_dissipated[None] = clamp_dissipated_init
    solver.contact_energy[None] = contact_energy_init
    solver.t_sim[None] = float(t_sim_init)

    # Pre-allocate dynamic history traces
    m_frames = n_steps // save_interval
    n_nodes = len(positions)
    n_springs = len(grid_springs)

    hist_positions = np.zeros((m_frames, n_nodes, 3), dtype=positions.dtype)
    hist_failed = np.zeros((m_frames, n_springs), dtype=np.bool_)
    hist_proj_pos = np.zeros((m_frames, 3), dtype=positions.dtype)
    hist_time = np.zeros(m_frames, dtype=positions.dtype)
    hist_ke = np.zeros(m_frames, dtype=positions.dtype)
    hist_se = np.zeros(m_frames, dtype=positions.dtype)
    hist_proj_ke = np.zeros(m_frames, dtype=positions.dtype)
    hist_peak_strain = np.zeros(m_frames, dtype=positions.dtype)

    w_h = proj_blade_width / 2.0
    t_h = proj_edge_thickness / 2.0
    proximity_threshold = dx * 2.0
    use_visc_val = 1 if use_viscous else 0

    # Running explicit integration loop in chunks of save_interval
    n_chunks = n_steps // save_interval
    for chunk in range(n_chunks):
        # Run save_interval steps on GPU
        dt = solver.run_substeps(
            save_interval,
            dt,
            rayleigh_beta,
            damage_onset_strain,
            failure_strain,
            n_plies,
            n_nodes_per_layer,
            t_ply,
            k_penalty,
            w_h,
            t_h,
            proximity_threshold,
            rayleigh_alpha,
            use_visc_val,
            cfl_factor,
            dx,
            fracture_energy_multiplier,
        )
        if solver.physics_violated[None] == 1:
            raise PhysicsViolationError(
                "Physics violation: Artificial kinetic energy from mass scaling "
                "exceeded 2% of total internal energy."
            )

        t_sim = float(solver.t_sim[None])

        # Fetch telemetry scalars from GPU (no full array downloads!)
        telem = solver.get_telemetry()
        hist_ke[chunk] = telem["ke"]
        hist_se[chunk] = telem["se"]
        hist_proj_ke[chunk] = telem["proj_ke"]
        hist_peak_strain[chunk] = telem["peak_strain"]
        hist_time[chunk] = t_sim

        # Fetch animation arrays from GPU to Host CPU
        pos_cpu = solver.positions.to_numpy().astype(positions.dtype)
        failed_cpu = solver.spring_failed.to_numpy() == 1
        proj_pos_cpu = solver.proj_position.to_numpy().astype(positions.dtype)

        hist_positions[chunk] = pos_cpu
        hist_failed[chunk] = failed_cpu
        hist_proj_pos[chunk] = proj_pos_cpu
        if hist_proj_quat is not None:
            hist_proj_quat[chunk] = solver.proj_quat.to_numpy()

    # Run any remaining steps
    remainder = n_steps % save_interval
    if remainder > 0:
        dt = solver.run_substeps(
            remainder,
            dt,
            rayleigh_beta,
            damage_onset_strain,
            failure_strain,
            n_plies,
            n_nodes_per_layer,
            t_ply,
            k_penalty,
            w_h,
            t_h,
            proximity_threshold,
            rayleigh_alpha,
            use_visc_val,
            cfl_factor,
            dx,
            fracture_energy_multiplier,
        )
        if solver.physics_violated[None] == 1:
            raise PhysicsViolationError(
                "Physics violation: Artificial kinetic energy from mass scaling "
                "exceeded 2% of total internal energy."
            )

    # Retrieve final simulation output vectors
    final_pos = solver.positions.to_numpy().astype(positions.dtype)
    final_vel = solver.velocities.to_numpy().astype(velocities.dtype)
    final_failed = solver.spring_failed.to_numpy() == 1
    final_proj_pos = solver.proj_position.to_numpy().astype(proj_position.dtype)
    final_proj_vel = solver.proj_velocity.to_numpy().astype(proj_velocity.dtype)
    final_damp = float(solver.damp_dissipated[None])
    final_failure = float(solver.failure_dissipated[None])
    final_clamp = float(solver.clamp_dissipated[None])
    t_sim = float(solver.t_sim[None])
    if grid_damage is not None:
        grid_damage[:] = solver.spring_damage.to_numpy()

    if proj_quat is not None:
        proj_quat[:] = solver.proj_quat.to_numpy().astype(proj_quat.dtype)
    if proj_omega is not None:
        proj_omega[:] = solver.proj_omega.to_numpy().astype(proj_omega.dtype)

    return (
        final_pos,
        final_vel,
        final_failed,
        final_proj_pos,
        final_proj_vel,
        final_damp,
        final_failure,
        final_clamp,
        t_sim,
        hist_positions,
        hist_failed,
        hist_proj_pos,
        hist_time,
        hist_ke,
        hist_se,
        hist_proj_ke,
        hist_peak_strain,
        float(solver.contact_energy[None]),
    )
