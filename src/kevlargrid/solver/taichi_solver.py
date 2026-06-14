"""Taichi Lang GPU Mass-Spring Solver compute kernels.

Implements Verlet explicit time integration, spring forces, interply contact,
and projectile contact on macOS Metal or Windows CUDA/Vulkan GPUs.
"""

import numpy as np

try:
    import taichi as ti
except ImportError:
    ti = None

from kevlargrid.solver.energy import compute_kinetic_energy, compute_strain_energy

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
    ) -> None:
        self.n_nodes = n_nodes
        self.n_springs = n_springs

        real_type = ti.f32

        # Node fields
        self.positions = ti.Vector.field(3, dtype=real_type, shape=n_nodes)
        self.velocities = ti.Vector.field(3, dtype=real_type, shape=n_nodes)
        self.forces = ti.Vector.field(3, dtype=real_type, shape=n_nodes)
        self.nodal_external_forces = ti.Vector.field(3, dtype=real_type, shape=n_nodes)
        self.masses = ti.field(dtype=real_type, shape=n_nodes)
        self.boundary_mask = ti.field(dtype=ti.i32, shape=n_nodes)
        self.node_initial_springs = ti.field(dtype=ti.i32, shape=n_nodes)
        self.node_active_counts = ti.field(dtype=ti.i32, shape=n_nodes)
        self.node_stiffness = ti.field(dtype=real_type, shape=n_nodes)  # Added S7.14
        self.dt_crit = ti.field(dtype=real_type, shape=())             # Added S7.14

        # Spring fields
        self.springs = ti.Vector.field(2, dtype=ti.i32, shape=n_springs)
        self.stiffnesses = ti.field(dtype=real_type, shape=n_springs)
        self.rest_lengths = ti.field(dtype=real_type, shape=n_springs)
        self.spring_failed = ti.field(dtype=ti.i32, shape=n_springs)
        self.tension_only = ti.field(dtype=ti.i32, shape=n_springs)

        # Projectile fields
        self.proj_position = ti.Vector.field(3, dtype=real_type, shape=())
        self.proj_velocity = ti.Vector.field(3, dtype=real_type, shape=())
        self.proj_reaction_force = ti.Vector.field(3, dtype=real_type, shape=())
        self.proj_mass = ti.field(dtype=real_type, shape=())
        self.strike_direction = ti.field(dtype=real_type, shape=())

        # Dissipated energy fields
        self.damp_dissipated = ti.field(dtype=real_type, shape=())
        self.failure_dissipated = ti.field(dtype=real_type, shape=())
        self.clamp_dissipated = ti.field(dtype=real_type, shape=())

        # Helper reduction fields for projectile contact
        self.node_w = ti.field(dtype=real_type, shape=n_nodes)
        self.w_sum = ti.field(dtype=real_type, shape=())
        self.n_contacts = ti.field(dtype=ti.i32, shape=())

        # Initialize GPU buffers from numpy inputs
        self.positions.from_numpy(positions_init.astype(np.float32))
        self.velocities.from_numpy(velocities_init.astype(np.float32))
        self.masses.from_numpy(masses_init.astype(np.float32))
        self.boundary_mask.from_numpy(boundary_mask_init.astype(np.int32))
        self.nodal_external_forces.from_numpy(nodal_external_forces_init.astype(np.float32))
        self.node_initial_springs.from_numpy(node_initial_springs_init.astype(np.int32))

        self.springs.from_numpy(springs_init.astype(np.int32))
        self.stiffnesses.from_numpy(stiffnesses_init.astype(np.float32))
        self.rest_lengths.from_numpy(rest_lengths_init.astype(np.float32))
        self.spring_failed.from_numpy(failed_init.astype(np.int32))
        self.tension_only.from_numpy(tension_only_init.astype(np.int32))

        self.proj_position[None] = proj_position_init.astype(np.float32)
        self.proj_velocity[None] = proj_velocity_init.astype(np.float32)
        self.proj_mass[None] = float(proj_mass_init)
        self.strike_direction[None] = float(strike_direction_init)

        self.proj_reaction_force[None] = [0.0, 0.0, 0.0]
        self.damp_dissipated[None] = 0.0
        self.failure_dissipated[None] = 0.0
        self.clamp_dissipated[None] = 0.0

    @ti.kernel
    def reset_forces(self):
        """Clear dynamic nodal forces and projectile reaction forces."""
        for i in range(self.n_nodes):
            self.forces[i] = ti.Vector([0.0, 0.0, 0.0])
        self.proj_reaction_force[None] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
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

            # Progressive damage model
            denom = failure_strain - damage_onset_strain
            denom_safe = denom if denom != 0.0 else 1.0
            val = (strain - damage_onset_strain) / denom_safe
            damage = 0.0
            if val > 0.0:
                damage = val if val < 1.0 else 1.0
            effective_k = self.stiffnesses[j] * (1.0 - damage)

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
                damp_mag = rayleigh_beta * self.stiffnesses[j] * v_proj
                damp_vec = (damp_mag / length_safe) * diff
                ti.atomic_add(self.forces[u], damp_vec)
                ti.atomic_add(self.forces[v], -damp_vec)
                ti.atomic_add(self.damp_dissipated[None], damp_mag * v_proj * dt)

    @ti.kernel
    def compute_interply_forces(
        self, n_nodes_per_layer: ti.i32, n_plies: ti.i32, t_ply: ti.f32, k_penalty: ti.f32
    ):
        """Compute contact force penalty values preventing interply penetration."""
        for u in range((n_plies - 1) * n_nodes_per_layer):
            v = u + n_nodes_per_layer
            delta = self.positions[u].z - self.positions[v].z + t_ply
            if delta > 0.0:
                f_mag = k_penalty * delta
                if self.node_active_counts[u] > 0 and self.node_active_counts[v] > 0:
                    ti.atomic_add(self.forces[u].z, -f_mag)
                    ti.atomic_add(self.forces[v].z, f_mag)

    @ti.kernel
    def compute_active_counts(self):
        """Compute the count of active (non-failed) springs connected to each node."""
        for i in range(self.n_nodes):
            self.node_active_counts[i] = 0
        for j in range(self.n_springs):
            if self.spring_failed[j] == 0:
                u, v = self.springs[j][0], self.springs[j][1]
                ti.atomic_add(self.node_active_counts[u], 1)
                ti.atomic_add(self.node_active_counts[v], 1)

    @ti.kernel
    def compute_projectile_forces(
        self, w_h: ti.f32, t_h: ti.f32, k_penalty: ti.f32, proximity_threshold: ti.f32
    ):
        """Compute blade-to-mesh contact interface force distribution."""
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
            x_proj = ti.max(proj_pos.x - w_h, ti.min(self.positions[i].x, proj_pos.x + w_h))
            y_proj = ti.max(proj_pos.y - t_h, ti.min(self.positions[i].y, proj_pos.y + t_h))
            dist = ti.sqrt(
                (self.positions[i].x - x_proj) ** 2
                + (self.positions[i].y - y_proj) ** 2
                + (self.positions[i].z - proj_pos.z) ** 2
            )

            if dist <= proximity_threshold:
                w = 1.0 / ti.max(dist, 1e-4)
                self.node_w[i] = w
                ti.atomic_add(self.w_sum[None], w)
                ti.atomic_add(self.n_contacts[None], 1)
            else:
                self.node_w[i] = 0.0

        # Phase 2: Distribute force based on IDW weights
        if self.n_contacts[None] > 0:
            w_mean = self.w_sum[None] / float(self.n_contacts[None])
            for i in range(self.n_nodes):
                if self.node_w[i] > 0.0:
                    w_normalized = self.node_w[i] / w_mean if w_mean > 0.0 else self.node_w[i]
                    penetration = ti.max(0.0, (proj_pos.z - self.positions[i].z) * direction)
                    
                    scale_factor = 0.0
                    if self.node_initial_springs[i] > 0:
                        scale_factor = float(self.node_active_counts[i]) / float(self.node_initial_springs[i])
                    
                    f_val = k_penalty * w_normalized * penetration * scale_factor
                    f_z = f_val * direction

                    self.forces[i].z += f_z
                    ti.atomic_add(self.proj_reaction_force[None].z, -f_z)

    @ti.kernel
    def integrate_nodes(self, dt: ti.f32, rayleigh_alpha: ti.f32, v_max: ti.f32, use_viscous: ti.i32):
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

    @ti.kernel
    def integrate_projectile(self, dt: ti.f32):
        """Update rigid body kinetics representing striking projectile."""
        proj_accel = self.proj_reaction_force[None] / self.proj_mass[None]
        self.proj_velocity[None] += proj_accel * dt
        self.proj_position[None] += self.proj_velocity[None] * dt

    @ti.kernel
    def evolve_failures(self, failure_strain: ti.f32):
        """Evolve spring rupture flags irreversibly."""
        for j in range(self.n_springs):
            if self.spring_failed[j] == 0:
                u, v = self.springs[j][0], self.springs[j][1]
                diff = self.positions[v] - self.positions[u]
                length = diff.norm()
                strain = (length - self.rest_lengths[j]) / self.rest_lengths[j]
                if strain > failure_strain:
                    self.spring_failed[j] = 1

    @ti.kernel
    def compute_failure_dissipated(
        self, failure_strain: ti.f32, damage_onset_strain: ti.f32, fracture_energy_multiplier: ti.f32
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
                u, v = self.springs[j][0], self.springs[j][1]
                diff = self.positions[v] - self.positions[u]
                length = diff.norm()
                x = (length - L0) / L0

                if x >= x_onset:
                    denom = x_fail - x_onset
                    denom_safe = denom if denom != 0.0 else 1.0
                    damage = (x - x_onset) / denom_safe
                    if damage > 1.0:
                        damage = 1.0
                    effective_k = k * (1.0 - damage)

                    w_input = (k * L0**2 / 6.0) * (x**2 + x * x_onset + x_onset**2)
                    se_actual = 0.5 * effective_k * (x * L0)**2
                    ti.atomic_add(self.failure_dissipated[None], fracture_energy_multiplier * (w_input - se_actual))

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
                damage = 0.0
                if val > 0.0:
                    damage = val if val < 1.0 else 1.0
                effective_k = self.stiffnesses[j] * (1.0 - damage)

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
            x_proj = ti.max(proj_pos.x - w_h, ti.min(self.positions[i].x, proj_pos.x + w_h))
            y_proj = ti.max(proj_pos.y - t_h, ti.min(self.positions[i].y, proj_pos.y + t_h))
            dist = ti.sqrt(
                (self.positions[i].x - x_proj) ** 2
                + (self.positions[i].y - y_proj) ** 2
                + (self.positions[i].z - proj_pos.z) ** 2
            )

            if dist <= proximity_threshold:
                w = 1.0 / ti.max(dist, 1e-4)
                self.node_w[i] = w
                ti.atomic_add(self.w_sum[None], w)
                ti.atomic_add(self.n_contacts[None], 1)
            else:
                self.node_w[i] = 0.0

        if self.n_contacts[None] > 0:
            w_mean = self.w_sum[None] / float(self.n_contacts[None])
            for i in range(self.n_nodes):
                if self.node_w[i] > 0.0:
                    w_normalized = self.node_w[i] / w_mean if w_mean > 0.0 else self.node_w[i]
                    scale_factor = 0.0
                    if self.node_initial_springs[i] > 0:
                        scale_factor = float(self.node_active_counts[i]) / float(self.node_initial_springs[i])

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
]:
    """Execute a block chunk of explicit time steps entirely in GPU compilation."""
    if ti is None:
        raise ImportError("Taichi compiler package not available.")

    global _SOLVER_CACHE
    n_nodes = len(positions)
    n_springs = len(grid_springs)

    if (
        _SOLVER_CACHE is None
        or _SOLVER_CACHE.n_nodes != n_nodes
        or _SOLVER_CACHE.n_springs != n_springs
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
        )
    else:
        # Re-use existing allocated GPU fields, updating dynamic variables in-place
        _SOLVER_CACHE.positions.from_numpy(positions.astype(np.float32))
        _SOLVER_CACHE.velocities.from_numpy(velocities.astype(np.float32))
        _SOLVER_CACHE.spring_failed.from_numpy(grid_failed.astype(np.int32))
        _SOLVER_CACHE.nodal_external_forces.from_numpy(nodal_external_forces.astype(np.float32))
        _SOLVER_CACHE.node_initial_springs.from_numpy(node_initial_springs.astype(np.int32))
        _SOLVER_CACHE.proj_position[None] = proj_position.astype(np.float32)
        _SOLVER_CACHE.proj_velocity[None] = proj_velocity.astype(np.float32)
        _SOLVER_CACHE.proj_mass[None] = float(proj_mass)
        _SOLVER_CACHE.strike_direction[None] = float(strike_direction)

    solver = _SOLVER_CACHE
    solver.damp_dissipated[None] = damp_dissipated_init
    solver.failure_dissipated[None] = failure_dissipated_init
    solver.clamp_dissipated[None] = clamp_dissipated_init

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

    w_h = proj_blade_width / 2.0
    t_h = proj_edge_thickness / 2.0
    proximity_threshold = dx * 2.0
    t_sim = t_sim_init

    # Running explicit integration loop
    for step in range(n_steps):
        # 0. Compute active counts (needed for contact scale factor and stiffness)
        solver.compute_active_counts()

        # 1. Compute dynamic dt and v_max
        v_max = dx / dt
        if cfl_factor > 0.0:
            dt = solver.compute_dynamic_dt(
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
        solver.evolve_failures(failure_strain)

        solver.reset_forces()

        # 3. Projectile Contact
        solver.compute_projectile_forces(w_h, t_h, k_penalty, proximity_threshold)

        # 4. Inter-ply Contact
        if n_plies > 1:
            solver.compute_interply_forces(n_nodes_per_layer, n_plies, t_ply, k_penalty)

        # 5. Internal Springs & stiffness damping
        beta_val = 0.0 if use_viscous else rayleigh_beta
        solver.compute_spring_forces(beta_val, damage_onset_strain, failure_strain, dt)

        # 6. Integrate (nodes and projectile)
        use_visc = 1 if use_viscous else 0
        solver.integrate_nodes(dt, rayleigh_alpha, v_max, use_visc)
        solver.integrate_projectile(dt)

        t_sim += dt

        # 7. Compute progressive damage dissipated energy continuously
        solver.compute_failure_dissipated(failure_strain, damage_onset_strain, fracture_energy_multiplier)

        # Save frame at telemetry interval
        step_1indexed = step + 1
        if step_1indexed % save_interval == 0:
            frame_idx = (step_1indexed // save_interval) - 1

            # Fetch states from GPU to Host CPU memory
            pos_cpu = solver.positions.to_numpy().astype(positions.dtype)
            vel_cpu = solver.velocities.to_numpy().astype(positions.dtype)
            failed_cpu = solver.spring_failed.to_numpy() == 1
            proj_pos_cpu = solver.proj_position.to_numpy().astype(positions.dtype)
            proj_vel_cpu = solver.proj_velocity.to_numpy().astype(positions.dtype)

            # Compute strains and energies on Host CPU (zero PCIE overhead)
            p1_cpu = pos_cpu[grid_springs[:, 0]]
            p2_cpu = pos_cpu[grid_springs[:, 1]]
            lengths_cpu = np.sqrt(np.sum((p2_cpu - p1_cpu) ** 2, axis=1))
            strains_cpu = (lengths_cpu - grid_rest_lengths) / grid_rest_lengths

            ke = compute_kinetic_energy(vel_cpu, grid_masses)
            
            # Use degraded strain energy S7.14
            denom = failure_strain - damage_onset_strain
            denom_safe = denom if denom != 0.0 else 1.0
            damage_telem = np.minimum(np.maximum((strains_cpu - damage_onset_strain) / denom_safe, 0.0), 1.0)
            se = compute_strain_energy(strains_cpu, grid_stiffnesses, grid_rest_lengths, failed_cpu, damage_telem)
            
            proj_ke = 0.5 * proj_mass * np.sum(proj_vel_cpu**2)

            hist_positions[frame_idx] = pos_cpu
            hist_failed[frame_idx] = failed_cpu
            hist_proj_pos[frame_idx] = proj_pos_cpu
            hist_time[frame_idx] = t_sim
            hist_ke[frame_idx] = ke
            hist_se[frame_idx] = se
            hist_proj_ke[frame_idx] = proj_ke

    # Retrieve final simulation output vectors
    final_pos = solver.positions.to_numpy().astype(positions.dtype)
    final_vel = solver.velocities.to_numpy().astype(velocities.dtype)
    final_failed = solver.spring_failed.to_numpy() == 1
    final_proj_pos = solver.proj_position.to_numpy().astype(proj_position.dtype)
    final_proj_vel = solver.proj_velocity.to_numpy().astype(proj_velocity.dtype)
    final_damp = float(solver.damp_dissipated[None])
    final_failure = float(solver.failure_dissipated[None])
    final_clamp = float(solver.clamp_dissipated[None])

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
    )
