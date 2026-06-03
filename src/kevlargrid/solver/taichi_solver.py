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
        proj_position_init: np.ndarray,
        proj_velocity_init: np.ndarray,
        proj_mass_init: float,
    ) -> None:
        self.n_nodes = n_nodes
        self.n_springs = n_springs

        real_type = ti.f32

        # Node fields
        self.positions = ti.Vector.field(3, dtype=real_type, shape=n_nodes)
        self.velocities = ti.Vector.field(3, dtype=real_type, shape=n_nodes)
        self.forces = ti.Vector.field(3, dtype=real_type, shape=n_nodes)
        self.masses = ti.field(dtype=real_type, shape=n_nodes)
        self.boundary_mask = ti.field(dtype=ti.i32, shape=n_nodes)

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

        # Dissipated energy field
        self.damp_dissipated = ti.field(dtype=real_type, shape=())

        # Helper reduction fields for projectile contact
        self.node_w = ti.field(dtype=real_type, shape=n_nodes)
        self.w_sum = ti.field(dtype=real_type, shape=())
        self.n_contacts = ti.field(dtype=ti.i32, shape=())

        # Initialize GPU buffers from numpy inputs
        self.positions.from_numpy(positions_init.astype(np.float32))
        self.velocities.from_numpy(velocities_init.astype(np.float32))
        self.masses.from_numpy(masses_init.astype(np.float32))
        self.boundary_mask.from_numpy(boundary_mask_init.astype(np.int32))

        self.springs.from_numpy(springs_init.astype(np.int32))
        self.stiffnesses.from_numpy(stiffnesses_init.astype(np.float32))
        self.rest_lengths.from_numpy(rest_lengths_init.astype(np.float32))
        self.spring_failed.from_numpy(failed_init.astype(np.int32))
        self.tension_only.from_numpy(tension_only_init.astype(np.int32))

        self.proj_position[None] = proj_position_init.astype(np.float32)
        self.proj_velocity[None] = proj_velocity_init.astype(np.float32)
        self.proj_mass[None] = float(proj_mass_init)

        self.proj_reaction_force[None] = [0.0, 0.0, 0.0]
        self.damp_dissipated[None] = 0.0

    @ti.kernel
    def reset_forces(self):
        """Clear dynamic nodal forces and projectile reaction forces."""
        for i in range(self.n_nodes):
            self.forces[i] = ti.Vector([0.0, 0.0, 0.0])
        self.proj_reaction_force[None] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def compute_spring_forces(self):
        """Compute structural mass-spring forces and accumulate them on nodes."""
        for j in range(self.n_springs):
            if self.spring_failed[j] == 1:
                continue

            u, v = self.springs[j][0], self.springs[j][1]
            diff = self.positions[v] - self.positions[u]
            length = diff.norm()
            length_safe = length if length > 0.0 else 1.0
            strain = (length - self.rest_lengths[j]) / self.rest_lengths[j]

            f_mag = self.stiffnesses[j] * strain * self.rest_lengths[j]

            # Orthogonal/tension-only spring logic
            if self.tension_only[j] == 1 and strain < 0.0:
                f_mag = 0.0

            f_vec = (f_mag / length_safe) * diff

            # Atomic force accumulations
            ti.atomic_add(self.forces[u], f_vec)
            ti.atomic_add(self.forces[v], -f_vec)

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
                ti.atomic_add(self.forces[u].z, -f_mag)
                ti.atomic_add(self.forces[v].z, f_mag)

    @ti.kernel
    def compute_projectile_forces(
        self, w_h: ti.f32, t_h: ti.f32, k_penalty: ti.f32, proximity_threshold: ti.f32
    ):
        """Compute blade-to-mesh contact interface force distribution."""
        self.w_sum[None] = 0.0
        self.n_contacts[None] = 0

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
                    f_val = k_penalty * w_normalized * penetration
                    f_z = f_val * direction

                    self.forces[i].z += f_z
                    ti.atomic_add(self.proj_reaction_force[None].z, -f_z)

    @ti.kernel
    def integrate_nodes(self, dt: ti.f32, damping_coeff: ti.f32):
        """Update node velocities and coordinates using leapfrog time integration."""
        for i in range(self.n_nodes):
            if self.boundary_mask[i] == 1:
                self.forces[i] = ti.Vector([0.0, 0.0, 0.0])
                self.velocities[i] = ti.Vector([0.0, 0.0, 0.0])
                continue

            # Viscous damping force
            damp_f = -damping_coeff * self.velocities[i]
            p_damp = damp_f.dot(self.velocities[i])
            ti.atomic_add(self.damp_dissipated[None], -p_damp * dt)

            net_f = self.forces[i] + damp_f
            accel = net_f / self.masses[i]
            self.velocities[i] += accel * dt
            self.positions[i] += self.velocities[i] * dt

    @ti.kernel
    def integrate_projectile(self, dt: ti.f32):
        """Update rigid body kinetics representing striking projectile."""
        proj_accel = self.proj_reaction_force[None] / self.proj_mass[None]
        self.proj_velocity[None] += proj_accel * dt
        self.proj_position[None] += self.proj_velocity[None] * dt

    @ti.kernel
    def evolve_failures(self, failure_strain: ti.f32):
        """Evolve and record spring rupture flags irreversibly."""
        for j in range(self.n_springs):
            u, v = self.springs[j][0], self.springs[j][1]
            diff = self.positions[v] - self.positions[u]
            length = diff.norm()
            strain = (length - self.rest_lengths[j]) / self.rest_lengths[j]
            if strain > failure_strain:
                self.spring_failed[j] = 1


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
    damping_coeff: float,
    failure_strain: float,
    dt: float,
    n_steps: int,
    save_interval: int,
    damp_dissipated_init: float,
    t_sim_init: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
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

    # Allocate a new Taichi solver context on GPU
    solver = TaichiSolver(
        n_nodes=len(positions),
        n_springs=len(grid_springs),
        positions_init=positions,
        velocities_init=velocities,
        springs_init=grid_springs,
        stiffnesses_init=grid_stiffnesses,
        rest_lengths_init=grid_rest_lengths,
        failed_init=grid_failed,
        masses_init=grid_masses,
        tension_only_init=grid_tension_only,
        boundary_mask_init=boundary_mask,
        proj_position_init=proj_position,
        proj_velocity_init=proj_velocity,
        proj_mass_init=proj_mass,
    )
    solver.damp_dissipated[None] = damp_dissipated_init

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
        solver.reset_forces()

        # 1. Projectile Contact
        solver.compute_projectile_forces(w_h, t_h, k_penalty, proximity_threshold)

        # 2. Inter-ply Contact
        if n_plies > 1:
            solver.compute_interply_forces(n_nodes_per_layer, n_plies, t_ply, k_penalty)

        # 3. Internal Springs
        solver.compute_spring_forces()

        # 4. Integrate
        solver.integrate_nodes(dt, damping_coeff)
        solver.integrate_projectile(dt)

        # 5. Failure Evolution
        solver.evolve_failures(failure_strain)

        t_sim += dt

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
            se = compute_strain_energy(strains_cpu, grid_stiffnesses, grid_rest_lengths, failed_cpu)
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

    return (
        final_pos,
        final_vel,
        final_failed,
        final_proj_pos,
        final_proj_vel,
        final_damp,
        t_sim,
        hist_positions,
        hist_failed,
        hist_proj_pos,
        hist_time,
        hist_ke,
        hist_se,
        hist_proj_ke,
    )
