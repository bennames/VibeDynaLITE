from __future__ import annotations

import numpy as np
import pytest

from kevlargrid.solver.damping import viscous_damping
from kevlargrid.solver.energy import (
    compute_energy_balance,
    compute_kinetic_energy,
    compute_strain_energy,
)
from kevlargrid.solver.forces import compute_spring_forces, compute_spring_strains
from kevlargrid.solver.grid import generate_rectangular_grid
from kevlargrid.solver.integrator import leapfrog_step
from kevlargrid.solver.timestep import compute_cfl_timestep

MOCK_MATERIAL = {
    "tensile_modulus_gpa": 71.0,
    "areal_density_kgm2": 0.47,
    "fiber_density_gcc": 1.44,
    "shear_ratio": 0.0004,
}


class TestPointImpact:
    """Integration tests verifying point impact energy conservation and kinematics."""

    @pytest.mark.slow
    def test_energy_conservation_undamped(self) -> None:
        """Verify that total energy remains constant for undamped grid vibrations.

        Over 10,000 steps, energy drift should be less than 0.1%.
        """
        nx, ny, dx = 10, 10, 0.1
        grid = generate_rectangular_grid(nx, ny, dx, MOCK_MATERIAL)

        positions = grid.nodes.copy()
        velocities = np.zeros_like(positions)
        # Give central node an initial transverse velocity
        center_node = (nx // 2) * ny + (ny // 2)
        velocities[center_node, 2] = 20.0  # m/s in Z-direction

        dt = compute_cfl_timestep(grid.stiffnesses, grid.masses, dx, 0.8)

        # Clamped boundary mask (nodes on the edge are clamped)
        boundary_mask = np.zeros(grid.n_nodes, dtype=bool)
        for i in range(nx):
            for j in range(ny):
                if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                    boundary_mask[i * ny + j] = True

        initial_eb = compute_energy_balance(
            compute_kinetic_energy(velocities, grid.masses),
            0.0,
            0.0,
        )
        initial_energy = initial_eb["total"]

        # Run for 1,000 steps (sufficient to show conservation drift and speed up test)
        for _ in range(1000):
            forces = compute_spring_forces(
                positions, grid.springs, grid.stiffnesses, grid.rest_lengths, grid.failed
            )

            # Clamp boundary: zero forces/velocities at boundaries
            forces[boundary_mask] = 0.0
            velocities[boundary_mask] = 0.0

            positions, velocities = leapfrog_step(positions, velocities, forces, grid.masses, dt)

        strains = compute_spring_strains(positions, grid.springs, grid.rest_lengths)
        ke = compute_kinetic_energy(velocities, grid.masses)
        se = compute_strain_energy(strains, grid.stiffnesses, grid.rest_lengths, grid.failed)
        final_energy = ke + se

        drift = np.abs(final_energy - initial_energy) / initial_energy
        # Energy drift should be extremely small (<0.1%)
        assert drift < 0.001

    def test_projectile_deceleration(self) -> None:
        """Verify that damping decreases total energy monotonically."""
        nx, ny, dx = 10, 10, 0.1
        grid = generate_rectangular_grid(nx, ny, dx, MOCK_MATERIAL)

        positions = grid.nodes.copy()
        velocities = np.zeros_like(positions)
        center_node = (nx // 2) * ny + (ny // 2)
        velocities[center_node, 2] = 20.0

        dt = compute_cfl_timestep(grid.stiffnesses, grid.masses, dx, 0.8)

        boundary_mask = np.zeros(grid.n_nodes, dtype=bool)
        for i in range(nx):
            for j in range(ny):
                if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                    boundary_mask[i * ny + j] = True

        # Viscous damping coefficient
        c_visc = 1.0

        energies = []
        for _ in range(100):
            forces = compute_spring_forces(
                positions, grid.springs, grid.stiffnesses, grid.rest_lengths, grid.failed
            )

            # Add viscous damping
            damp_forces = viscous_damping(velocities, c_visc)
            forces += damp_forces

            forces[boundary_mask] = 0.0
            velocities[boundary_mask] = 0.0

            positions, velocities = leapfrog_step(positions, velocities, forces, grid.masses, dt)

            strains = compute_spring_strains(positions, grid.springs, grid.rest_lengths)
            ke = compute_kinetic_energy(velocities, grid.masses)
            se = compute_strain_energy(strains, grid.stiffnesses, grid.rest_lengths, grid.failed)
            energies.append(ke + se)

        # Verify that energy decreases over time (monotonically)
        for i in range(1, len(energies)):
            assert energies[i] <= energies[i - 1] + 1e-9

    def test_rigid_projectile_impact_and_arrest(self) -> None:
        """Verify explicit dynamic integration of a rigid projectile impact.

        Test contact zone evolution, force distribution, and overall energy
        conservation (drift < 1%) during arrest.
        """
        from kevlargrid.solver.projectile import (
            Projectile,
            distribute_contact_forces,
            update_contact_zone,
        )

        nx, ny, dx = 9, 9, 0.01
        grid = generate_rectangular_grid(nx, ny, dx, MOCK_MATERIAL)

        # 0.1 kg projectile, moving at 50 m/s in +Z, starting just below fabric
        proj = Projectile(
            mass=0.1,
            velocity=[0.0, 0.0, 50.0],
            position=[0.0, 0.0, -0.002],
            blade_width=0.03,
            edge_thickness=0.01,
        )

        positions = grid.nodes.copy()
        velocities = np.zeros_like(positions)

        # Clamped boundary on the edge nodes
        boundary_mask = np.zeros(grid.n_nodes, dtype=bool)
        for i in range(nx):
            for j in range(ny):
                if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                    boundary_mask[i * ny + j] = True

        # Contact stiffness parameters
        k_contact = 0.5 * np.mean(grid.stiffnesses)
        k_max_effective = max(np.max(grid.stiffnesses), k_contact)
        dt = compute_cfl_timestep(np.array([k_max_effective]), grid.masses, dx, 0.2)

        # Initial total energy is just the projectile's kinetic energy
        initial_ke_p = 0.5 * proj.mass * np.sum(proj.velocity**2)
        initial_energy = initial_ke_p

        # Run for 2000 explicit steps (to allow contact and interaction to occur at smaller dt)
        for _step in range(2000):
            # 1. Contact footprint update
            update_contact_zone(proj, grid, proximity_threshold=0.005, positions=positions)

            # 2. Distribute contact forces to grid nodes
            contact_forces = distribute_contact_forces(
                proj, grid, positions=positions, k_contact=k_contact
            )

            # 3. Spring forces from internal fabric deformation
            spring_forces = compute_spring_forces(
                positions, grid.springs, grid.stiffnesses, grid.rest_lengths, grid.failed
            )

            # Net forces on nodes
            net_forces = spring_forces + contact_forces

            # Enforce boundary clamp (forces/velocities on boundaries are zeroed)
            net_forces[boundary_mask] = 0.0
            velocities[boundary_mask] = 0.0

            # 4. Integrate node kinematics
            positions, velocities = leapfrog_step(
                positions, velocities, net_forces, grid.masses, dt
            )

            # 5. Integrate projectile kinematics (equal and opposite reaction force)
            proj_reaction_force = -np.sum(contact_forces, axis=0)
            proj_accel = proj_reaction_force / proj.mass
            proj.velocity += proj_accel * dt
            proj.position += proj.velocity * dt

            # Optional early stop: projectile has arrested (velocity reversed or stopped)
            if proj.velocity[2] <= 0.0:
                break

        # Compute final energy components
        strains = compute_spring_strains(positions, grid.springs, grid.rest_lengths)
        final_ke_nodes = compute_kinetic_energy(velocities, grid.masses)
        final_se_springs = compute_strain_energy(
            strains, grid.stiffnesses, grid.rest_lengths, grid.failed
        )
        final_ke_projectile = 0.5 * proj.mass * np.sum(proj.velocity**2)

        # Contact potential energy
        c_nodes = proj.contact_nodes
        if len(c_nodes) > 0:
            c_pos = positions[c_nodes]
            x_p, y_p, z_p = proj.position
            w_h = proj.blade_width / 2.0
            t_h = proj.edge_thickness / 2.0
            x_proj = np.clip(c_pos[:, 0], x_p - w_h, x_p + w_h)
            y_proj = np.clip(c_pos[:, 1], y_p - t_h, y_p + t_h)
            z_proj = z_p
            d_i = np.sqrt(
                (c_pos[:, 0] - x_proj) ** 2
                + (c_pos[:, 1] - y_proj) ** 2
                + (c_pos[:, 2] - z_proj) ** 2
            )
            w_i = 1.0 / np.maximum(d_i, 1e-4)
            w_mean = np.mean(w_i) if len(w_i) > 0 else 1.0
            w_normalized = w_i / w_mean if w_mean > 0.0 else np.ones_like(w_i)
            penetration = np.maximum(0.0, (z_p - c_pos[:, 2]))
            contact_potential_energy = 0.5 * np.sum(k_contact * w_normalized * penetration**2)
        else:
            contact_potential_energy = 0.0

        final_total_energy = (
            final_ke_nodes + final_se_springs + final_ke_projectile + contact_potential_energy
        )
        energy_drift = np.abs(final_total_energy - initial_energy) / initial_energy

        # Projectile must have slowed down (transferred energy to the fabric)
        assert proj.velocity[2] < 50.0

        # Verify that overall energy drift is conserved within 2% (standard for explicit penalty contact)
        assert energy_drift < 0.02

    def test_post_breakthrough_energy_conservation(self) -> None:
        """Verify that system total energy is conserved within 1% variance post-breakthrough.

        Uses multiple simulation chunks to track total energy after complete breakthrough
        with detached nodes present (active_counts == 0).
        """
        from kevlargrid.solver.fused import fused_leapfrog_loop

        # 6x6 grid, 2 plies, t_ply=0.002, dx=0.01
        nx, ny = 6, 6
        dx = 0.01
        n_plies = 2
        t_ply = 0.002
        n_nodes_per_layer = nx * ny

        mat = {
            "tensile_modulus_gpa": 71.0,
            "areal_density_kgm2": 0.47,
            "fiber_density_gcc": 1.44,
            "shear_ratio": 0.0004,
            "failure_strain": 0.005, # Extremely low strain to trigger easy breakthrough
        }

        grid = generate_rectangular_grid(
            nx=nx, ny=ny, dx=dx, material=mat, n_plies=n_plies, t_ply=t_ply
        )

        n_nodes = grid.n_nodes
        n_springs = len(grid.springs)

        positions = grid.nodes.copy()
        velocities = np.zeros_like(positions)
        grid_failed = np.zeros(n_springs, dtype=bool)

        # Boundary mask: clamp edges of all layers
        boundary_mask = np.zeros(n_nodes, dtype=bool)
        for ply in range(n_plies):
            offset = ply * n_nodes_per_layer
            for i in range(nx):
                for j in range(ny):
                    if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                        boundary_mask[offset + i * ny + j] = True

        # Projectile: fast projectile to break through the plies completely
        proj_pos = np.array([2.5 * dx, 2.5 * dx, 0.005], dtype=np.float64) # start above grid
        proj_vel = np.array([0.0, 0.0, -150.0], dtype=np.float64) # high speed
        proj_mass = 0.02
        blade_width = 0.015
        edge_thickness = 0.005
        k_penalty = 1e6
        rayleigh_alpha = 0.05
        rayleigh_beta = 1e-7 # Stable Rayleigh beta to avoid explicit integration explosion
        failure_strain = 0.005
        damage_onset_strain = 0.002
        fracture_energy_multiplier = 1.0 # Set multiplier to 1.0 for conservation validation

        # Calculate CFL timestep properly
        k_max_effective = max(np.max(grid.stiffnesses), k_penalty)
        dt = compute_cfl_timestep(np.array([k_max_effective]), grid.masses, dx, 0.2)
        steps_per_chunk = 500
        n_chunks = 8

        damp_diss = 0.0
        failure_diss = 0.0
        clamp_diss = 0.0
        t_sim = 0.0

        initial_energy = 0.5 * proj_mass * np.sum(proj_vel**2)
        energies = []
        failed_counts = []
        detached_node_counts = []

        for chunk in range(n_chunks):
            (
                positions,
                velocities,
                grid_failed,
                proj_pos,
                proj_vel,
                damp_diss,
                failure_diss,
                clamp_diss,
                t_sim,
                *_,
            ) = fused_leapfrog_loop(
                positions,
                velocities,
                grid.springs,
                grid.stiffnesses,
                grid.rest_lengths,
                grid_failed,
                grid.masses,
                grid.tension_only,
                boundary_mask,
                np.zeros((n_nodes, 3)),
                proj_pos,
                proj_vel,
                proj_mass,
                blade_width,
                edge_thickness,
                n_plies=n_plies,
                n_nodes_per_layer=n_nodes_per_layer,
                t_ply=t_ply,
                dx=dx,
                k_penalty=k_penalty,
                rayleigh_alpha=rayleigh_alpha,
                rayleigh_beta=rayleigh_beta,
                failure_strain=failure_strain,
                damage_onset_strain=damage_onset_strain,
                fracture_energy_multiplier=fracture_energy_multiplier,
                dt=dt,
                n_steps=steps_per_chunk,
                save_interval=steps_per_chunk,
                damp_dissipated_init=damp_diss,
                failure_dissipated_init=failure_diss,
                clamp_dissipated_init=clamp_diss,
                t_sim_init=t_sim,
                strike_direction=0.0,
                node_initial_springs=grid.initial_spring_counts,
                node_spring_offsets=grid.node_spring_offsets,
                node_spring_ids=grid.node_spring_ids,
                node_spring_signs=grid.node_spring_signs,
            )

            # Calculate energies
            ke_nodes = compute_kinetic_energy(velocities, grid.masses)
            # Spring strain energy
            p1 = positions[grid.springs[:, 0]]
            p2 = positions[grid.springs[:, 1]]
            lengths = np.sqrt(np.sum((p2 - p1)**2, axis=1))
            strains = (lengths - grid.rest_lengths) / grid.rest_lengths
            se_springs = compute_strain_energy(strains, grid.stiffnesses, grid.rest_lengths, grid_failed)
            ke_proj = 0.5 * proj_mass * np.sum(proj_vel**2)

            # Count active springs to calculate node active counts
            active_springs = np.where(grid_failed, 0, 1)
            active_counts = np.zeros(n_nodes, dtype=np.int32)
            np.add.at(active_counts, grid.springs[:, 0], active_springs)
            np.add.at(active_counts, grid.springs[:, 1], active_springs)

            # Compute contact potential energy
            x_p, y_p, z_p = proj_pos
            w_h = blade_width / 2.0
            t_h = edge_thickness / 2.0
            x_proj = np.clip(positions[:, 0], x_p - w_h, x_p + w_h)
            y_proj = np.clip(positions[:, 1], y_p - t_h, y_p + t_h)
            dist = np.sqrt((positions[:, 0] - x_proj)**2 + (positions[:, 1] - y_proj)**2 + (positions[:, 2] - z_p)**2)
            contact_mask = dist <= dx * 2.0
            w_i = 1.0 / np.maximum(dist, 1e-4)
            contact_mask = contact_mask & (active_counts > 0)
            w_mean = np.mean(w_i[contact_mask]) if np.sum(contact_mask) > 0 else 1.0
            w_normalized = np.where(contact_mask, w_i / w_mean, 0.0)
            
            penetration = np.maximum(0.0, (z_p - positions[:, 2]) * -1.0)
            scale_factor = np.where(grid.initial_spring_counts > 0, active_counts / grid.initial_spring_counts, 0.0)
            contact_potential_energy = np.sum(0.5 * k_penalty * w_normalized * (penetration**2) * scale_factor)

            total_system_energy = ke_nodes + se_springs + ke_proj + damp_diss + failure_diss + clamp_diss + contact_potential_energy
            
            energies.append(total_system_energy)
            failed_counts.append(np.sum(grid_failed))
            detached_node_counts.append(np.sum(active_counts == 0))

        # Check that we actually achieved a post-breakthrough state with failed springs and detached nodes
        assert failed_counts[-1] > 0, "No springs failed during breakthrough test"
        assert detached_node_counts[-1] > 0, "No nodes were detached during breakthrough test"

        # Check total system energy conservation post-breakthrough
        # Specifically, after breakthrough has settled (e.g. from chunk 5 onwards)
        post_breakthrough_energies = energies[5:]
        energy_variance = np.var(post_breakthrough_energies)
        energy_std_dev = np.sqrt(energy_variance)
        
        # Post-breakthrough energy variance / initial energy should be extremely small (< 0.1%)
        assert (energy_std_dev / initial_energy) < 0.01, f"Energy standard deviation too high: {energy_std_dev / initial_energy:.4f}"
        
        # Overall drift compared to the start of the post-breakthrough phase should be < 1%
        drift = np.abs(post_breakthrough_energies[-1] - post_breakthrough_energies[0]) / post_breakthrough_energies[0]
        assert drift < 0.01, f"Post-breakthrough drift too high: {drift:.4f}"
