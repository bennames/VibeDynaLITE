from __future__ import annotations

import numpy as np
import pytest

from kevlargrid.solver.grid import generate_rectangular_grid

MOCK_MATERIAL = {
    "tensile_modulus_gpa": 71.0,
    "areal_density_kgm2": 0.47,
    "fiber_density_gcc": 1.44,
    "shear_ratio": 0.0004,
}


class TestMultiPlyImpact:
    """Integration tests verifying multi-ply fabric interaction modes."""

    @pytest.mark.slow
    def test_mode_a_ply_scaling(self) -> None:
        """Verify that Mode A wave speed scales correctly with ply count.

        Mode A assumes uniform equivalent areal density and stiffness scaling.
        Doubling plies should double masses and stiffnesses, keeping wave speed unchanged.
        """
        grid_1ply = generate_rectangular_grid(10, 10, 0.1, MOCK_MATERIAL, n_plies=1)
        grid_4plies = generate_rectangular_grid(10, 10, 0.1, MOCK_MATERIAL, n_plies=4)

        # 1. Assert mass and stiffness scaling
        np.testing.assert_allclose(grid_4plies.masses, 4.0 * grid_1ply.masses)
        np.testing.assert_allclose(grid_4plies.stiffnesses, 4.0 * grid_1ply.stiffnesses)

        # 2. Assert wave speeds (E / m ratio) are exactly the same
        c_1ply = np.sqrt(grid_1ply.stiffnesses[0] / grid_1ply.masses[10])
        c_4plies = np.sqrt(grid_4plies.stiffnesses[0] / grid_4plies.masses[10])
        assert np.abs(c_1ply - c_4plies) < 1e-7

    def test_mode_b_layer_count(self) -> None:
        """Verify correct grid layer count and spacing for Mode B.

        Mode B explicitly instantiates individual plies with Z separation.
        """
        nx, ny, dx = 6, 6, 0.01
        n_plies = 4
        t_ply = 0.001  # 1mm spacing

        grid = generate_rectangular_grid(nx, ny, dx, MOCK_MATERIAL, n_plies=n_plies, t_ply=t_ply)

        # Expected counts
        n_nodes_per_layer = nx * ny
        assert grid.n_nodes == n_plies * n_nodes_per_layer

        # Verify initial Z positions of layer centers
        for ply in range(n_plies):
            start = ply * n_nodes_per_layer
            end = start + n_nodes_per_layer
            layer_z = grid.nodes[start:end, 2]
            np.testing.assert_allclose(layer_z, ply * t_ply)

    def test_mode_b_sequential_contact_and_energy(self) -> None:
        """Verify sequential ply interaction, penalty forces, and energy conservation.

        Confirm that loading transfers from layer 0 to layer 1 via inter-ply contact
        and that the total combined energy (projectile + nodes + springs + inter-ply potential)
        is conserved within 2% tolerance.
        """
        from kevlargrid.solver.energy import compute_kinetic_energy, compute_strain_energy
        from kevlargrid.solver.forces import (
            compute_interply_contact_forces,
            compute_spring_forces,
            compute_spring_strains,
        )
        from kevlargrid.solver.integrator import leapfrog_step
        from kevlargrid.solver.projectile import (
            Projectile,
            distribute_contact_forces,
            update_contact_zone,
        )
        from kevlargrid.solver.timestep import compute_cfl_timestep

        nx, ny, dx = 5, 5, 0.01
        n_plies = 2
        t_ply = 0.001
        grid = generate_rectangular_grid(nx, ny, dx, MOCK_MATERIAL, n_plies=n_plies, t_ply=t_ply)

        # 0.05 kg projectile, moving at 40 m/s in +Z, starting just below layer 0 (Z = -0.002)
        proj = Projectile(
            mass=0.05,
            velocity=[0.0, 0.0, 40.0],
            position=[0.0, 0.0, -0.002],
            blade_width=0.02,
            edge_thickness=0.005,
        )

        positions = grid.nodes.copy()
        velocities = np.zeros_like(positions)

        # Clamped boundary on the edge nodes of BOTH plies
        boundary_mask = np.zeros(grid.n_nodes, dtype=bool)
        n_nodes_per_layer = nx * ny
        for ply in range(n_plies):
            offset = ply * n_nodes_per_layer
            for i in range(nx):
                for j in range(ny):
                    if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                        boundary_mask[offset + i * ny + j] = True

        # Stable timestep limits factoring in the contact stiffness
        k_penalty = 1.0 * np.mean(grid.stiffnesses)
        k_max_effective = max(np.max(grid.stiffnesses), k_penalty)
        dt = compute_cfl_timestep(np.array([k_max_effective]), grid.masses, dx, 0.1)

        initial_energy = 0.5 * proj.mass * np.sum(proj.velocity**2)

        # Run explicit dynamics
        contact_ply1_occurred = False
        for _step in range(1000):
            # 1. Projectile-to-fabric contact (on layer 0)
            update_contact_zone(proj, grid, proximity_threshold=0.003, positions=positions)
            proj_forces = distribute_contact_forces(
                proj, grid, positions=positions, k_contact=k_penalty
            )

            # 2. Inter-ply contact forces (between Layer 0 and Layer 1)
            interply_forces, interply_energy = compute_interply_contact_forces(
                positions, n_nodes_per_layer, n_plies, t_ply, k_penalty
            )

            # Check if layer 1 nodes started experiencing inter-ply contact forces (indicating load transfer!)
            if np.any(np.abs(interply_forces[n_nodes_per_layer:, 2]) > 0.0):
                contact_ply1_occurred = True

            # 3. Internal spring forces (warp/weft/shear)
            spring_forces = compute_spring_forces(
                positions, grid.springs, grid.stiffnesses, grid.rest_lengths, grid.failed
            )

            # Net forces
            net_forces = spring_forces + proj_forces + interply_forces
            net_forces[boundary_mask] = 0.0
            velocities[boundary_mask] = 0.0

            # 4. Node integration
            positions, velocities = leapfrog_step(
                positions, velocities, net_forces, grid.masses, dt
            )

            # 5. Projectile integration
            proj_reaction_force = -np.sum(proj_forces, axis=0)
            proj_accel = proj_reaction_force / proj.mass
            proj.velocity += proj_accel * dt
            proj.position += proj.velocity * dt

            if proj.velocity[2] <= 0.0:
                break

        # Verify load transfer to ply 1 occurred successfully!
        assert contact_ply1_occurred is True

        # Total energy balance
        strains = compute_spring_strains(positions, grid.springs, grid.rest_lengths)
        final_ke_nodes = compute_kinetic_energy(velocities, grid.masses)
        final_se_springs = compute_strain_energy(
            strains, grid.stiffnesses, grid.rest_lengths, grid.failed
        )
        final_ke_projectile = 0.5 * proj.mass * np.sum(proj.velocity**2)

        # Projectile-to-fabric contact potential energy
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
            proj_contact_energy = 0.5 * np.sum(k_penalty * w_normalized * penetration**2)
        else:
            proj_contact_energy = 0.0

        final_total_energy = (
            final_ke_nodes
            + final_se_springs
            + final_ke_projectile
            + proj_contact_energy
            + interply_energy
        )
        energy_drift = np.abs(final_total_energy - initial_energy) / initial_energy

        # Projectile must have slowed down (transferred energy to the multi-ply stacked panel)
        assert proj.velocity[2] < 40.0

        # Energy conserved within 2% (standard for penalty contact explicit dynamics)
        assert energy_drift < 0.02

    def test_mode_b_rupture_energy_conservation(self) -> None:
        """Verify energy conservation in multi-ply panel under high velocity impact with spring ruptures."""
        from kevlargrid.solver.energy import compute_kinetic_energy, compute_strain_energy
        from kevlargrid.solver.forces import (
            compute_interply_contact_forces,
            compute_spring_forces,
            compute_spring_strains,
        )
        from kevlargrid.solver.integrator import leapfrog_step
        from kevlargrid.solver.projectile import (
            Projectile,
            distribute_contact_forces,
            update_contact_zone,
        )
        from kevlargrid.solver.timestep import compute_cfl_timestep

        nx, ny, dx = 6, 6, 0.01
        n_plies = 2
        t_ply = 0.001

        # Override material to have a lower failure strain so springs break easily
        material = MOCK_MATERIAL.copy()
        material["failure_strain"] = 0.02
        material["fracture_energy_multiplier"] = 1.0

        grid = generate_rectangular_grid(nx, ny, dx, material, n_plies=n_plies, t_ply=t_ply)

        # High-velocity impact (120 m/s) to force spring failures
        proj = Projectile(
            mass=0.05,
            velocity=[0.0, 0.0, 120.0],
            position=[0.0, 0.0, -0.002],
            blade_width=0.02,
            edge_thickness=0.005,
        )

        positions = grid.nodes.copy()
        velocities = np.zeros_like(positions)
        boundary_mask = np.zeros(grid.n_nodes, dtype=bool)

        k_penalty = 5.0 * np.mean(grid.stiffnesses)
        k_max_effective = max(np.max(grid.stiffnesses), k_penalty)

        # Use stable CFL factor 0.4
        dt = compute_cfl_timestep(np.array([k_max_effective]), grid.masses, dx, 0.4)

        initial_energy = 0.5 * proj.mass * np.sum(proj.velocity**2)
        failure_dissipated = 0.0

        # Run for 200 steps
        for _step in range(200):
            # 1. Spring failures & forces
            strains = compute_spring_strains(positions, grid.springs, grid.rest_lengths)
            newly_failed = (~grid.failed) & (strains > material["failure_strain"])

            # Record fracture energy dissipation
            fracture_se = np.where(
                newly_failed, 0.5 * grid.stiffnesses * (strains * grid.rest_lengths) ** 2, 0.0
            )
            failure_dissipated += np.sum(fracture_se) * material["fracture_energy_multiplier"]
            grid.failed = grid.failed | newly_failed

            spring_forces = compute_spring_forces(
                positions, grid.springs, grid.stiffnesses, grid.rest_lengths, grid.failed
            )

            # 2. Projectile contact forces
            update_contact_zone(proj, grid, proximity_threshold=0.003, positions=positions)
            active_counts = np.zeros(grid.n_nodes)
            active_springs = np.where(grid.failed, 0, 1)
            np.add.at(active_counts, grid.springs[:, 0], active_springs)
            np.add.at(active_counts, grid.springs[:, 1], active_springs)

            proj_forces = distribute_contact_forces(
                proj, grid, positions=positions, k_contact=k_penalty
            )
            # Scale by remaining active springs
            scale_factor = np.where(
                grid.initial_spring_counts > 0, active_counts / grid.initial_spring_counts, 0.0
            )
            proj_forces = proj_forces * scale_factor[:, np.newaxis]

            # 3. Inter-ply contact forces (using the updated symmetric function)
            interply_forces, interply_energy = compute_interply_contact_forces(
                positions, nx * ny, n_plies, t_ply, k_penalty, active_counts
            )

            # Net forces
            net_forces = spring_forces + proj_forces + interply_forces
            net_forces[boundary_mask] = 0.0
            velocities[boundary_mask] = 0.0

            # 4. Integrate nodes
            positions, velocities = leapfrog_step(
                positions, velocities, net_forces, grid.masses, dt
            )

            # 5. Integrate projectile
            proj_reaction_force = -np.sum(proj_forces, axis=0)
            proj_accel = proj_reaction_force / proj.mass
            proj.velocity += proj_accel * dt
            proj.position += proj.velocity * dt

        # Verify spring failure occurred
        assert np.sum(grid.failed) > 0

        # Total energy balance
        final_ke_nodes = compute_kinetic_energy(velocities, grid.masses)
        final_se_springs = compute_strain_energy(
            strains, grid.stiffnesses, grid.rest_lengths, grid.failed
        )
        final_ke_projectile = 0.5 * proj.mass * np.sum(proj.velocity**2)

        # Projectile contact potential energy
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
            proj_contact_energy = 0.5 * np.sum(k_penalty * w_normalized * penetration**2)
        else:
            proj_contact_energy = 0.0

        final_total_energy = (
            final_ke_nodes
            + final_se_springs
            + final_ke_projectile
            + proj_contact_energy
            + interply_energy
            + failure_dissipated
        )
        energy_drift = np.abs(final_total_energy - initial_energy) / initial_energy

        # Must conserve energy to a very tight tolerance (< 1.5%) S7.13
        assert energy_drift < 0.015
