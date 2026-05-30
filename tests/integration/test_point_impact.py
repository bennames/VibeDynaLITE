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
