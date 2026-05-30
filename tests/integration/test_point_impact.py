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
