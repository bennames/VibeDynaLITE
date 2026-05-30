from __future__ import annotations

import numpy as np
import pytest

from kevlargrid.solver.forces import compute_spring_forces
from kevlargrid.solver.grid import generate_rectangular_grid
from kevlargrid.solver.integrator import leapfrog_step
from kevlargrid.solver.timestep import compute_cfl_timestep

MOCK_MATERIAL = {
    "tensile_modulus_gpa": 71.0,
    "areal_density_kgm2": 0.47,
    "fiber_density_gcc": 1.44,
    "shear_ratio": 0.0004,
}


class TestWavePropagation:
    """Integration tests verifying wave propagation velocities in woven fabric grids."""

    @pytest.mark.slow
    def test_1d_wave_speed(self) -> None:
        """Verify that wave speed in 1D matches analytical solutions.

        Impulse propagates through a 1D grid; wave speed matches the
        analytical c = sqrt(E / rho) within 2%.
        """
        # 1D line of nodes: nx=200, ny=1, dx=0.1
        nx, ny, dx = 200, 1, 0.1
        grid = generate_rectangular_grid(nx, ny, dx, MOCK_MATERIAL)

        # Theoretical wave speed
        # k = stiffnesses[0]
        # m = masses[interior]
        k = grid.stiffnesses[0]
        m = grid.masses[100]  # Interior node mass
        c_theory = dx * np.sqrt(k / m)

        # Timestep
        dt = compute_cfl_timestep(grid.stiffnesses, grid.masses, dx, 0.8)

        # Apply velocity impulse at node 0 along -X axis to pull the tension-only fabric
        positions = grid.nodes.copy()
        velocities = np.zeros_like(positions)
        velocities[0, 0] = -50.0  # m/s along -X (tension)

        # Run explicit solver loop
        t = 0.0
        target_node = 80
        threshold = 1e-4
        arrival_time = None

        # Max 1000 steps
        for _ in range(1000):
            # Compute forces
            forces = compute_spring_forces(
                positions, grid.springs, grid.stiffnesses, grid.rest_lengths, grid.failed
            )

            # Enforce zero boundary at the right end (node 199) to keep it stable
            forces[199] = 0.0
            velocities[199] = 0.0

            # Step integrator
            positions, velocities = leapfrog_step(positions, velocities, forces, grid.masses, dt)
            t += dt

            # Check if wave has reached target node (significant displacement)
            if np.abs(positions[target_node, 0] - grid.nodes[target_node, 0]) > threshold:
                arrival_time = t
                break

        assert arrival_time is not None, "Wave did not reach target node"
        expected_time = (target_node * dx) / c_theory
        error = np.abs(arrival_time - expected_time) / expected_time
        # wave arrival matches theoretical speed within 2%
        assert error < 0.02

    def test_wave_reflection_clamped(self) -> None:
        """Verify that waves reflect correctly off clamped boundary nodes."""
        nx, ny, dx = 50, 1, 0.1
        grid = generate_rectangular_grid(nx, ny, dx, MOCK_MATERIAL)

        # Clamped boundary at the right node (node 49)
        clamped_mask = np.zeros(grid.n_nodes, dtype=bool)
        clamped_mask[49] = True

        positions = grid.nodes.copy()
        velocities = np.zeros_like(positions)
        # Large displacement wave traveling towards node 49
        velocities[40, 0] = 100.0
        dt = compute_cfl_timestep(grid.stiffnesses, grid.masses, dx, 0.8)

        # Run simulation and check if wave reflects (velocity sign changes upon reflection)
        has_reflected = False
        for _ in range(200):
            forces = compute_spring_forces(
                positions, grid.springs, grid.stiffnesses, grid.rest_lengths, grid.failed
            )
            # Enforce clamped boundary: velocity is zero at node 49
            forces[49] = 0.0
            velocities[49] = 0.0

            positions, velocities = leapfrog_step(positions, velocities, forces, grid.masses, dt)

            # If node 45 has negative velocity, it means wave reflected off boundary at 49
            if velocities[45, 0] < -0.1:
                has_reflected = True
                break

        assert has_reflected is True
