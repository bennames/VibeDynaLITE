from __future__ import annotations

import numpy as np
import pytest

from kevlargrid.solver.timestep import compute_cfl_timestep


class TestCFLTimestep:
    """Tests for Courant-Friedrichs-Lewy (CFL) stable timestep calculations."""

    def test_cfl_timestep_known_values(self) -> None:
        """Verify that the calculated dt matches analytical hand calculation."""
        stiffnesses = np.array([100.0, 150.0])
        masses = np.array([2.0, 3.0])
        dx = 1.0
        cfl = 0.8

        # m_min = 2.0, k_max = 150.0
        # dt_crit = sqrt(2.0 / 150.0) = sqrt(1.0 / 75.0) = 0.11547
        # Expected dt = 0.8 * 0.11547 = 0.092376
        dt = compute_cfl_timestep(stiffnesses, masses, dx, cfl)
        expected = 0.8 * np.sqrt(2.0 / 150.0)
        assert np.abs(dt - expected) < 1e-7

    def test_cfl_safety_factor(self) -> None:
        """Verify that reducing the CFL ratio reduces the stable timestep proportionally."""
        stiffnesses = np.array([100.0])
        masses = np.array([4.0])
        dx = 1.0

        # m_min = 4.0, k_max = 100.0 -> dt_crit = sqrt(4/100) = 0.2
        dt1 = compute_cfl_timestep(stiffnesses, masses, dx, 0.8)
        dt2 = compute_cfl_timestep(stiffnesses, masses, dx, 0.4)
        assert np.abs(dt1 - 0.16) < 1e-7
        assert np.abs(dt2 - 0.08) < 1e-7

    def test_cfl_range_validation(self) -> None:
        """Verify that a CFL ratio outside the range (0, 1] raises an error."""
        stiffnesses = np.array([100.0])
        masses = np.array([1.0])
        dx = 1.0

        with pytest.raises(ValueError, match="CFL safety factor"):
            compute_cfl_timestep(stiffnesses, masses, dx, 0.0)

        with pytest.raises(ValueError, match="CFL safety factor"):
            compute_cfl_timestep(stiffnesses, masses, dx, 1.1)

        with pytest.raises(ValueError, match="CFL safety factor"):
            compute_cfl_timestep(stiffnesses, masses, dx, -0.5)

    def test_dynamic_nodal_cfl_timestep(self) -> None:
        """Verify that the dynamic nodal stiffness Taichi kernel computes the expected values."""
        from kevlargrid.solver.taichi_solver import TaichiSolver

        positions = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=np.float32)
        velocities = np.zeros_like(positions)
        springs = np.array([[0, 1]], dtype=np.int32)
        stiffnesses = np.array([1000.0], dtype=np.float32)
        rest_lengths = np.array([1.0], dtype=np.float32)
        failed = np.zeros(1, dtype=np.int32)
        masses = np.array([1.0, 1.0], dtype=np.float32)
        tension_only = np.zeros(1, dtype=np.int32)
        boundary_mask = np.zeros(2, dtype=np.int32)
        nodal_external_forces = np.zeros_like(positions)
        node_initial_springs = np.array([1, 1], dtype=np.int32)

        solver = TaichiSolver(
            n_nodes=2,
            n_springs=1,
            positions_init=positions,
            velocities_init=velocities,
            springs_init=springs,
            stiffnesses_init=stiffnesses,
            rest_lengths_init=rest_lengths,
            failed_init=failed,
            masses_init=masses,
            tension_only_init=tension_only,
            boundary_mask_init=boundary_mask,
            nodal_external_forces_init=nodal_external_forces,
            proj_position_init=np.zeros(3),
            proj_velocity_init=np.zeros(3),
            proj_mass_init=1.0,
            strike_direction_init=1.0,
            node_initial_springs_init=node_initial_springs,
        )

        dt = solver.compute_dynamic_dt(
            failure_strain=0.5,
            damage_onset_strain=0.3,
            w_h=0.0,
            t_h=0.0,
            k_penalty=0.0,
            proximity_threshold=0.0,
            n_nodes_per_layer=2,
            n_plies=1,
            t_ply=0.0,
            cfl_factor=1.0,
        )
        # k_i = 500, m_i = 1.0 -> dt_i = sqrt(1.0 / 500.0)
        assert np.abs(dt - np.sqrt(1.0 / 500.0)) < 1e-4

