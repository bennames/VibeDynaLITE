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
        """Verify that the dynamic nodal stiffness JIT helpers compute the expected values."""
        from kevlargrid.solver.fused import numba_compute_effective_k, numba_sum_nodal_k_springs
        
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        springs = np.array([[0, 1]], dtype=np.int32)
        stiffnesses = np.array([1000.0], dtype=np.float64)
        rest_lengths = np.array([1.0], dtype=np.float64)
        failed = np.zeros(1, dtype=bool)
        
        # Stretched past damage onset but before failure:
        # strain = 0.4, damage = (0.4 - 0.3) / 0.2 = 0.5
        positions_degraded = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=np.float64)
        
        eff_k = numba_compute_effective_k(
            positions_degraded, springs, stiffnesses, rest_lengths, failed, 0.3, 0.5
        )
        # Expected: 1000 * (1.0 - 0.5) = 500.0
        assert np.abs(eff_k[0] - 500.0) < 1e-7
        
        # Nodal spring stiffness summation:
        node_spring_offsets = np.array([0, 1, 2], dtype=np.int32)
        node_spring_ids = np.array([0, 0], dtype=np.int32)
        nodal_k = numba_sum_nodal_k_springs(eff_k, node_spring_offsets, node_spring_ids)
        
        # Both nodes are connected to spring 0, so both get stiffness 500.0
        assert np.abs(nodal_k[0] - 500.0) < 1e-7
        assert np.abs(nodal_k[1] - 500.0) < 1e-7
