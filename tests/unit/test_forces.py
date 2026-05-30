from __future__ import annotations

import numpy as np

from kevlargrid.solver.forces import compute_spring_forces, compute_spring_strains


class TestSpringForce:
    """Tests for spring force computation."""

    def test_single_spring_10pct_stretch(self) -> None:
        """Spring stretched 10% returns F = k * 0.1 * L_rest."""
        # 2 nodes, 1 spring
        positions = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]])
        springs = np.array([[0, 1]], dtype=np.int32)
        stiffnesses = np.array([100.0])
        rest_lengths = np.array([1.0])
        failed = np.zeros(1, dtype=bool)

        forces = compute_spring_forces(positions, springs, stiffnesses, rest_lengths, failed)
        # Expected: F = 100 * 0.1 * 1.0 = 10.0 N along X-axis
        # On node 0: in direction of 1 -> positive X
        # On node 1: in direction of 0 -> negative X
        np.testing.assert_allclose(forces[0], np.array([10.0, 0.0, 0.0]))
        np.testing.assert_allclose(forces[1], np.array([-10.0, 0.0, 0.0]))

    def test_force_direction(self) -> None:
        """Force is directed along the spring axis."""
        # 45 deg stretch
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        springs = np.array([[0, 1]], dtype=np.int32)
        stiffnesses = np.array([10.0])
        rest_lengths = np.array([1.0])
        failed = np.zeros(1, dtype=bool)

        forces = compute_spring_forces(positions, springs, stiffnesses, rest_lengths, failed)
        # Force is parallel to unit axis
        unit_vec = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
        assert np.abs(np.dot(forces[0], unit_vec) - np.linalg.norm(forces[0])) < 1e-7

    def test_zero_displacement_zero_force(self) -> None:
        """No displacement produces no force."""
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        springs = np.array([[0, 1]], dtype=np.int32)
        stiffnesses = np.array([100.0])
        rest_lengths = np.array([1.0])
        failed = np.zeros(1, dtype=bool)

        forces = compute_spring_forces(positions, springs, stiffnesses, rest_lengths, failed)
        np.testing.assert_allclose(forces, np.zeros_like(positions))


class TestTensionOnly:
    """Tests for tension-only spring behavior."""

    def test_tension_only_compression(self) -> None:
        """Compression returns zero force for tension-only springs."""
        # Compression by 10%
        positions = np.array([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]])
        springs = np.array([[0, 1]], dtype=np.int32)
        stiffnesses = np.array([100.0])
        rest_lengths = np.array([1.0])
        failed = np.zeros(1, dtype=bool)
        tension_only = np.array([True])

        # Tension-only spring -> should return 0 force
        forces = compute_spring_forces(
            positions, springs, stiffnesses, rest_lengths, failed, tension_only
        )
        np.testing.assert_allclose(forces, np.zeros_like(positions))


class TestStrainComputation:
    """Tests for engineering strain calculation."""

    def test_strain_computation(self) -> None:
        """Strain = (L - L0) / L0."""
        positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        springs = np.array([[0, 1]], dtype=np.int32)
        rest_lengths = np.array([1.0])

        strains = compute_spring_strains(positions, springs, rest_lengths)
        # (2.0 - 1.0) / 1.0 = 1.0 (100% strain)
        assert strains[0] == 1.0
