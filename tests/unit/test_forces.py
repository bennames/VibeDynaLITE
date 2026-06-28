from __future__ import annotations

import numpy as np
import pytest

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

    def test_interply_contact_forces(self) -> None:
        """Verify inter-ply contact force computation, direction, and energy."""
        from kevlargrid.solver.forces import compute_interply_contact_forces

        n_nodes_per_layer = 4
        n_plies = 2
        t_ply = 0.002
        k_penalty = 1000.0

        # Layer 0 at Z=0, Layer 1 at Z=0.002
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                # Layer 1
                [0.0, 0.0, 0.002],
                [0.0, 0.0, 0.002],
                [0.0, 0.0, 0.002],
                [0.0, 0.0, 0.002],
            ],
            dtype=np.float64,
        )

        # 1. No penetration: separation is exactly t_ply
        forces, energy, _ = compute_interply_contact_forces(
            positions, n_nodes_per_layer, n_plies, t_ply, k_penalty
        )
        assert np.all(forces == 0.0)
        assert energy == 0.0

        # 2. Push node 0 of layer 0 up to Z = 0.0015 -> penetrates corresponding node 4 of layer 1 (Z=0.002)
        # Delta = 0.0015 - 0.002 + 0.002 = 0.0015 m
        positions[0, 2] = 0.0015

        forces, energy, _ = compute_interply_contact_forces(
            positions, n_nodes_per_layer, n_plies, t_ply, k_penalty
        )

        # F = k_penalty * delta = 1000.0 * 0.0015 = 1.5 N
        # Node 0 (layer 0) pushed down (-1.5 N along Z)
        # Node 4 (layer 1) pushed up (+1.5 N along Z)
        assert forces[0, 2] == -1.5
        assert forces[4, 2] == 1.5

        # All other forces should be zero
        assert np.all(forces[1:4] == 0.0)
        assert np.all(forces[5:] == 0.0)

        # Potential energy: 0.5 * k * x^2 = 0.5 * 1000.0 * 0.0015^2 = 0.001125 J
        assert energy == pytest.approx(0.001125)

    def test_interply_contact_forces_friction(self) -> None:
        """Verify inter-ply friction forces under relative velocity."""
        from kevlargrid.solver.forces import compute_interply_contact_forces

        n_nodes_per_layer = 4
        n_plies = 2
        t_ply = 0.002
        k_penalty = 1000.0

        # Layer 0 at Z=0.0015, Layer 1 at Z=0.002 (penetration = 0.0015m)
        positions = np.array(
            [
                [0.0, 0.0, 0.0015],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                # Layer 1
                [0.0, 0.0, 0.002],
                [0.0, 0.0, 0.002],
                [0.0, 0.0, 0.002],
                [0.0, 0.0, 0.002],
            ],
            dtype=np.float64,
        )

        # Velocities: node 0 moving at +10m/s along X, node 4 stationary
        velocities = np.zeros_like(positions)
        velocities[0, 0] = 10.0

        # F_normal = 1000.0 * 0.0015 = 1.5 N
        # Friction with mu_s = 0.2
        # v_rel = 10.0, v0 = 0.01 -> denom = sqrt(100.0 + 0.0001) = 10.000005
        # f_fric = 0.2 * 1.5 * (10 / 10.000005) = 0.299999 N
        forces, energy, fric_diss = compute_interply_contact_forces(
            positions,
            n_nodes_per_layer,
            n_plies,
            t_ply,
            k_penalty,
            velocities=velocities,
            mu_s=0.2,
            dt=1e-6,
        )

        assert forces[0, 0] == pytest.approx(-0.3, abs=1e-3)
        assert forces[4, 0] == pytest.approx(0.3, abs=1e-3)
        assert fric_diss == pytest.approx(0.3 * 10.0 * 1e-6, abs=1e-6)
