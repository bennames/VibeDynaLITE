from __future__ import annotations

import numpy as np
import pytest

from kevlargrid.solver.energy import (
    compute_energy_balance,
    compute_kinetic_energy,
    compute_strain_energy,
)


class TestEnergy:
    """Tests for kinetic and strain energy tracking."""

    def test_kinetic_energy(self) -> None:
        """Verify kinetic energy calculations match 0.5 * m * v^2 for known values."""
        velocities = np.array([[1.0, 2.0, 3.0], [-2.0, 0.0, 1.0]])
        masses = np.array([2.0, 4.0])

        # Node 0: v^2 = 1 + 4 + 9 = 14 -> KE = 0.5 * 2.0 * 14 = 14.0
        # Node 1: v^2 = 4 + 0 + 1 = 5 -> KE = 0.5 * 4.0 * 5 = 10.0
        # Total KE = 24.0 J
        ke = compute_kinetic_energy(velocities, masses)
        assert np.abs(ke - 24.0) < 1e-7

    def test_strain_energy(self) -> None:
        """Verify strain energy calculations match 0.5 * k * dx^2 for active springs."""
        strains = np.array([0.1, 0.2, 0.05])
        stiffnesses = np.array([100.0, 50.0, 200.0])
        rest_lengths = np.array([1.0, 2.0, 1.0])
        failed = np.array([False, False, True])

        # Spring 0: strain=0.1, k=100, L0=1.0 -> SE = 0.5 * 100 * (0.1 * 1.0)^2 = 0.5 J
        # Spring 1: strain=0.2, k=50, L0=2.0 -> SE = 0.5 * 50 * (0.2 * 2.0)^2 = 0.5 * 50 * 0.16 = 4.0 J
        # Spring 2: failed -> SE = 0.0 J
        # Total SE = 4.5 J
        se = compute_strain_energy(strains, stiffnesses, rest_lengths, failed)
        assert np.abs(se - 4.5) < 1e-7

    def test_energy_balance_closure(self) -> None:
        """Verify that total energy (KE + SE + damped) remains constant or balances."""
        ke = 100.0
        se = 50.0
        damped = 10.0

        eb = compute_energy_balance(ke, se, damped)
        assert eb["kinetic"] == 100.0
        assert eb["strain"] == 50.0
        assert eb["damped"] == 10.0
        assert eb["total"] == 160.0

    def test_layer_energies(self) -> None:
        """Verify that per-layer energies compute correctly and sum to total energy."""
        from kevlargrid.solver.energy import (
            compute_layer_kinetic_energy,
            compute_layer_strain_energy,
        )

        n_nodes_per_layer = 4
        n_plies = 2

        # 1. Kinetic energy per layer
        velocities = np.array(
            [
                [1.0, 2.0, 3.0],  # Ply 0, Node 0: v^2 = 14
                [0.0, 0.0, 0.0],  # Ply 0, Node 1: v^2 = 0
                [0.0, 0.0, 0.0],  # Ply 0, Node 2: v^2 = 0
                [0.0, 0.0, 0.0],  # Ply 0, Node 3: v^2 = 0
                # Ply 1
                [2.0, 0.0, 1.0],  # Ply 1, Node 4: v^2 = 5
                [0.0, 0.0, 0.0],  # Ply 1, Node 5: v^2 = 0
                [0.0, 0.0, 0.0],  # Ply 1, Node 6: v^2 = 0
                [0.0, 0.0, 0.0],  # Ply 1, Node 7: v^2 = 0
            ]
        )
        masses = np.array([2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 4.0])

        ke_layers = compute_layer_kinetic_energy(velocities, masses, n_nodes_per_layer, n_plies)
        # Ply 0: 0.5 * 2.0 * 14 = 14.0 J
        # Ply 1: 0.5 * 4.0 * 5 = 10.0 J
        assert ke_layers[0] == 14.0
        assert ke_layers[1] == 10.0
        assert np.sum(ke_layers) == compute_kinetic_energy(velocities, masses)

        # 2. Strain energy per layer
        # 2 springs: spring 0 belongs to ply 0, spring 1 to ply 1
        springs = np.array([[0, 1], [4, 5]], dtype=np.int32)
        strains = np.array([0.1, 0.2])
        stiffnesses = np.array([100.0, 50.0])
        rest_lengths = np.array([1.0, 2.0])
        failed = np.array([False, False])

        se_layers = compute_layer_strain_energy(
            strains, stiffnesses, rest_lengths, springs, n_nodes_per_layer, n_plies, failed
        )
        # Ply 0: 0.5 * 100 * (0.1 * 1.0)^2 = 0.5 J
        # Ply 1: 0.5 * 50 * (0.2 * 2.0)^2 = 4.0 J
        assert se_layers[0] == pytest.approx(0.5)
        assert se_layers[1] == pytest.approx(4.0)
        assert np.sum(se_layers) == pytest.approx(compute_strain_energy(
            strains, stiffnesses, rest_lengths, failed
        ))
