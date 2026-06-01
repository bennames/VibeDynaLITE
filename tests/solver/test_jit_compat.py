"""Tests for verifying strict JIT compiler compatibility (Numba / JAX).

Asserts that key mathematical operations (such as tension-only energy calculation)
compile cleanly under the selected high-performance backends.
"""

from __future__ import annotations

import numpy as np
import pytest

from kevlargrid.solver.energy import compute_layer_strain_energy, compute_strain_energy


def test_strain_energy_jit_compilation() -> None:
    """Assert that compute_strain_energy JIT-compiles cleanly."""
    strains = np.array([0.02, -0.01, 0.05, -0.05, 0.0], dtype=np.float64)
    stiffnesses = np.array([1e6, 1e6, 1e6, 1e6, 1e6], dtype=np.float64)
    rest_lengths = np.array([0.01, 0.01, 0.01, 0.01, 0.01], dtype=np.float64)
    failed = np.array([False, False, False, True, False], dtype=bool)

    # Trigger compilation and run
    res = compute_strain_energy(strains, stiffnesses, rest_lengths, failed)

    # 0.5 * k * (epsilon * L0)^2 for active tensile springs
    # spring 0: 0.5 * 1e6 * (0.02 * 0.01)^2 = 500,000 * 4e-8 = 0.02 J
    # spring 1: compressed (-0.01) -> 0.0 J (tension-only)
    # spring 2: 0.5 * 1e6 * (0.05 * 0.01)^2 = 500,000 * 2.5e-7 = 0.125 J
    # spring 3: failed -> 0.0 J
    # spring 4: zero strain -> 0.0 J
    # Total = 0.02 + 0.125 = 0.145 J
    assert res == pytest.approx(0.145)


def test_layer_strain_energy_jit_compilation() -> None:
    """Assert that compute_layer_strain_energy JIT-compiles cleanly."""
    strains = np.array([0.02, -0.01, 0.05, 0.01], dtype=np.float64)
    stiffnesses = np.array([1e6, 1e6, 1e6, 1e6], dtype=np.float64)
    rest_lengths = np.array([0.01, 0.01, 0.01, 0.01], dtype=np.float64)
    springs = np.array([[0, 1], [1, 2], [3, 4], [4, 5]], dtype=np.int32)
    failed = np.array([False, False, False, False], dtype=bool)

    # 2 plies, 3 nodes per layer
    res = compute_layer_strain_energy(
        strains, stiffnesses, rest_lengths, springs, n_nodes_per_layer=3, n_plies=2, failed=failed
    )

    assert len(res) == 2
    # Layer 0 (springs 0 & 1): spring 0 (0.02 J), spring 1 (compressed -> 0 J) -> 0.02 J
    # Layer 1 (springs 2 & 3): spring 2 (0.125 J), spring 3 (0.5 * 1e6 * (0.01 * 0.01)^2 = 0.005 J) -> 0.13 J
    assert res[0] == pytest.approx(0.02)
    assert res[1] == pytest.approx(0.13)
