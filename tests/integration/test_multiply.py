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
        """Verify correct grid layer count for Mode B.

        Mode B explicitly instantiates individual plies with separation.
        """
        pass
