from __future__ import annotations

import numpy as np

from kevlargrid.solver.grid import generate_rectangular_grid
from kevlargrid.solver.projectile import Projectile, distribute_contact_forces, update_contact_zone

MOCK_MATERIAL = {
    "tensile_modulus_gpa": 71.0,
    "areal_density_kgm2": 0.47,
    "fiber_density_gcc": 1.44,
    "shear_ratio": 0.0004,
}


class TestProjectile:
    """Unit tests for the Projectile class and contact solver routines."""

    def test_projectile_initialization(self) -> None:
        """Verify that the projectile is correctly initialized."""
        proj = Projectile(
            mass=0.5,
            velocity=[0.0, 0.0, 150.0],
            position=[0.0, 0.0, -0.05],
            blade_width=0.05,
            edge_thickness=0.002,
        )
        assert proj.mass == 0.5
        np.testing.assert_allclose(proj.velocity, [0.0, 0.0, 150.0])
        np.testing.assert_allclose(proj.position, [0.0, 0.0, -0.05])
        assert proj.blade_width == 0.05
        assert proj.edge_thickness == 0.002
        assert len(proj.contact_nodes) == 0

    def test_update_contact_zone(self) -> None:
        """Verify contact zone footprint mapping and proximity updates."""
        nx, ny, dx = 11, 11, 0.01
        grid = generate_rectangular_grid(nx, ny, dx, MOCK_MATERIAL)

        # Place projectile centered in X-Y and slightly below the grid Z=0
        proj = Projectile(
            mass=0.5,
            velocity=[0.0, 0.0, 10.0],
            position=[0.0, 0.0, -0.001],
            blade_width=0.03,  # spans 3 nodes in X
            edge_thickness=0.01,  # spans 1 node in Y
        )

        # Nodes at Z=0 should be within proximity threshold of 0.002m
        contact_nodes = update_contact_zone(proj, grid, proximity_threshold=0.002)
        assert len(contact_nodes) > 0

        # Verify that all returned contact nodes have X and Y coords within blade limits
        pos = grid.nodes[contact_nodes]
        assert np.all(np.abs(pos[:, 0]) <= proj.blade_width / 2.0 + 1e-5)
        assert np.all(np.abs(pos[:, 1]) <= proj.edge_thickness / 2.0 + 1e-5)

    def test_distribute_contact_forces_symmetry(self) -> None:
        """Verify that contact force distribution is symmetric and obeys conservation."""
        nx, ny, dx = 5, 5, 0.01
        grid = generate_rectangular_grid(nx, ny, dx, MOCK_MATERIAL)

        # Center projectile, moving in +Z
        proj = Projectile(
            mass=0.5,
            velocity=[0.0, 0.0, 10.0],
            position=[0.0, 0.0, 0.001],  # Penetrating in Z
            blade_width=0.02,
            edge_thickness=0.005,
        )

        positions = grid.nodes.copy()
        # Set node positions symmetrically
        update_contact_zone(proj, grid, proximity_threshold=0.005, positions=positions)
        forces = distribute_contact_forces(proj, grid, positions=positions)

        # Retrieve forces on contact nodes
        c_forces = forces[proj.contact_nodes]
        assert len(c_forces) > 0

        # Since projectile moves in +Z, contact forces on fabric should be in +Z
        assert np.all(c_forces[:, 2] >= 0.0)
        assert np.all(c_forces[:, :2] == 0.0)

        # Find nodes at symmetrical X offsets
        x_coords = positions[proj.contact_nodes, 0]
        # Check that nodes at equal absolute X offsets receive the same force
        for i, x1 in enumerate(x_coords):
            for j, x2 in enumerate(x_coords):
                if np.abs(np.abs(x1) - np.abs(x2)) < 1e-5:
                    assert np.abs(c_forces[i, 2] - c_forces[j, 2]) < 1e-5

    def test_no_contact_zero_force(self) -> None:
        """Verify that a far away projectile yields zero contact nodes and forces."""
        nx, ny, dx = 5, 5, 0.01
        grid = generate_rectangular_grid(nx, ny, dx, MOCK_MATERIAL)

        # Projectile placed far below the grid
        proj = Projectile(
            mass=0.5,
            velocity=[0.0, 0.0, 10.0],
            position=[0.0, 0.0, -1.0],
            blade_width=0.02,
            edge_thickness=0.002,
        )

        contact_nodes = update_contact_zone(proj, grid, proximity_threshold=0.01)
        assert len(contact_nodes) == 0

        forces = distribute_contact_forces(proj, grid)
        assert np.all(forces == 0.0)
