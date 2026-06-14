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

    def test_check_termination(self) -> None:
        """Verify that simulation termination criteria are detected correctly."""
        from kevlargrid.solver.projectile import check_termination

        nx, ny, dx = 5, 5, 0.01
        grid = generate_rectangular_grid(nx, ny, dx, MOCK_MATERIAL)
        positions = grid.nodes.copy()

        # Projectile moving in +Z (initially 50 m/s)
        proj = Projectile(
            mass=0.1,
            velocity=[0.0, 0.0, 50.0],
            position=[0.0, 0.0, -0.005],
            blade_width=0.02,
            edge_thickness=0.005,
        )

        # Active state: not arrested, not timed out, not passed fabric yet
        assert (
            check_termination(
                proj, grid, positions, t_current=0.001, t_max=0.01, initial_velocity_z=50.0
            )
            is None
        )

        # 1. Timeout trigger
        assert (
            check_termination(
                proj, grid, positions, t_current=0.01, t_max=0.01, initial_velocity_z=50.0
            )
            == "timeout"
        )

        # 2. Arrest trigger (velocity Z becomes zero or negative)
        proj.velocity = np.array([0.0, 0.0, -1.0])
        assert (
            check_termination(
                proj, grid, positions, t_current=0.001, t_max=0.01, initial_velocity_z=50.0
            )
            == "arrest"
        )

        # Restore velocity
        proj.velocity = np.array([0.0, 0.0, 30.0])

        # 3. Penetration trigger
        # Projectile moves past Z=0
        proj.position = np.array([0.0, 0.0, 0.005])
        # Mark all springs as failed
        grid.failed[:] = True
        # Setup some contact nodes
        proj.contact_nodes = np.array([12], dtype=np.int32)
        assert (
            check_termination(
                proj, grid, positions, t_current=0.001, t_max=0.01, initial_velocity_z=50.0
            )
            == "penetration"
        )

    def test_generate_impact_report(self) -> None:
        """Verify correct calculation of report metrics in the impact summary."""
        from kevlargrid.solver.projectile import generate_impact_report

        proj = Projectile(
            mass=0.2,
            velocity=[0.0, 0.0, 100.0],
            position=[0.0, 0.0, 0.0],
            blade_width=0.02,
            edge_thickness=0.005,
        )

        initial_ke = 0.5 * proj.mass * 150.0**2  # 2250 J

        # 1. Arrest case
        proj.velocity = np.array([0.0, 0.0, 0.0])
        report = generate_impact_report(proj, initial_ke, "arrest")
        assert report["arrested"] is True
        assert report["penetration"] is False
        assert report["timeout"] is False
        assert report["exit_velocity_m_s"] == 0.0
        assert report["residual_ke_j"] == 0.0
        assert report["energy_absorbed_j"] == initial_ke

        # 2. Penetration case
        proj.velocity = np.array([0.0, 0.0, 40.0])  # exit velocity = 40 m/s
        report = generate_impact_report(proj, initial_ke, "penetration")
        assert report["arrested"] is False
        assert report["penetration"] is True
        assert report["timeout"] is False
        assert report["exit_velocity_m_s"] == 40.0
        assert report["residual_ke_j"] == 0.5 * 0.2 * 40.0**2  # 160 J
        assert report["energy_absorbed_j"] == initial_ke - 160.0

    def test_check_termination_jax_compatibility(self) -> None:
        """Verify check_termination with JAX arrays for failed springs."""
        try:
            import jax.numpy as jnp
        except ImportError:
            return

        from kevlargrid.solver.projectile import check_termination

        nx, ny, dx = 5, 5, 0.01
        grid = generate_rectangular_grid(nx, ny, dx, MOCK_MATERIAL)
        positions = grid.nodes.copy()

        # Projectile past Z=0
        proj = Projectile(
            mass=0.1,
            velocity=[0.0, 0.0, 50.0],
            position=[0.0, 0.0, 0.005],
            blade_width=0.02,
            edge_thickness=0.005,
        )

        # Set contact nodes
        proj.contact_nodes = np.array([12], dtype=np.int32)

        # Simulate JAX device array for failed flags
        grid.failed = jnp.ones(len(grid.failed), dtype=bool)

        res = check_termination(
            proj, grid, positions, t_current=0.001, t_max=0.01, initial_velocity_z=50.0
        )
        assert res == "penetration"

