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

    def test_run_solver_process_tangency(self) -> None:
        """Verify that run_solver_process auto-adjusts projectile positions to ensure tangency."""
        import multiprocessing

        from kevlargrid.solver.worker import run_solver_process

        config = {
            "material": {
                "name": "Kevlar 29",
                "tensile_modulus_gpa": 71.0,
                "failure_strain": 0.036,
                "tensile_strength_gpa": 2.92,
                "fiber_density_gcc": 1.44,
                "areal_density_kgm2": 0.47,
                "shear_ratio": 0.0004,
            },
            "grid": {
                "nx": 5,
                "ny": 5,
                "dx": 0.01,
                "n_plies": 1,
                "t_ply": None,
                "boundary_type": "fixed",
            },
            "projectile": {
                "mass": 0.05,
                "velocity": [0.0, 0.0, 400.0],
                "position": [0.0, 0.0, 0.005],  # Overlaps grid (z=0)
                "shape_type": "sphere",
                "radius": 0.01,
            },
            "simulation": {
                "duration": 1e-6,
                "cfl_factor": 0.8,
                "damping_model": "rayleigh",
                "snapshot_interval": 1,
                "backend": "numba",
            },
        }

        ctx = multiprocessing.get_context("spawn")
        mock_queue = ctx.Queue()
        parent_conn, child_conn = ctx.Pipe()
        parent_conn.send("stop")

        proc = ctx.Process(target=run_solver_process, args=(config, mock_queue, child_conn))
        proc.start()
        proc.join(timeout=10)

        init_msg = None
        while not mock_queue.empty():
            msg = mock_queue.get()
            if msg.get("type") == "init":
                init_msg = msg
                break

        assert init_msg is not None
        # Auto-adjusted position for a sphere of R=0.01 striking from below must be z <= -0.01
        np.testing.assert_allclose(init_msg["projectile_pos"], [0.0, 0.0, -0.01])

    def test_numba_solver_execution(self) -> None:
        """Verify that the Numba backend runs and JIT compiles without errors."""
        import multiprocessing

        from kevlargrid.solver.worker import run_solver_process

        config = {
            "material": {
                "name": "Kevlar 29",
                "tensile_modulus_gpa": 71.0,
                "failure_strain": 0.036,
                "tensile_strength_gpa": 2.92,
                "fiber_density_gcc": 1.44,
                "areal_density_kgm2": 0.47,
                "shear_ratio": 0.0004,
            },
            "grid": {
                "nx": 5,
                "ny": 5,
                "dx": 0.01,
                "n_plies": 1,
                "t_ply": None,
                "boundary_type": "fixed",
            },
            "projectile": {
                "mass": 0.05,
                "velocity": [0.0, 0.0, 400.0],
                "position": [0.0, 0.0, -0.015],
                "shape_type": "sphere",
                "radius": 0.01,
            },
            "simulation": {
                "duration": 1.5e-6,
                "dt": 3e-7,
                "cfl_factor": 0.8,
                "damping_model": "rayleigh",
                "snapshot_interval": 1,
                "backend": "numba",
            },
        }

        ctx = multiprocessing.get_context("spawn")
        mock_queue = ctx.Queue()
        parent_conn, child_conn = ctx.Pipe()

        proc = ctx.Process(target=run_solver_process, args=(config, mock_queue, child_conn))
        proc.start()
        proc.join(timeout=60)

        messages = []
        while not mock_queue.empty():
            messages.append(mock_queue.get())

        types = [m.get("type") for m in messages]

        errors = [m for m in messages if m.get("type") == "error"]
        if errors:
            print("\nSOLVER SUBPROCESS ERROR DETECTED:")
            print("Message:", errors[0].get("message"))
            print("Traceback:")
            print(errors[0].get("traceback"))

        assert "init" in types
        assert "completed" in types
        assert "error" not in types
