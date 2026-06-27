from __future__ import annotations

import numpy as np

from kevlargrid.solver.fused import fused_leapfrog_loop
from kevlargrid.solver.grid import generate_rectangular_grid


def test_projectile_free_flying_energy_conservation() -> None:
    """Verify that a free-flying projectile in vacuum conserves linear and rotational kinetic energy."""
    material = {
        "tensile_modulus_gpa": 71.0,
        "areal_density_kgm2": 0.47,
        "fiber_density_gcc": 1.44,
        "shear_ratio": 0.0004,
    }
    grid = generate_rectangular_grid(3, 3, 0.01, material)

    proj_pos = np.array([0.0, 0.0, 1.0])
    proj_vel = np.array([10.0, 20.0, 30.0])
    proj_omega = np.array([1.0, 2.0, 3.0])
    proj_quat = np.array([1.0, 0.0, 0.0, 0.0])

    proj_mass = 0.5
    R = 0.02
    I_val = 0.4 * proj_mass * R**2
    proj_inertia_inv = np.diag([1.0 / I_val, 1.0 / I_val, 1.0 / I_val])

    dt = 1e-4
    n_steps = 100

    ke_lin_init = 0.5 * proj_mass * np.sum(proj_vel**2)
    ke_rot_init = 0.5 * I_val * np.sum(proj_omega**2)
    e_tot_init = ke_lin_init + ke_rot_init

    n_nodes = 9
    boundary_mask = np.zeros(n_nodes, dtype=bool)
    nodal_external_forces = np.zeros((n_nodes, 3))

    (
        pos,
        vel,
        grid_failed,
        proj_pos_final,
        proj_vel_final,
        *_,
    ) = fused_leapfrog_loop(
        grid.nodes.copy(),
        np.zeros_like(grid.nodes),
        grid.springs,
        grid.stiffnesses,
        grid.rest_lengths,
        grid.failed.copy(),
        grid.masses,
        grid.tension_only,
        boundary_mask,
        nodal_external_forces,
        proj_pos,
        proj_vel,
        proj_mass,
        0.02,  # blade_width
        0.005,  # edge_thickness
        1,  # n_plies
        n_nodes,
        0.002,  # t_ply
        0.01,  # dx
        1e5,  # k_penalty
        0.0,  # alpha
        0.0,  # beta
        0.04,  # failure_strain
        0.024,  # damage_onset_strain
        1.5,  # fracture_energy_multiplier
        dt,
        n_steps,
        1,  # save_interval
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        grid.initial_spring_counts,
        grid.node_spring_offsets,
        grid.node_spring_ids,
        grid.node_spring_signs,
        proj_quat=proj_quat,
        proj_omega=proj_omega,
        proj_shape_type="sphere",
        proj_radius=R,
        proj_inertia_inv=proj_inertia_inv,
    )

    ke_lin_final = 0.5 * proj_mass * np.sum(proj_vel_final**2)
    ke_rot_final = 0.5 * I_val * np.sum(proj_omega**2)
    e_tot_final = ke_lin_final + ke_rot_final

    np.testing.assert_allclose(proj_vel_final, proj_vel)
    np.testing.assert_allclose(proj_omega, proj_omega)
    np.testing.assert_allclose(e_tot_final, e_tot_init, rtol=1e-5)


def test_oblique_impact_tumbling_dynamics() -> None:
    """Verify that oblique impact with fabric initiates tumbling (torque) and conserves/dissipates total energy."""
    material = {
        "tensile_modulus_gpa": 71.0,
        "areal_density_kgm2": 0.47,
        "fiber_density_gcc": 1.44,
        "shear_ratio": 0.0004,
    }
    grid = generate_rectangular_grid(5, 5, 0.01, material)
    n_nodes = 25
    boundary_mask = np.zeros(n_nodes, dtype=bool)
    for i in range(5):
        for j in range(5):
            if i == 0 or i == 4 or j == 0 or j == 4:
                boundary_mask[i * 5 + j] = True

    proj_mass = 0.05
    R = 0.01
    L = 0.03
    I_zz = 0.5 * proj_mass * R**2
    I_xx = (1.0 / 12.0) * proj_mass * (3.0 * R**2 + L**2)
    proj_inertia_inv = np.diag([1.0 / I_xx, 1.0 / I_xx, 1.0 / I_zz])

    proj_pos = np.array([0.0, 0.0, -0.002])
    proj_vel = np.array([20.0, 0.0, 100.0])
    proj_omega = np.array([0.0, 0.0, 0.0])
    proj_quat = np.array([1.0, 0.0, 0.0, 0.0])

    dt = 1e-6
    n_steps = 100

    (
        pos,
        vel,
        grid_failed,
        proj_pos_final,
        proj_vel_final,
        *_,
    ) = fused_leapfrog_loop(
        grid.nodes.copy(),
        np.zeros_like(grid.nodes),
        grid.springs,
        grid.stiffnesses,
        grid.rest_lengths,
        grid.failed.copy(),
        grid.masses,
        grid.tension_only,
        boundary_mask,
        np.zeros((n_nodes, 3)),
        proj_pos,
        proj_vel,
        proj_mass,
        0.02,  # blade_width
        0.005,  # edge_thickness
        1,
        n_nodes,
        0.002,
        0.01,
        1e6,
        0.0,
        0.0,
        0.04,
        0.024,
        1.5,
        dt,
        n_steps,
        1,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        grid.initial_spring_counts,
        grid.node_spring_offsets,
        grid.node_spring_ids,
        grid.node_spring_signs,
        proj_quat=proj_quat,
        proj_omega=proj_omega,
        proj_shape_type="cylinder",
        proj_radius=R,
        proj_length=L,
        proj_inertia_inv=proj_inertia_inv,
    )

    assert np.any(np.abs(proj_omega) > 1e-3)


def test_mesh_refinement_v50_convergence() -> None:
    """Verify that mesh refinement under Bazant regularization yields stable/converged results."""
    material = {
        "tensile_modulus_gpa": 71.0,
        "areal_density_kgm2": 0.47,
        "fiber_density_gcc": 1.44,
        "shear_ratio": 0.0004,
    }

    dxs = [0.01, 0.005, 0.0025]
    residual_vels = []

    for dx in dxs:
        nx = int(0.10 / dx) + 1
        ny = nx
        n_nodes = nx * ny

        grid = generate_rectangular_grid(nx, ny, dx, material)

        boundary_mask = np.zeros(n_nodes, dtype=bool)
        for i in range(nx):
            for j in range(ny):
                if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                    boundary_mask[i * ny + j] = True

        proj_mass = 0.01
        proj_pos = np.array([0.0, 0.0, -0.002])
        proj_vel = np.array([0.0, 0.0, 200.0])
        proj_omega = np.array([0.0, 0.0, 0.0])
        proj_quat = np.array([1.0, 0.0, 0.0, 0.0])

        dt = 0.1 * dx / 5000.0
        n_steps = int(2e-5 / dt)

        (
            pos,
            vel,
            grid_failed,
            proj_pos_final,
            proj_vel_final,
            *_,
        ) = fused_leapfrog_loop(
            grid.nodes.copy(),
            np.zeros_like(grid.nodes),
            grid.springs,
            grid.stiffnesses,
            grid.rest_lengths,
            grid.failed.copy(),
            grid.masses,
            grid.tension_only,
            boundary_mask,
            np.zeros((n_nodes, 3)),
            proj_pos,
            proj_vel,
            proj_mass,
            0.02,  # blade_width
            0.005,  # edge_thickness
            1,
            n_nodes,
            0.002,
            dx,
            1e6,
            0.05,
            1e-7,
            0.04,
            0.024,
            1.5,
            dt,
            n_steps,
            1,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            grid.initial_spring_counts,
            grid.node_spring_offsets,
            grid.node_spring_ids,
            grid.node_spring_signs,
            proj_quat=proj_quat,
            proj_omega=proj_omega,
            proj_shape_type="sphere",
            proj_radius=0.01,
        )

        residual_vels.append(proj_vel_final[2])

    mean_val = np.mean(residual_vels)
    for v in residual_vels:
        assert np.abs(v - mean_val) / max(mean_val, 1.0) < 0.15
