"""Tests for mathematical parity and consistency of the Taichi GPU explicit solver.

Asserts that running the simulation using the Taichi GPU leapfrog loop produces
mathematically equivalent trajectories compared to the Numba CPU reference solver.
"""

from __future__ import annotations

import contextlib

import dearpygui.dearpygui as dpg
import numpy as np

from kevlargrid.gui.viewport3d import HAS_PYVISTA, Viewport3D
from kevlargrid.solver.fused import fused_leapfrog_loop
from kevlargrid.solver.grid import Grid
from kevlargrid.solver.taichi_solver import taichi_leapfrog_loop


def test_taichi_step_consistency() -> None:
    """Assert that taichi_leapfrog_loop matches fused_leapfrog_loop output."""
    # Create a small 4x4 grid (16 nodes, 1 ply)
    n_nodes = 16
    positions = np.zeros((n_nodes, 3), dtype=np.float64)
    # Simple planar grid at Z=0
    for y in range(4):
        for x in range(4):
            idx = y * 4 + x
            positions[idx] = [x * 0.05, y * 0.05, 0.0]

    velocities = np.zeros((n_nodes, 3), dtype=np.float64)

    # connectivity: spring between adjacent nodes
    springs_list = []
    for y in range(4):
        for x in range(4):
            idx = y * 4 + x
            if x < 3:
                springs_list.append([idx, idx + 1])
            if y < 3:
                springs_list.append([idx, idx + 4])

    grid_springs = np.array(springs_list, dtype=np.int32)
    n_springs = len(grid_springs)
    grid_stiffnesses = np.ones(n_springs, dtype=np.float64) * 1e5
    grid_rest_lengths = np.ones(n_springs, dtype=np.float64) * 0.05
    grid_failed = np.zeros(n_springs, dtype=bool)
    grid_tension_only = np.ones(n_springs, dtype=bool)
    grid_masses = np.ones(n_nodes, dtype=np.float64) * 0.02
    boundary_mask = np.zeros(n_nodes, dtype=bool)
    boundary_mask[[0, 3, 12, 15]] = True  # clamp corners

    # Projectile
    proj_pos = np.array([0.075, 0.075, 0.01], dtype=np.float64)
    proj_vel = np.array([0.0, 0.0, -10.0], dtype=np.float64)
    proj_mass = 0.5
    blade_width = 0.02
    edge_thickness = 0.005

    # Simulation params
    dt = 1e-5
    n_steps = 20
    save_interval = 10
    k_penalty = 1e6
    damping_coeff = 0.1
    failure_strain = 0.05

    # 1. Run CPU JIT fused loop
    (
        pos_cpu,
        vel_cpu,
        failed_cpu,
        proj_pos_cpu,
        proj_vel_cpu,
        damp_cpu,
        t_sim_cpu,
        hist_pos_cpu,
        hist_failed_cpu,
        hist_proj_pos_cpu,
        hist_time_cpu,
        hist_ke_cpu,
        hist_se_cpu,
        hist_proj_ke_cpu,
    ) = fused_leapfrog_loop(
        positions.copy(),
        velocities.copy(),
        grid_springs.copy(),
        grid_stiffnesses.copy(),
        grid_rest_lengths.copy(),
        grid_failed.copy(),
        grid_masses.copy(),
        grid_tension_only.copy(),
        boundary_mask.copy(),
        proj_pos.copy(),
        proj_vel.copy(),
        proj_mass,
        blade_width,
        edge_thickness,
        1,  # n_plies
        16,  # n_nodes_per_layer
        0.002,  # t_ply
        0.05,  # dx
        k_penalty,
        damping_coeff,
        failure_strain,
        dt,
        n_steps,
        save_interval,
        0.0,  # damp_dissipated_init
        0.0,  # t_sim_init
    )

    # 2. Run GPU Taichi loop
    (
        pos_gpu,
        vel_gpu,
        failed_gpu,
        proj_pos_gpu,
        proj_vel_gpu,
        damp_gpu,
        t_sim_gpu,
        hist_pos_gpu,
        hist_failed_gpu,
        hist_proj_pos_gpu,
        hist_time_gpu,
        hist_ke_gpu,
        hist_se_gpu,
        hist_proj_ke_gpu,
    ) = taichi_leapfrog_loop(
        positions.copy(),
        velocities.copy(),
        grid_springs.copy(),
        grid_stiffnesses.copy(),
        grid_rest_lengths.copy(),
        grid_failed.copy(),
        grid_masses.copy(),
        grid_tension_only.copy(),
        boundary_mask.copy(),
        proj_pos.copy(),
        proj_vel.copy(),
        proj_mass,
        blade_width,
        edge_thickness,
        1,  # n_plies
        16,  # n_nodes_per_layer
        0.002,  # t_ply
        0.05,  # dx
        k_penalty,
        damping_coeff,
        failure_strain,
        dt,
        n_steps,
        save_interval,
        0.0,  # damp_dissipated_init
        0.0,  # t_sim_init
    )

    # Assert outputs are mathematically close (allow for float32 precision limits)
    assert np.allclose(pos_cpu, pos_gpu, atol=1e-4, rtol=1e-4)
    assert np.allclose(vel_cpu, vel_gpu, atol=1e-4, rtol=1e-4)
    assert np.array_equal(failed_cpu, failed_gpu)
    assert np.allclose(proj_pos_cpu, proj_pos_gpu, atol=1e-4, rtol=1e-4)
    assert np.allclose(proj_vel_cpu, proj_vel_gpu, atol=1e-4, rtol=1e-4)
    assert np.allclose(damp_cpu, damp_gpu, atol=1e-4, rtol=1e-4)
    assert np.allclose(t_sim_cpu, t_sim_gpu, atol=1e-4, rtol=1e-4)

    # Check history metrics
    assert np.allclose(hist_pos_cpu, hist_pos_gpu, atol=1e-4, rtol=1e-4)
    assert np.array_equal(hist_failed_cpu, hist_failed_gpu)
    assert np.allclose(hist_proj_pos_cpu, hist_proj_pos_gpu, atol=1e-4, rtol=1e-4)
    assert np.allclose(hist_time_cpu, hist_time_gpu, atol=1e-4, rtol=1e-4)
    assert np.allclose(hist_ke_cpu, hist_ke_gpu, atol=1e-4, rtol=1e-4)
    assert np.allclose(hist_se_cpu, hist_se_gpu, atol=1e-4, rtol=1e-4)
    assert np.allclose(hist_proj_ke_cpu, hist_proj_ke_gpu, atol=1e-4, rtol=1e-4)


def test_viewport_pyvista_offscreen() -> None:
    """Verify that Viewport3D resets and builds the offscreen plotter successfully."""
    # Create a simple 4-node grid
    nodes = np.zeros((4, 3))
    springs = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
    grid = Grid(
        nodes=nodes,
        springs=springs,
        masses=np.ones(4) * 0.1,
        stiffnesses=np.ones(3) * 1e4,
        rest_lengths=np.ones(3) * 0.1,
        failed=np.zeros(3, dtype=bool),
        tension_only=np.zeros(3, dtype=bool),
    )

    dpg.create_context()
    try:
        viewport = Viewport3D()
        viewport.reset(grid, n_plies=1, n_nodes_per_layer=4)

        if HAS_PYVISTA:
            assert viewport.has_pyvista is True
            assert viewport.plotter is not None
            assert viewport.mesh is not None
            assert viewport.actor is not None
            assert viewport.proj_actor is not None
    finally:
        with contextlib.suppress(Exception):
            dpg.destroy_context()
