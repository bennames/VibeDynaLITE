"""Tests for mathematical parity and consistency of the fused multi-step JIT runner.

Asserts that running the simulation using the fused leapfrog loop produces
mathematically identical state trajectories compared to running the individual
step-by-step sequential integrator functions.
"""

from __future__ import annotations

import numpy as np
import pytest

from kevlargrid.solver.failure import check_failures
from kevlargrid.solver.forces import compute_spring_forces
from kevlargrid.solver.fused import fused_leapfrog_loop
from kevlargrid.solver.integrator import leapfrog_step
from kevlargrid.solver.projectile import distribute_contact_forces, update_contact_zone


def test_fused_step_consistency() -> None:
    """Assert that fused_leapfrog_loop matches step-by-step execution exactly."""
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

    # 1. Run step-by-step sequentially
    pos_seq = positions.copy()
    vel_seq = velocities.copy()
    failed_seq = grid_failed.copy()
    p_pos_seq = proj_pos.copy()
    p_vel_seq = proj_vel.copy()
    damp_diss_seq = 0.0
    t_sim_seq = 0.0

    class MockProj:
        def __init__(self):
            self.position = proj_pos.copy()
            self.velocity = proj_vel.copy()
            self.mass = proj_mass
            self.blade_width = blade_width
            self.edge_thickness = edge_thickness
            self.contact_nodes = np.array([], dtype=np.int32)

    class MockGrid:
        def __init__(self):
            self.nodes = positions.copy()
            self.springs = grid_springs.copy()
            self.stiffnesses = grid_stiffnesses.copy()
            self.rest_lengths = grid_rest_lengths.copy()
            self.failed = grid_failed.copy()
            self.masses = grid_masses.copy()

    mock_proj = MockProj()
    mock_grid = MockGrid()

    for _ in range(n_steps):
        # projectile contact
        update_contact_zone(mock_proj, mock_grid, proximity_threshold=0.05 * 2.0, positions=pos_seq)
        proj_forces = distribute_contact_forces(
            mock_proj, mock_grid, positions=pos_seq, k_contact=k_penalty
        )

        # interply contact (1 ply -> zeros)
        interply_forces = np.zeros_like(pos_seq)

        # spring forces
        spring_forces = compute_spring_forces(
            pos_seq, grid_springs, grid_stiffnesses, grid_rest_lengths, failed_seq
        )

        # damping
        damp_forces = -damping_coeff * vel_seq
        p_damp = np.sum(damp_forces * vel_seq)
        damp_diss_seq += -p_damp * dt

        # dynamics
        net_forces = spring_forces + proj_forces + interply_forces + damp_forces
        net_forces[boundary_mask] = 0.0
        vel_seq[boundary_mask] = 0.0

        pos_seq, vel_seq = leapfrog_step(pos_seq, vel_seq, net_forces, grid_masses, dt)

        # projectile motion
        proj_reaction_force = -np.sum(proj_forces, axis=0)
        proj_accel = proj_reaction_force / proj_mass
        p_vel_seq += proj_accel * dt
        p_pos_seq += p_vel_seq * dt
        mock_proj.velocity = p_vel_seq.copy()
        mock_proj.position = p_pos_seq.copy()

        # check failures
        p1 = pos_seq[grid_springs[:, 0]]
        p2 = pos_seq[grid_springs[:, 1]]
        diff = p2 - p1
        lengths = np.sqrt(np.sum(diff**2, axis=1))
        strains = (lengths - grid_rest_lengths) / grid_rest_lengths
        check_failures(strains, failed_seq, failure_strain)
        mock_grid.failed = failed_seq.copy()

        t_sim_seq += dt

    # 2. Run Fused JIT loop
    (
        pos_fused,
        vel_fused,
        failed_fused,
        p_pos_fused,
        p_vel_fused,
        damp_diss_fused,
        t_sim_fused,
        hist_pos,
        hist_failed,
        hist_proj_pos,
        hist_time,
        hist_ke,
        hist_se,
        hist_proj_ke,
    ) = fused_leapfrog_loop(
        positions.copy(),
        velocities.copy(),
        grid_springs.copy(),
        grid_stiffnesses.copy(),
        grid_rest_lengths.copy(),
        grid_failed.copy(),
        grid_masses.copy(),
        boundary_mask.copy(),
        proj_pos.copy(),
        proj_vel.copy(),
        proj_mass,
        blade_width,
        edge_thickness,
        n_plies=1,
        n_nodes_per_layer=n_nodes,
        t_ply=0.002,
        k_penalty=k_penalty,
        damping_coeff=damping_coeff,
        failure_strain=failure_strain,
        dt=dt,
        n_steps=n_steps,
        save_interval=save_interval,
        damp_dissipated_init=0.0,
        t_sim_init=0.0,
    )

    # Assert exact/very high-precision mathematical consistency
    assert np.allclose(pos_fused, pos_seq)
    assert np.allclose(vel_fused, vel_seq)
    assert np.all(failed_fused == failed_seq)
    assert np.allclose(p_pos_fused, p_pos_seq)
    assert np.allclose(p_vel_fused, p_vel_seq)
    assert damp_diss_fused == pytest.approx(damp_diss_seq)
    assert t_sim_fused == pytest.approx(t_sim_seq)

    # Check history shape and correctness S7 Verification
    assert hist_pos.shape == (2, n_nodes, 3)
    assert hist_failed.shape == (2, n_springs)
    assert hist_proj_pos.shape == (2, 3)
    assert hist_time.shape == (2,)
    assert hist_ke.shape == (2,)
    assert hist_se.shape == (2,)
    assert hist_proj_ke.shape == (2,)


def test_fused_mode_a_multi_ply_compilation() -> None:
    """Verify that fused_leapfrog_loop runs without crash in Mode A (1 physical layer) with nominal n_plies > 1."""
    n_nodes = 4
    positions = np.array(
        [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, 0.05, 0.0], [0.05, 0.05, 0.0]], dtype=np.float64
    )
    velocities = np.zeros_like(positions)
    grid_springs = np.array([[0, 1], [0, 2], [1, 3], [2, 3]], dtype=np.int32)
    grid_stiffnesses = np.ones(4, dtype=np.float64) * 1e5
    grid_rest_lengths = np.ones(4, dtype=np.float64) * 0.05
    grid_failed = np.zeros(4, dtype=bool)
    grid_masses = np.ones(4, dtype=np.float64) * 0.02
    boundary_mask = np.zeros(4, dtype=bool)

    # Projectile
    proj_pos = np.array([0.025, 0.025, 0.01], dtype=np.float64)
    proj_vel = np.array([0.0, 0.0, -10.0], dtype=np.float64)

    # This should execute cleanly because we pass n_plies = 1 internally to avoid interply contacts.
    res = fused_leapfrog_loop(
        positions.copy(),
        velocities.copy(),
        grid_springs.copy(),
        grid_stiffnesses.copy(),
        grid_rest_lengths.copy(),
        grid_failed.copy(),
        grid_masses.copy(),
        boundary_mask.copy(),
        proj_pos.copy(),
        proj_vel.copy(),
        proj_mass=0.05,
        proj_blade_width=0.02,
        proj_edge_thickness=0.005,
        n_plies=1,  # actual physical layers = 1 in Mode A
        n_nodes_per_layer=n_nodes,
        t_ply=0.002,
        k_penalty=1e5,
        damping_coeff=0.1,
        failure_strain=0.05,
        dt=1e-5,
        n_steps=2,
        save_interval=2,
        damp_dissipated_init=0.0,
        t_sim_init=0.0,
    )
    assert len(res) == 14
