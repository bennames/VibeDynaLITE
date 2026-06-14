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
    rayleigh_alpha = 0.1
    rayleigh_beta = 0.001
    failure_strain = 0.05
    damage_onset_strain = 0.03
    fracture_energy_multiplier = 1.5

    # 1. Run step-by-step sequentially
    pos_seq = positions.copy()
    vel_seq = velocities.copy()
    failed_seq = grid_failed.copy()
    p_pos_seq = proj_pos.copy()
    p_vel_seq = proj_vel.copy()
    damp_diss_seq = 0.0
    failure_diss_seq = 0.0
    clamp_diss_seq = 0.0
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
    node_initial_springs = np.zeros(n_nodes, dtype=np.int32)
    np.add.at(node_initial_springs, grid_springs[:, 0], 1)
    np.add.at(node_initial_springs, grid_springs[:, 1], 1)

    # Build CSR adjacency for JIT loop (from grid.py)
    node_counts = np.zeros(n_nodes, dtype=np.int32)
    np.add.at(node_counts, grid_springs[:, 0], 1)
    np.add.at(node_counts, grid_springs[:, 1], 1)
    node_spring_offsets = np.zeros(n_nodes + 1, dtype=np.int32)
    node_spring_offsets[1:] = np.cumsum(node_counts)
    current_offset = node_spring_offsets[:-1].copy()
    node_spring_ids = np.zeros(2 * n_springs, dtype=np.int32)
    node_spring_signs = np.zeros(2 * n_springs, dtype=np.float64)
    for j in range(n_springs):
        n0 = grid_springs[j, 0]
        n1 = grid_springs[j, 1]
        offset_0 = current_offset[n0]
        node_spring_ids[offset_0] = j
        node_spring_signs[offset_0] = 1.0
        current_offset[n0] += 1
        offset_1 = current_offset[n1]
        node_spring_ids[offset_1] = j
        node_spring_signs[offset_1] = -1.0
        current_offset[n1] += 1

    dx_param = 0.05
    for _ in range(n_steps):
        # 0. Irreversible failure updates FIRST
        p1 = pos_seq[grid_springs[:, 0]]
        p2 = pos_seq[grid_springs[:, 1]]
        diff = p2 - p1
        lengths = np.sqrt(np.sum(diff**2, axis=1))
        strains = (lengths - grid_rest_lengths) / grid_rest_lengths
        newly_failed = (~failed_seq) & (strains > failure_strain)
        
        # Track fracture energy
        fracture_se = np.where(newly_failed, 0.5 * grid_stiffnesses * (strains * grid_rest_lengths)**2, 0.0)
        failure_diss_seq += np.sum(fracture_se) * fracture_energy_multiplier
        failed_seq = failed_seq | newly_failed
        mock_grid.failed = failed_seq.copy()

        # projectile contact
        update_contact_zone(mock_proj, mock_grid, proximity_threshold=0.05 * 2.0, positions=pos_seq)
        proj_forces = distribute_contact_forces(
            mock_proj, mock_grid, positions=pos_seq, k_contact=k_penalty
        )
        
        # Scale the contact force by the fraction of remaining active springs S7.6.1
        active_springs = np.where(failed_seq, 0, 1)
        active_counts = np.zeros(n_nodes, dtype=np.int32)
        np.add.at(active_counts, grid_springs[:, 0], active_springs)
        np.add.at(active_counts, grid_springs[:, 1], active_springs)
        scale_factor = np.where(node_initial_springs > 0, active_counts / node_initial_springs, 0.0)
        proj_forces = proj_forces * scale_factor[:, np.newaxis]

        # interply contact (1 ply -> zeros)
        interply_forces = np.zeros_like(pos_seq)

        # spring forces with progressive damage model
        spring_forces = compute_spring_forces(
            pos_seq,
            grid_springs,
            grid_stiffnesses,
            grid_rest_lengths,
            failed_seq,
            tension_only=grid_tension_only,
            damage_onset_strain=damage_onset_strain,
            failure_strain=failure_strain,
        )

        # Rayleigh damping forces
        f_mass_damp = -rayleigh_alpha * grid_masses[:, np.newaxis] * vel_seq
        
        lengths_safe = np.where(lengths == 0.0, 1.0, lengths)
        v1 = vel_seq[grid_springs[:, 0]]
        v2 = vel_seq[grid_springs[:, 1]]
        v_rel = v2 - v1
        unit_axes = diff / lengths_safe[:, np.newaxis]
        v_proj = np.sum(v_rel * unit_axes, axis=1)
        stiff_damp_mag = rayleigh_beta * grid_stiffnesses * v_proj
        stiff_damp_mag = np.where(failed_seq, 0.0, stiff_damp_mag)
        stiff_damp_vecs = stiff_damp_mag[:, np.newaxis] * unit_axes
        
        f_stiff_damp = np.zeros_like(vel_seq)
        np.add.at(f_stiff_damp, grid_springs[:, 0], stiff_damp_vecs)
        np.add.at(f_stiff_damp, grid_springs[:, 1], -stiff_damp_vecs)
        
        damp_forces = f_mass_damp + f_stiff_damp
        
        # Dissipated energy power
        p_mass_damp = np.sum(f_mass_damp * vel_seq)
        stiff_damp_power = np.sum(stiff_damp_mag * v_proj)
        damp_diss_seq += (-p_mass_damp + stiff_damp_power) * dt

        # net forces
        net_forces = spring_forces + proj_forces + interply_forces + damp_forces
        net_forces[boundary_mask] = 0.0
        vel_seq[boundary_mask] = 0.0

        pos_seq, vel_seq = leapfrog_step(pos_seq, vel_seq, net_forces, grid_masses, dt)

        # CFL velocity clamping
        v_mag = np.sqrt(np.sum(vel_seq**2, axis=1))
        v_max = dx_param / dt
        scale = np.where(v_mag > v_max, v_max / np.maximum(v_mag, 1e-30), 1.0)
        excess_ke = np.sum(0.5 * grid_masses * (v_mag**2) * (1.0 - scale**2))
        clamp_diss_seq += excess_ke
        vel_seq = vel_seq * scale[:, np.newaxis]
        
        vel_seq[boundary_mask] = 0.0

        # projectile motion
        proj_reaction_force = -np.sum(proj_forces, axis=0)
        proj_accel = proj_reaction_force / proj_mass
        p_vel_seq += proj_accel * dt
        p_pos_seq += p_vel_seq * dt
        mock_proj.velocity = p_vel_seq.copy()
        mock_proj.position = p_pos_seq.copy()

        t_sim_seq += dt

    # 2. Run Fused JIT loop
    (
        pos_fused,
        vel_fused,
        failed_fused,
        p_pos_fused,
        p_vel_fused,
        damp_diss_fused,
        failure_diss_fused,
        clamp_diss_fused,
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
        grid_tension_only.copy(),
        boundary_mask.copy(),
        np.zeros((n_nodes, 3)),
        proj_pos.copy(),
        proj_vel.copy(),
        proj_mass,
        blade_width,
        edge_thickness,
        n_plies=1,
        n_nodes_per_layer=n_nodes,
        t_ply=0.002,
        dx=dx_param,
        k_penalty=k_penalty,
        rayleigh_alpha=rayleigh_alpha,
        rayleigh_beta=rayleigh_beta,
        failure_strain=failure_strain,
        damage_onset_strain=damage_onset_strain,
        fracture_energy_multiplier=fracture_energy_multiplier,
        dt=dt,
        n_steps=n_steps,
        save_interval=save_interval,
        damp_dissipated_init=0.0,
        failure_dissipated_init=0.0,
        clamp_dissipated_init=0.0,
        t_sim_init=0.0,
        strike_direction=0.0,
        node_initial_springs=node_initial_springs,
        node_spring_offsets=node_spring_offsets,
        node_spring_ids=node_spring_ids,
        node_spring_signs=node_spring_signs,
    )

    # Assert exact/very high-precision mathematical consistency
    assert np.allclose(pos_fused, pos_seq)
    assert np.allclose(vel_fused, vel_seq)
    assert np.all(failed_fused == failed_seq)
    assert np.allclose(p_pos_fused, p_pos_seq)
    assert np.allclose(p_vel_fused, p_vel_seq)
    assert damp_diss_fused == pytest.approx(damp_diss_seq)
    assert failure_diss_fused == pytest.approx(failure_diss_seq)
    assert clamp_diss_fused == pytest.approx(clamp_diss_seq)
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
    from kevlargrid.solver.grid import Grid
    n_nodes = 4
    positions = np.array(
        [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, 0.05, 0.0], [0.05, 0.05, 0.0]], dtype=np.float64
    )
    velocities = np.zeros_like(positions)
    grid_springs = np.array([[0, 1], [0, 2], [1, 3], [2, 3]], dtype=np.int32)
    grid_stiffnesses = np.ones(4, dtype=np.float64) * 1e5
    grid_rest_lengths = np.ones(4, dtype=np.float64) * 0.05
    grid_failed = np.zeros(4, dtype=bool)
    grid_tension_only = np.ones(4, dtype=bool)
    grid_masses = np.ones(4, dtype=np.float64) * 0.02
    boundary_mask = np.zeros(4, dtype=bool)

    # Projectile
    proj_pos = np.array([0.025, 0.025, 0.01], dtype=np.float64)
    proj_vel = np.array([0.0, 0.0, -10.0], dtype=np.float64)

    node_initial_springs = np.zeros(n_nodes, dtype=np.int32)
    np.add.at(node_initial_springs, grid_springs[:, 0], 1)
    np.add.at(node_initial_springs, grid_springs[:, 1], 1)

    dummy_grid = Grid(
        nodes=positions,
        springs=grid_springs,
        masses=grid_masses,
        stiffnesses=grid_stiffnesses,
        rest_lengths=grid_rest_lengths,
        failed=grid_failed,
        tension_only=grid_tension_only,
    )

    # This should execute cleanly because we pass n_plies = 1 internally to avoid interply contacts.
    res = fused_leapfrog_loop(
        positions.copy(),
        velocities.copy(),
        grid_springs.copy(),
        grid_stiffnesses.copy(),
        grid_rest_lengths.copy(),
        grid_failed.copy(),
        grid_masses.copy(),
        grid_tension_only.copy(),
        boundary_mask.copy(),
        np.zeros((n_nodes, 3)),
        proj_pos.copy(),
        proj_vel.copy(),
        proj_mass=0.05,
        proj_blade_width=0.02,
        proj_edge_thickness=0.005,
        n_plies=1,  # actual physical layers = 1 in Mode A
        n_nodes_per_layer=n_nodes,
        t_ply=0.002,
        dx=0.05,
        k_penalty=1e5,
        rayleigh_alpha=0.1,
        rayleigh_beta=0.0001,
        failure_strain=0.05,
        damage_onset_strain=0.03,
        fracture_energy_multiplier=1.5,
        dt=1e-5,
        n_steps=2,
        save_interval=2,
        damp_dissipated_init=0.0,
        failure_dissipated_init=0.0,
        clamp_dissipated_init=0.0,
        t_sim_init=0.0,
        strike_direction=0.0,
        node_initial_springs=node_initial_springs,
        node_spring_offsets=dummy_grid.node_spring_offsets,
        node_spring_ids=dummy_grid.node_spring_ids,
        node_spring_signs=dummy_grid.node_spring_signs,
    )
    assert len(res) == 16


def test_fused_mode_b_multiply_parity() -> None:
    """Verify that fused_leapfrog_loop matches step-by-step sequential path in Mode B (multi-ply)."""
    from kevlargrid.solver.grid import generate_rectangular_grid

    # Create a 6x6 grid, 3 plies, t_ply=0.002, dx=0.01
    nx, ny = 6, 6
    dx = 0.01
    n_plies = 3
    t_ply = 0.002

    mat = {
        "density": 1440.0,
        "k_ortho": 1e6,
        "k_shear": 2e5,
        "failure_strain": 0.05,
    }

    grid = generate_rectangular_grid(
        nx=nx, ny=ny, dx=dx, material=mat, n_plies=n_plies, t_ply=t_ply
    )

    n_nodes = grid.n_nodes
    n_springs = len(grid.springs)

    positions = grid.nodes.copy()
    velocities = np.zeros_like(positions)
    grid_failed = np.zeros(n_springs, dtype=bool)

    # Boundary mask: clamp edges of all layers
    boundary_mask = np.zeros(n_nodes, dtype=bool)
    n_nodes_per_layer = nx * ny
    for ply in range(n_plies):
        offset = ply * n_nodes_per_layer
        for i in range(nx):
            for j in range(ny):
                if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                    boundary_mask[offset + i * ny + j] = True

    # Projectile setup near center of top layer, moving downward
    proj_pos = np.array([0.025, 0.025, 0.006], dtype=np.float64)
    proj_vel = np.array([0.0, 0.0, -10.0], dtype=np.float64)
    proj_mass = 0.05
    blade_width = 0.02
    edge_thickness = 0.005
    k_penalty = 1e6
    damping_coeff = 0.1
    failure_strain = 0.05

    dt = 1e-6
    n_steps = 10
    save_interval = 5

    # Run Fused
    (
        pos_fused,
        vel_fused,
        failed_fused,
        p_pos_fused,
        p_vel_fused,
        damp_diss_fused,
        failure_diss_fused,
        clamp_diss_fused,
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
        grid.springs.copy(),
        grid.stiffnesses.copy(),
        grid.rest_lengths.copy(),
        grid_failed.copy(),
        grid.masses.copy(),
        grid.tension_only.copy(),
        boundary_mask.copy(),
        np.zeros((n_nodes, 3)),
        proj_pos.copy(),
        proj_vel.copy(),
        proj_mass,
        blade_width,
        edge_thickness,
        n_plies=n_plies,
        n_nodes_per_layer=n_nodes_per_layer,
        t_ply=t_ply,
        dx=dx,
        k_penalty=k_penalty,
        rayleigh_alpha=damping_coeff,
        rayleigh_beta=0.0001,
        failure_strain=failure_strain,
        damage_onset_strain=0.03,
        fracture_energy_multiplier=1.5,
        dt=dt,
        n_steps=n_steps,
        save_interval=save_interval,
        damp_dissipated_init=0.0,
        failure_dissipated_init=0.0,
        clamp_dissipated_init=0.0,
        t_sim_init=0.0,
        strike_direction=0.0,
        node_initial_springs=grid.initial_spring_counts,
        node_spring_offsets=grid.node_spring_offsets,
        node_spring_ids=grid.node_spring_ids,
        node_spring_signs=grid.node_spring_signs,
    )

    # Assert return shapes are correct
    assert pos_fused.shape == positions.shape
    assert vel_fused.shape == velocities.shape
    assert len(failed_fused) == n_springs
    assert p_pos_fused.shape == (3,)
    assert p_vel_fused.shape == (3,)
    assert isinstance(damp_diss_fused, float)
    assert t_sim_fused == pytest.approx(10 * dt)
    assert hist_pos.shape == (2, n_nodes, 3)
    assert hist_failed.shape == (2, n_springs)
    assert hist_proj_pos.shape == (2, 3)
    assert hist_time.shape == (2,)
    assert hist_ke.shape == (2,)
    assert hist_se.shape == (2,)
    assert hist_proj_ke.shape == (2,)


def test_fused_contact_force_nonzero() -> None:
    """Verify that a projectile hitting a dx=0.01 grid produces clear decelerating reaction force."""
    from kevlargrid.solver.grid import generate_rectangular_grid

    nx, ny = 10, 10
    dx = 0.01

    mat = {
        "density": 1440.0,
        "k_ortho": 1e6,
        "k_shear": 2e5,
        "failure_strain": 0.05,
    }

    grid = generate_rectangular_grid(nx=nx, ny=ny, dx=dx, material=mat)

    positions = grid.nodes.copy()
    # Position projectile exactly in contact with grid at center
    proj_pos = np.array([0.045, 0.045, 0.0001], dtype=np.float64)
    proj_vel = np.array([0.0, 0.0, -100.0], dtype=np.float64)  # Moving downward fast

    boundary_mask = np.zeros(grid.n_nodes, dtype=bool)
    # Clamped boundary
    for i in range(nx):
        for j in range(ny):
            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                boundary_mask[i * ny + j] = True

    # Run for 20 steps
    (
        _pos_fused,
        _vel_fused,
        _failed_fused,
        _p_pos_fused,
        p_vel_fused,
        *_,
    ) = fused_leapfrog_loop(
        positions.copy(),
        np.zeros_like(positions),
        grid.springs.copy(),
        grid.stiffnesses.copy(),
        grid.rest_lengths.copy(),
        np.zeros(len(grid.springs), dtype=bool),
        grid.masses.copy(),
        grid.tension_only.copy(),
        boundary_mask,
        np.zeros((grid.n_nodes, 3)),
        proj_pos.copy(),
        proj_vel.copy(),
        proj_mass=0.01,  # Light projectile to see clear deceleration
        proj_blade_width=0.015,
        proj_edge_thickness=0.005,
        n_plies=1,
        n_nodes_per_layer=grid.n_nodes,
        t_ply=0.002,
        dx=dx,
        k_penalty=5e6,
        rayleigh_alpha=0.1,
        rayleigh_beta=0.0,
        failure_strain=0.05,
        damage_onset_strain=0.03,
        fracture_energy_multiplier=1.5,
        dt=1e-6,
        n_steps=20,
        save_interval=10,
        damp_dissipated_init=0.0,
        failure_dissipated_init=0.0,
        clamp_dissipated_init=0.0,
        t_sim_init=0.0,
        strike_direction=0.0,
        node_initial_springs=grid.initial_spring_counts,
        node_spring_offsets=grid.node_spring_offsets,
        node_spring_ids=grid.node_spring_ids,
        node_spring_signs=grid.node_spring_signs,
    )

    # Projectile was moving at -100 m/s. It should slow down because of contact forces.
    # Its Z-velocity should be less negative (i.e. > -100.0)
    assert p_vel_fused[2] > -100.0
    # Deceleration should be significant (at least 0.1 m/s over 20us)
    assert p_vel_fused[2] > -99.9


def test_fused_proximity_threshold_correctness() -> None:
    """Assert that contact is localized for dx=0.01 rather than spreading to 1000+ nodes."""
    from kevlargrid.solver.grid import generate_rectangular_grid

    nx, ny = 50, 50
    dx = 0.01
    mat = {
        "density": 1440.0,
        "k_ortho": 1e6,
        "k_shear": 2e5,
        "failure_strain": 0.05,
    }
    grid = generate_rectangular_grid(nx=nx, ny=ny, dx=dx, material=mat)

    # Projectile exactly at center node
    proj_pos = np.array([25 * dx, 25 * dx, -0.0001], dtype=np.float64)

    boundary_mask = np.zeros(grid.n_nodes, dtype=bool)

    # Run 1 step
    (
        _pos_fused,
        vel_fused,
        _failed_fused,
        _p_pos_fused,
        _p_vel_fused,
        *_,
    ) = fused_leapfrog_loop(
        grid.nodes.copy(),
        np.zeros_like(grid.nodes),
        grid.springs.copy(),
        grid.stiffnesses.copy(),
        grid.rest_lengths.copy(),
        np.zeros(len(grid.springs), dtype=bool),
        grid.masses.copy(),
        grid.tension_only.copy(),
        boundary_mask,
        np.zeros((grid.n_nodes, 3)),
        proj_pos.copy(),
        np.array([0.0, 0.0, -10.0]),
        proj_mass=0.1,
        proj_blade_width=0.01,
        proj_edge_thickness=0.002,
        n_plies=1,
        n_nodes_per_layer=grid.n_nodes,
        t_ply=0.002,
        dx=dx,
        k_penalty=1e6,
        rayleigh_alpha=0.0,
        rayleigh_beta=0.0,
        failure_strain=0.05,
        damage_onset_strain=0.03,
        fracture_energy_multiplier=1.5,
        dt=1e-7,
        n_steps=1,
        save_interval=1,
        damp_dissipated_init=0.0,
        failure_dissipated_init=0.0,
        clamp_dissipated_init=0.0,
        t_sim_init=0.0,
        strike_direction=0.0,
        node_initial_springs=grid.initial_spring_counts,
        node_spring_offsets=grid.node_spring_offsets,
        node_spring_ids=grid.node_spring_ids,
        node_spring_signs=grid.node_spring_signs,
    )

    # If the proximity threshold was computed correctly as dx * 2 = 0.02,
    # the nodes that moved (got positive velocities/accelerations in Z due to contact force)
    # should be small (definitely < 100 nodes).
    # If the fallback dx=0.05 was used, threshold = 0.10, bounding box would cover center +/- 0.105,
    # which would cover a 21x21 node area = 441 nodes!
    contact_active_nodes = np.sum(vel_fused[:, 2] != 0.0)
    assert 0 < contact_active_nodes < 100
