from __future__ import annotations

import numpy as np
import pytest
import taichi as ti

from kevlargrid.solver.taichi_solver import TaichiSolver, PhysicsViolationError
from kevlargrid.solver.failure import check_progressive_damage
from kevlargrid.solver.boundary import apply_impedance_boundary
from kevlargrid.solver.grid import generate_rectangular_grid
from kevlargrid.solver.timestep import compute_cfl_timestep
from kevlargrid.solver.forces import compute_spring_forces


def test_progressive_damage_irreversible_cpu() -> None:
    """Verify that CPU check_progressive_damage evolves damage and does not self-heal on unloading."""
    strains = np.array([0.005])
    damage = np.array([0.0])
    failed = np.array([False])
    onset = 0.002
    fail = 0.006

    # Load: strain 0.005 (damage should be (0.005 - 0.002) / 0.004 = 0.75)
    d, f = check_progressive_damage(strains, damage, failed, onset, fail)
    assert np.isclose(d[0], 0.75)
    assert not f[0]

    # Unload: strain 0.001 (damage should remain at 0.75)
    d, f = check_progressive_damage(np.array([0.001]), d, f, onset, fail)
    assert np.isclose(d[0], 0.75)
    assert not f[0]

    # Rupture: strain 0.007 (damage should reach 1.0, marked as failed)
    d, f = check_progressive_damage(np.array([0.007]), d, f, onset, fail)
    assert np.isclose(d[0], 1.0)
    assert f[0]


def test_taichi_progressive_damage_irreversible() -> None:
    """Verify that TaichiSolver evolves damage irreversibly and does not self-heal."""
    # A single spring between node 0 and node 1
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    velocities = np.zeros_like(positions)
    springs = np.array([[0, 1]], dtype=np.int32)
    stiffnesses = np.array([1000.0], dtype=np.float32)
    rest_lengths = np.array([1.0], dtype=np.float32)
    failed = np.zeros(1, dtype=np.int32)
    masses = np.array([1.0, 1.0], dtype=np.float32)
    tension_only = np.zeros(1, dtype=np.int32)
    boundary_mask = np.zeros(2, dtype=np.int32)
    nodal_external_forces = np.zeros_like(positions)
    node_initial_springs = np.array([1, 1], dtype=np.int32)
    
    solver = TaichiSolver(
        n_nodes=2,
        n_springs=1,
        positions_init=positions,
        velocities_init=velocities,
        springs_init=springs,
        stiffnesses_init=stiffnesses,
        rest_lengths_init=rest_lengths,
        failed_init=failed,
        masses_init=masses,
        tension_only_init=tension_only,
        boundary_mask_init=boundary_mask,
        nodal_external_forces_init=nodal_external_forces,
        proj_position_init=np.zeros(3),
        proj_velocity_init=np.zeros(3),
        proj_mass_init=1.0,
        strike_direction_init=1.0,
        node_initial_springs_init=node_initial_springs,
    )
    
    # Set positions to strain 0.005 (node 1 at [1.005, 0.0, 0.0])
    solver.positions[1] = ti.Vector([1.005, 0.0, 0.0])
    solver.advance_substeps(
        1, 1e-6, 0.0, 0.002, 0.006, 1, 2, 0.002, 100.0, 0.01, 0.005, 0.02, 0.0, 0, 0.0, 1.0, 1.0
    )
    
    dmg = solver.spring_damage.to_numpy()
    assert np.isclose(dmg[0], 0.75)
    
    # Strain decreases to 0.001
    solver.positions[1] = ti.Vector([1.001, 0.0, 0.0])
    solver.advance_substeps(
        1, 1e-6, 0.0, 0.002, 0.006, 1, 2, 0.002, 100.0, 0.01, 0.005, 0.02, 0.0, 0, 0.0, 1.0, 1.0
    )
    
    dmg = solver.spring_damage.to_numpy()
    assert np.isclose(dmg[0], 0.75)  # Irreversible, stays at 0.75


def test_mass_scaling_energy_abort() -> None:
    """Verify that PhysicsViolationError is raised when artificial kinetic energy exceeds 2% of internal energy."""
    positions = np.array([[0.0, 0.0, 0.0], [1.01, 0.0, 0.0]], dtype=np.float32)
    # Node 1 moves quickly
    velocities = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)
    springs = np.array([[0, 1]], dtype=np.int32)
    stiffnesses = np.array([1000.0], dtype=np.float32)
    rest_lengths = np.array([1.0], dtype=np.float32)
    failed = np.zeros(1, dtype=np.int32)
    
    # Scaled mass = 100.0, Physical mass = 1.0 (Huge mass scaling!)
    masses = np.array([100.0, 100.0], dtype=np.float32)
    masses_phys = np.array([1.0, 1.0], dtype=np.float32)
    
    tension_only = np.zeros(1, dtype=np.int32)
    boundary_mask = np.zeros(2, dtype=np.int32)
    nodal_external_forces = np.zeros_like(positions)
    node_initial_springs = np.array([1, 1], dtype=np.int32)
    
    from kevlargrid.solver.taichi_solver import taichi_leapfrog_loop
    
    with pytest.raises(PhysicsViolationError, match="Physics violation: Artificial kinetic energy"):
        taichi_leapfrog_loop(
            positions=positions,
            velocities=velocities,
            grid_springs=springs,
            grid_stiffnesses=stiffnesses,
            grid_rest_lengths=rest_lengths,
            grid_failed=failed,
            grid_masses=masses,
            grid_tension_only=tension_only,
            boundary_mask=boundary_mask,
            nodal_external_forces=nodal_external_forces,
            proj_position=np.zeros(3),
            proj_velocity=np.zeros(3),
            proj_mass=1.0,
            proj_blade_width=0.01,
            proj_edge_thickness=0.005,
            n_plies=1,
            n_nodes_per_layer=2,
            t_ply=0.002,
            dx=0.5,
            k_penalty=100.0,
            rayleigh_alpha=0.0,
            rayleigh_beta=0.0,
            failure_strain=0.5,
            damage_onset_strain=0.1,
            fracture_energy_multiplier=1.0,
            dt=1e-6,
            n_steps=10,
            save_interval=10,
            damp_dissipated_init=0.0,
            failure_dissipated_init=0.0,
            clamp_dissipated_init=0.0,
            t_sim_init=0.0,
            strike_direction=1.0,
            node_initial_springs=node_initial_springs,
            grid_masses_physical=masses_phys,
        )


def test_impedance_boundary_absorption_cpu() -> None:
    """Verify that the CPU impedance boundary condition absorbs waves with <5% reflection."""
    nx, ny, dx = 50, 1, 0.1
    n_nodes = nx * ny
    MOCK_MATERIAL = {
        "tensile_modulus_gpa": 71.0,
        "areal_density_kgm2": 0.47,
        "fiber_density_gcc": 1.44,
        "shear_ratio": 0.0004,
        "failure_strain": 0.5,
    }
    grid = generate_rectangular_grid(nx, ny, dx, MOCK_MATERIAL)
    
    boundary_mask = np.zeros(n_nodes, dtype=np.int32)
    boundary_mask[49] = 2  # Impedance boundary condition
    
    positions = grid.nodes.copy()
    velocities = np.zeros((n_nodes, 3), dtype=np.float64)
    velocities[5, 0] = -10.0
    
    dt = compute_cfl_timestep(grid.stiffnesses, grid.masses, dx, 0.5)
    
    vel_30_history = []
    
    for _ in range(200):
        # Compute forces using the codebase's safe function
        forces = compute_spring_forces(
            positions, grid.springs, grid.stiffnesses, grid.rest_lengths, grid.failed,
            tension_only=grid.tension_only
        )
            
        forces = apply_impedance_boundary(
            forces, velocities, grid.masses, grid.springs, grid.stiffnesses, grid.damage, grid.failed, boundary_mask
        )
        
        # Integrate
        accel = forces / grid.masses[:, np.newaxis]
        velocities += accel * dt
        positions += velocities * dt
        
        vel_30_history.append(velocities[30, 0])
        
    vel_30_history = np.array(vel_30_history)
    incident_wave = vel_30_history[:100]
    reflected_wave = vel_30_history[100:]
    
    peak_incident = np.max(np.abs(incident_wave))
    peak_reflected = np.max(np.abs(reflected_wave)) if len(reflected_wave) > 0 else 0.0
    
    reflection_coeff = peak_reflected / peak_incident
    assert peak_incident > 0.5
    assert reflection_coeff < 0.05


def test_impedance_boundary_absorption_gpu() -> None:
    """Verify that the Taichi solver impedance boundary condition absorbs waves with <5% reflection."""
    nx, ny, dx = 50, 1, 0.1
    n_nodes = nx * ny
    MOCK_MATERIAL = {
        "tensile_modulus_gpa": 71.0,
        "areal_density_kgm2": 0.47,
        "fiber_density_gcc": 1.44,
        "shear_ratio": 0.0004,
        "failure_strain": 0.5,
    }
    grid = generate_rectangular_grid(nx, ny, dx, MOCK_MATERIAL)
    
    boundary_mask = np.zeros(n_nodes, dtype=np.int32)
    boundary_mask[49] = 2  # Impedance matched boundary
    
    velocities = np.zeros((n_nodes, 3), dtype=np.float64)
    # Pulse node 5 with negative velocity to launch a tension wave to the right
    velocities[5, 0] = -10.0
    
    positions = grid.nodes.copy()
    grid_failed = np.zeros(len(grid.springs), dtype=bool)
    
    dt = compute_cfl_timestep(grid.stiffnesses, grid.masses, dx, 0.5)
    t_sim = 0.0
    
    vel_30_history = []
    
    from kevlargrid.solver.taichi_solver import taichi_leapfrog_loop
    
    for _ in range(200):
        (
            positions,
            velocities,
            grid_failed,
            proj_pos_val,
            proj_vel_val,
            damp_diss_val,
            failure_diss_val,
            clamp_diss_val,
            t_sim,
            *_,
        ) = taichi_leapfrog_loop(
            positions,
            velocities,
            grid.springs,
            grid.stiffnesses,
            grid.rest_lengths,
            grid_failed,
            grid.masses,
            grid.tension_only,
            boundary_mask,
            np.zeros((n_nodes, 3)),
            np.zeros(3),
            np.zeros(3),
            1.0, 1.0, 1.0, 1, n_nodes, 0.002, dx, 1e6, 0.0, 0.0, 0.5, 0.3, 1.0,
            dt, 1, 1, 0.0, 0.0, 0.0, t_sim, 0.0,
            grid.initial_spring_counts, grid.node_spring_offsets, grid.node_spring_ids, grid.node_spring_signs
        )
        vel_30_history.append(velocities[30, 0])
        
    vel_30_history = np.array(vel_30_history)
    incident_wave = vel_30_history[:100]
    reflected_wave = vel_30_history[100:]
    
    peak_incident = np.max(np.abs(incident_wave))
    peak_reflected = np.max(np.abs(reflected_wave)) if len(reflected_wave) > 0 else 0.0
    
    reflection_coeff = peak_reflected / peak_incident
    assert peak_incident > 0.5
    assert reflection_coeff < 0.05
