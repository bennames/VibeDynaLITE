import numpy as np

from kevlargrid.solver.grid import generate_rectangular_grid
from kevlargrid.solver.taichi_solver import taichi_leapfrog_loop
from kevlargrid.solver.timestep import compute_cfl_timestep


def run_debug(v_strike):
    nx, ny = 31, 31
    dx = 0.01
    n_nodes = nx * ny
    center_idx = (nx // 2) * ny + (ny // 2)

    material_kev29 = {
        "tensile_modulus_gpa": 71.0,
        "areal_density_kgm2": 0.47,
        "fiber_density_gcc": 1.44,
        "shear_ratio": 0.0004,
    }

    proj_mass = 0.0011
    blade_width = 0.00635
    edge_thickness = 0.00635
    k_penalty = 2e5

    boundary_mask = np.zeros(n_nodes, dtype=bool)
    for i in range(nx):
        for j in range(ny):
            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                boundary_mask[i * ny + j] = True

    grid = generate_rectangular_grid(nx, ny, dx, material_kev29)
    positions = grid.nodes.copy()
    velocities = np.zeros_like(positions)

    proj_pos = np.array([0.0, 0.0, -0.002], dtype=np.float64)
    proj_vel = np.array([0.0, 0.0, v_strike], dtype=np.float64)

    dt = compute_cfl_timestep(grid.stiffnesses, grid.masses, dx, 0.2)
    t_sim = 0.0

    # Get center node springs
    start = grid.node_spring_offsets[center_idx]
    end = grid.node_spring_offsets[center_idx + 1]
    c_spring_ids = grid.node_spring_ids[start:end]
    c_spring_tension_only = grid.tension_only[c_spring_ids]

    print(f"Center Node Springs: {c_spring_ids}")
    print(f"Center Node Springs Tension Only: {c_spring_tension_only}")

    print(f"\n--- Detailed Debug: strike velocity = {v_strike} m/s ---")
    for chunk in range(6):
        (
            positions,
            velocities,
            grid.failed,
            proj_pos,
            proj_vel,
            *_,
        ) = taichi_leapfrog_loop(
            positions,
            velocities,
            grid.springs,
            grid.stiffnesses,
            grid.rest_lengths,
            grid.failed,
            grid.masses,
            grid.tension_only,
            boundary_mask,
            np.zeros((n_nodes, 3)),
            proj_pos,
            proj_vel,
            proj_mass,
            blade_width,
            edge_thickness,
            1,
            n_nodes,
            0.002,
            dx,
            k_penalty,
            0.05,
            1e-7,
            0.04,
            0.024,
            1.5,
            dt,
            100,
            100,
            0.0,
            0.0,
            0.0,
            t_sim,
            1.0,
            grid.initial_spring_counts,
            grid.node_spring_offsets,
            grid.node_spring_ids,
            grid.node_spring_signs,
        )
        t_sim += 100 * dt

        # Check failed status of each center spring
        failed_statuses = grid.failed[c_spring_ids]
        print(f"  Chunk {chunk + 1}: t_sim = {t_sim:.3e} s")
        print(f"    proj_pos Z = {proj_pos[2]:.6f} m | proj_vel Z = {proj_vel[2]:.1f} m/s")
        print(f"    Center node springs failed: {failed_statuses}")

        if proj_vel[2] <= 0.0:
            print("  Arrested/rebounded!")
            break
        if proj_pos[2] > 0.015 and proj_vel[2] > 0.0:
            print("  Penetrated and passed Z = 0.015!")
            break


run_debug(400.0)
