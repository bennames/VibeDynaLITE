import numpy as np

from kevlargrid.solver.fused import fused_leapfrog_loop
from kevlargrid.solver.grid import generate_rectangular_grid
from kevlargrid.solver.timestep import compute_cfl_timestep

nx, ny = 31, 31
dx = 0.01
n_nodes = nx * ny

material_kev29 = {
    "tensile_modulus_gpa": 71.0,
    "areal_density_kgm2": 0.47,
    "fiber_density_gcc": 1.44,
    "shear_ratio": 0.0004,
}

proj_mass = 0.0011
blade_width = 0.00635
edge_thickness = 0.00635

boundary_mask = np.zeros(n_nodes, dtype=bool)
for i in range(nx):
    for j in range(ny):
        if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
            boundary_mask[i * ny + j] = True

for k_penalty in [5e4, 1e5, 1.5e5, 2e5, 2.5e5, 3e5]:
    grid_b = generate_rectangular_grid(nx, ny, dx, material_kev29)
    pos_b = grid_b.nodes.copy()
    vel_b = np.zeros_like(pos_b)
    proj_pos_b = np.array([0.0, 0.0, -0.002])
    proj_vel_b = np.array([0.0, 0.0, 400.0])
    dt = compute_cfl_timestep(grid_b.stiffnesses, grid_b.masses, dx, 0.2)
    t_sim = 0.0

    penetrated = False
    for _step in range(600):
        (
            pos_b, vel_b, grid_b.failed, proj_pos_b, proj_vel_b, *_
        ) = fused_leapfrog_loop(
            pos_b, vel_b, grid_b.springs, grid_b.stiffnesses, grid_b.rest_lengths,
            grid_b.failed, grid_b.masses, grid_b.tension_only, boundary_mask, np.zeros((n_nodes, 3)),
            proj_pos_b, proj_vel_b, proj_mass, blade_width, edge_thickness,
            1, n_nodes, 0.002, dx, k_penalty, 0.05, 1e-7, 0.04, 0.024, 1.5,
            dt, 1, 1, 0.0, 0.0, 0.0, t_sim, 1.0,
            grid_b.initial_spring_counts, grid_b.node_spring_offsets, grid_b.node_spring_ids, grid_b.node_spring_signs
        )
        t_sim += dt

        # Check if penetrated
        if proj_pos_b[2] > 0.01 and proj_vel_b[2] > 100.0 and np.sum(grid_b.failed) > 0:
            penetrated = True
            break

    print(f"k_penalty = {k_penalty:.2e} | penetrated? {penetrated} | final Z-pos={proj_pos_b[2]:.6f}, final Z-vel={proj_vel_b[2]:.2f}, failed springs={np.sum(grid_b.failed)}")
