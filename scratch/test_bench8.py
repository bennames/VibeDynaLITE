import time

import numpy as np
import taichi as ti

from kevlargrid.solver.grid import generate_rectangular_grid
from kevlargrid.solver.taichi_solver import taichi_leapfrog_loop
from kevlargrid.solver.timestep import compute_cfl_timestep

# Initialize Taichi
ti.init(arch=ti.metal if ti.metal else ti.cpu)


def main():
    print("Setting up grid...")
    nx, ny = 138, 138
    dx = 0.00182
    nx * ny

    material_kev29 = {
        "tensile_modulus_gpa": 70.5,
        "areal_density_kgm2": 0.475,
        "fiber_density_gcc": 1.44,
        "failure_strain": 0.038,
        "shear_ratio": 0.0004,
    }

    grid = generate_rectangular_grid(nx, ny, dx, material_kev29, n_plies=13, t_ply=0.0001)
    print(f"Total nodes: {grid.n_nodes}, Total springs: {grid.n_springs}")

    boundary_mask = np.zeros(grid.n_nodes, dtype=bool)
    n_nodes_per_layer = nx * ny
    for ply in range(13):
        offset = ply * n_nodes_per_layer
        for i in range(nx):
            for j in range(ny):
                if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                    boundary_mask[offset + i * ny + j] = True

    proj_mass = 0.0011
    R = 0.00273
    L = 0.006
    I_zz = 0.5 * proj_mass * R**2
    I_xx = (1.0 / 12.0) * proj_mass * (3.0 * R**2 + L**2)
    proj_inertia_inv = np.diag([1.0 / I_xx, 1.0 / I_xx, 1.0 / I_zz])

    proj_pos = np.array([0.0, 0.0, -0.002], dtype=np.float64)
    proj_vel = np.array([0.0, 0.0, 503.0], dtype=np.float64)
    proj_omega = np.zeros(3, dtype=np.float64)
    proj_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    k_penalty = 1.5e6
    mu_s = 0.20
    dt = compute_cfl_timestep(grid.stiffnesses, grid.masses, dx, 0.1)
    print(f"Computed timestep: {dt} s")

    node_initial_springs = grid.initial_spring_counts
    node_spring_offsets = grid.node_spring_offsets
    node_spring_ids = grid.node_spring_ids
    node_spring_signs = grid.node_spring_signs

    t0 = time.perf_counter()
    print("Running simulation step...")
    res = taichi_leapfrog_loop(
        grid.nodes.copy(),
        np.zeros_like(grid.nodes),
        grid.springs.copy(),
        grid.stiffnesses.copy(),
        grid.rest_lengths.copy(),
        grid.failed.copy(),
        grid.masses.copy(),
        grid.tension_only.copy(),
        boundary_mask,
        np.zeros((grid.n_nodes, 3)),
        proj_pos,
        proj_vel,
        proj_mass,
        0.0,  # blade_width
        0.0,  # edge_thickness
        13,
        n_nodes_per_layer,
        0.0001,
        dx,
        k_penalty,
        0.01,  # rayleigh_alpha
        5e-8,  # rayleigh_beta
        0.038,  # failure_strain
        0.0228,  # damage_onset_strain
        1.5,  # fracture_energy_multiplier
        dt,
        1000,
        1000,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        node_initial_springs,
        node_spring_offsets,
        node_spring_ids,
        node_spring_signs,
        use_viscous=False,
        cfl_factor=0.1,
        mu_s=mu_s,
        proj_quat=proj_quat,
        proj_omega=proj_omega,
        proj_shape_type="cylinder",
        proj_radius=R,
        proj_length=L,
        proj_inertia_inv=proj_inertia_inv,
    )
    t1 = time.perf_counter()
    print(f"Completed 1000 steps in {t1 - t0:.2f} s")
    print(f"Final projectile pos Z: {res[3][2]:.5f} m")
    print(f"Final projectile vel Z: {res[4][2]:.2f} m/s")


if __name__ == "__main__":
    main()
