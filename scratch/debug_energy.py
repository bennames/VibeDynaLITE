import numpy as np

from kevlargrid.solver.energy import compute_kinetic_energy, compute_strain_energy
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
k_penalty = 1e6

boundary_mask = np.zeros(n_nodes, dtype=bool)
for i in range(nx):
    for j in range(ny):
        if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
            boundary_mask[i * ny + j] = True

grid_b = generate_rectangular_grid(nx, ny, dx, material_kev29)
pos_b = grid_b.nodes.copy()
vel_b = np.zeros_like(pos_b)
proj_pos_b = np.array([0.0, 0.0, -0.002])
proj_vel_b = np.array([0.0, 0.0, 400.0])
dt = compute_cfl_timestep(grid_b.stiffnesses, grid_b.masses, dx, 0.2)
t_sim = 0.0

damp_diss = 0.0
failure_diss = 0.0
clamp_diss = 0.0

print("Step | Proj KE | Grid KE | Grid SE | Damp Diss | Failure Diss | Total Energy | Max Strain")
print("-----------------------------------------------------------------------------------------")

for step in range(200):
    (
        pos_b,
        vel_b,
        grid_b.failed,
        proj_pos_b,
        proj_vel_b,
        damp_diss,
        failure_diss,
        clamp_diss,
        t_sim,
        *_,
    ) = fused_leapfrog_loop(
        pos_b,
        vel_b,
        grid_b.springs,
        grid_b.stiffnesses,
        grid_b.rest_lengths,
        grid_b.failed,
        grid_b.masses,
        grid_b.tension_only,
        boundary_mask,
        np.zeros((n_nodes, 3)),
        proj_pos_b,
        proj_vel_b,
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
        1,
        1,
        damp_diss,
        failure_diss,
        clamp_diss,
        t_sim,
        1.0,
        grid_b.initial_spring_counts,
        grid_b.node_spring_offsets,
        grid_b.node_spring_ids,
        grid_b.node_spring_signs,
    )

    # Calculate energies
    proj_ke = 0.5 * proj_mass * proj_vel_b[2] ** 2
    grid_ke = compute_kinetic_energy(vel_b, grid_b.masses)

    p1 = pos_b[grid_b.springs[:, 0]]
    p2 = pos_b[grid_b.springs[:, 1]]
    lengths = np.sqrt(np.sum((p2 - p1) ** 2, axis=1))
    strains = (lengths - grid_b.rest_lengths) / grid_b.rest_lengths
    grid_se = compute_strain_energy(strains, grid_b.stiffnesses, grid_b.rest_lengths, grid_b.failed)

    total_e = proj_ke + grid_ke + grid_se + damp_diss + failure_diss + clamp_diss
    max_strain = np.max(strains)

    if step % 10 == 0 or step >= 140:
        print(
            f"{step:4d} | {proj_ke:7.3f} | {grid_ke:7.3f} | {grid_se:7.3f} | {damp_diss:9.3f} | {failure_diss:12.3f} | {total_e:12.3f} | {max_strain:10.6f}"
        )
