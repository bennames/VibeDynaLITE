import numpy as np

from kevlargrid.solver.energy import compute_kinetic_energy, compute_strain_energy
from kevlargrid.solver.fused import fused_leapfrog_loop
from kevlargrid.solver.grid import generate_rectangular_grid
from kevlargrid.solver.timestep import compute_cfl_timestep

MOCK_MATERIAL = {
    "tensile_modulus_gpa": 71.0,
    "areal_density_kgm2": 0.47,
    "fiber_density_gcc": 1.44,
    "shear_ratio": 0.0004,
}

nx, ny, dx = 10, 10, 0.01
n_nodes = nx * ny

for beta in [0.0, 1e-7]:
    grid = generate_rectangular_grid(nx, ny, dx, MOCK_MATERIAL)
    positions = grid.nodes.copy()
    velocities = np.zeros_like(positions)
    center_node = (nx // 2) * ny + (ny // 2)
    velocities[center_node, 2] = 30.0

    boundary_mask = np.zeros(n_nodes, dtype=bool)
    for i in range(nx):
        for j in range(ny):
            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                boundary_mask[i * ny + j] = True

    # CFL safety factor = 0.1
    dt = compute_cfl_timestep(grid.stiffnesses, grid.masses, dx, 0.1)

    t_sim = 0.0
    damp_diss = 0.0
    failure_diss = 0.0
    clamp_diss = 0.0

    physical_energies = []
    total_energies = []

    print(f"\n--- Testing rayleigh_beta = {beta} ---")
    for idx in range(20):
        (
            positions,
            velocities,
            grid.failed,
            proj_pos_val,
            proj_vel_val,
            damp_diss,
            failure_diss,
            clamp_diss,
            t_sim,
            *hist_vars,
        ) = fused_leapfrog_loop(
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
            np.array([0.0, 0.0, 10.0]),
            np.zeros(3),
            1.0,
            1.0,
            1.0,
            1,
            n_nodes,
            0.002,
            dx,
            1e6,
            0.1,
            beta,
            0.05,
            0.03,
            1.0,
            dt,
            10,
            10,
            damp_diss,
            failure_diss,
            clamp_diss,
            t_sim,
            0.0,
            grid.initial_spring_counts,
            grid.node_spring_offsets,
            grid.node_spring_ids,
            grid.node_spring_signs,
        )

        ke = compute_kinetic_energy(velocities, grid.masses)
        p1 = positions[grid.springs[:, 0]]
        p2 = positions[grid.springs[:, 1]]
        lengths = np.sqrt(np.sum((p2 - p1) ** 2, axis=1))
        strains = (lengths - grid.rest_lengths) / grid.rest_lengths
        se = compute_strain_energy(strains, grid.stiffnesses, grid.rest_lengths, grid.failed)

        e_physical = ke + se
        e_total = e_physical + damp_diss + failure_diss + clamp_diss

        if idx % 5 == 0 or idx == 19:
            print(
                f"Iter {idx + 1}: ke={ke:.3e}, se={se:.3e}, damp_diss={damp_diss:.3e}, clamp_diss={clamp_diss:.3e}, total={e_total:.3e}"
            )
        physical_energies.append(e_physical)
        total_energies.append(e_total)

    # Check conservation
    initial_total_energy = total_energies[0]
    conserved = True
    for e_tot in total_energies:
        if np.abs(e_tot - initial_total_energy) / initial_total_energy >= 0.001:
            conserved = False

    decaying = True
    for i in range(1, len(physical_energies)):
        if physical_energies[i] > physical_energies[i - 1] + 1e-9:
            decaying = False

    print(f"Conserved? {conserved} | Decaying? {decaying}")
