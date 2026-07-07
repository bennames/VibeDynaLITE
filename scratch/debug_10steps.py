import numpy as np

from kevlargrid.solver.grid import generate_rectangular_grid

MOCK_MATERIAL = {
    "tensile_modulus_gpa": 71.0,
    "areal_density_kgm2": 0.47,
    "fiber_density_gcc": 1.44,
    "shear_ratio": 0.0004,
}

nx, ny, dx = 10, 10, 0.01
n_nodes = nx * ny
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

dt = 7.120690e-08

# Run 10 steps with beta = 0
for step in range(10):
    p1 = positions[grid.springs[:, 0]]
    p2 = positions[grid.springs[:, 1]]
    diff = p2 - p1
    lengths = np.sqrt(np.sum(diff**2, axis=1))
    strains = (lengths - grid.rest_lengths) / grid.rest_lengths
    grid_failed = grid.failed # no failure yet

    lengths_safe = np.where(lengths == 0.0, 1.0, lengths)
    damage = np.minimum(np.maximum((strains - 0.03) / 0.02, 0.0), 1.0)
    effective_k = grid.stiffnesses * (1.0 - damage)
    f_mag = effective_k * strains * grid.rest_lengths
    f_mag = np.where(grid.tension_only & (strains < 0.0), 0.0, f_mag)
    f_mag = np.where(grid_failed, 0.0, f_mag)

    # beta = 0
    stiff_damp_mag = np.zeros_like(f_mag)

    total_mag = f_mag + stiff_damp_mag
    unit_axes = diff / lengths_safe[:, np.newaxis]
    force_vecs = total_mag[:, np.newaxis] * unit_axes

    spring_forces = np.zeros(positions.shape)
    for i, (n0, n1) in enumerate(grid.springs):
        spring_forces[n0] += force_vecs[i]
        spring_forces[n1] -= force_vecs[i]

    f_mass_damp = -0.1 * grid.masses[:, np.newaxis] * velocities
    net_forces = spring_forces + f_mass_damp

    accel = net_forces / grid.masses[:, np.newaxis]
    accel[boundary_mask] = 0.0
    velocities = velocities + accel * dt
    velocities[boundary_mask] = 0.0
    positions = positions + velocities * dt

    ke = 0.5 * np.sum(grid.masses[:, np.newaxis] * velocities**2)
    print(f"Step {step+1}: max_vel={np.max(np.abs(velocities)):.3e}, grid_ke={ke:.3e}")
