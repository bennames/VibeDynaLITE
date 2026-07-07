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

# 1 step manual calculation
# Let's compute strains and spring forces
p1 = positions[grid.springs[:, 0]]
p2 = positions[grid.springs[:, 1]]
diff = p2 - p1
lengths = np.sqrt(np.sum(diff**2, axis=1))
strains = (lengths - grid.rest_lengths) / grid.rest_lengths
newly_failed = (strains > 0.05)
grid_failed = grid.failed | newly_failed

lengths_safe = np.where(lengths == 0.0, 1.0, lengths)
damage = np.minimum(np.maximum((strains - 0.03) / 0.02, 0.0), 1.0)
effective_k = grid.stiffnesses * (1.0 - damage)
f_mag = effective_k * strains * grid.rest_lengths
f_mag = np.where(grid.tension_only & (strains < 0.0), 0.0, f_mag)
f_mag = np.where(grid_failed, 0.0, f_mag)

# Stiffness damping
v1 = velocities[grid.springs[:, 0]]
v2 = velocities[grid.springs[:, 1]]
v_rel = v2 - v1
unit_axes = diff / lengths_safe[:, np.newaxis]
v_proj = np.sum(v_rel * unit_axes, axis=1)
stiff_damp_mag = 0.0001 * grid.stiffnesses * v_proj
stiff_damp_mag = np.where(grid_failed, 0.0, stiff_damp_mag)

total_mag = f_mag + stiff_damp_mag
force_vecs = total_mag[:, np.newaxis] * unit_axes

spring_forces = np.zeros(positions.shape)
for i, (n0, n1) in enumerate(grid.springs):
    spring_forces[n0] += force_vecs[i]
    spring_forces[n1] -= force_vecs[i]

# Projectile contact
w_h = 0.5
t_h = 0.5
proj_position = np.array([0.0, 0.0, 10.0])
proj_velocity = np.zeros(3)
proj_mass = 1.0
k_penalty = 1e6
rayleigh_alpha = 0.1

x_proj = np.maximum(proj_position[0] - w_h, np.minimum(positions[:, 0], proj_position[0] + w_h))
y_proj = np.maximum(proj_position[1] - t_h, np.minimum(positions[:, 1], proj_position[1] + t_h))
dists = np.sqrt((positions[:, 0] - x_proj)**2 + (positions[:, 1] - y_proj)**2 + (positions[:, 2] - proj_position[2])**2)
contact_mask = dists <= 0.02

w_i = np.where(contact_mask, 1.0 / np.maximum(dists, 1e-4), 0.0)
n_contacts = np.sum(contact_mask)
w_sum = np.sum(w_i)
w_mean = w_sum / (n_contacts if n_contacts > 0 else 1)
w_mean_safe = w_mean if w_mean > 0.0 else 1.0
w_normalized = np.where(contact_mask, w_i / w_mean_safe, 0.0)

penetration = np.maximum(0.0, (proj_position[2] - positions[:, 2]) * 1.0)
f_i = k_penalty * w_normalized * penetration

active_springs = np.where(grid_failed, 0, 1)
active_counts = np.zeros(n_nodes)
for i, (n0, n1) in enumerate(grid.springs):
    active_counts[n0] += active_springs[i]
    active_counts[n1] += active_springs[i]

node_initial_springs = active_counts.copy() # assuming no failed initial springs
scale_factor = np.where(node_initial_springs > 0, active_counts / node_initial_springs, 0.0)
f_i = f_i * scale_factor
proj_forces = np.zeros_like(positions)
proj_forces[:, 2] = f_i * 1.0

# Mass damping
f_mass_damp = -rayleigh_alpha * grid.masses[:, np.newaxis] * velocities

# Net forces
net_forces = spring_forces + proj_forces + f_mass_damp

print("--- Step 1 Forces ---")
print(f"Max spring force: {np.max(np.abs(spring_forces)):.3e}")
print(f"Max proj force: {np.max(np.abs(proj_forces)):.3e}")
print(f"Max mass damp force: {np.max(np.abs(f_mass_damp)):.3e}")
print(f"Masses: min={np.min(grid.masses):.3e}, max={np.max(grid.masses):.3e}")
print(f"Boundary mask count: {np.sum(boundary_mask)}")

accel = net_forces / grid.masses[:, np.newaxis]
accel[boundary_mask] = 0.0
new_vel = velocities + accel * dt
new_vel[boundary_mask] = 0.0

print(f"Max acceleration: {np.max(np.abs(accel)):.3e}")
print(f"Max new velocity: {np.max(np.abs(new_vel)):.3e}")
