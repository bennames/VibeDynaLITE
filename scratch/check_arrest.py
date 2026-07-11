import sys
from pathlib import Path

import numpy as np

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import taichi as ti

from kevlargrid.solver.grid import generate_rectangular_grid
from kevlargrid.solver.taichi_solver import TaichiSolver
from kevlargrid.solver.timestep import compute_cfl_timestep

print("Initializing grid...")
sys.stdout.flush()
nx, ny = 184, 184
dx = 0.001365
n_nodes_per_layer = nx * ny
n_plies = 13

material_kev29 = {
    "tensile_modulus_gpa": 70.5,
    "areal_density_kgm2": 0.475,
    "fiber_density_gcc": 1.44,
    "failure_strain": 0.038,
    "shear_ratio": 0.0004,
}

grid = generate_rectangular_grid(nx, ny, dx, material_kev29, n_plies=n_plies, t_ply=0.0001)

boundary_mask = np.zeros(grid.n_nodes, dtype=bool)
for ply in range(n_plies):
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
# Case A strike velocity: 450 m/s
proj_vel = np.array([0.0, 0.0, 450.0], dtype=np.float64)
proj_omega = np.zeros(3, dtype=np.float64)
proj_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

k_penalty = 2.0e5
mu_s = 0.20
dt = compute_cfl_timestep(grid.stiffnesses, grid.masses, dx, 0.1)

node_initial_springs = grid.initial_spring_counts
node_spring_offsets = grid.node_spring_offsets
node_spring_ids = grid.node_spring_ids
node_spring_signs = grid.node_spring_signs

pos = grid.nodes.copy()
vel = np.zeros_like(pos)
grid_damage = np.zeros(grid.n_springs, dtype=np.float64)
failed = grid.failed.copy()

solver = TaichiSolver(
    n_nodes=grid.n_nodes,
    n_springs=grid.n_springs,
    positions_init=pos,
    velocities_init=vel,
    springs_init=grid.springs,
    stiffnesses_init=grid.stiffnesses,
    rest_lengths_init=grid.rest_lengths,
    failed_init=failed,
    masses_init=grid.masses,
    tension_only_init=grid.tension_only,
    boundary_mask_init=boundary_mask,
    nodal_external_forces_init=np.zeros((grid.n_nodes, 3)),
    proj_position_init=proj_pos,
    proj_velocity_init=proj_vel,
    proj_mass_init=proj_mass,
    strike_direction_init=1.0,
    node_initial_springs_init=node_initial_springs,
    n_plies=n_plies,
    n_nodes_per_layer=n_nodes_per_layer,
)

# Set shape parameters
solver.proj_shape_type[None] = 1  # cylinder
solver.radius_field[None] = R
solver.length_field[None] = L
solver.span_field[None] = 0.0
solver.proj_inertia_inv[None] = np.diagonal(proj_inertia_inv).astype(np.float32)
solver.proj_quat[None] = ti.Vector(proj_quat)
solver.proj_omega[None] = ti.Vector(proj_omega)

solver.k_penalty[None] = k_penalty
solver.mu_s[None] = mu_s

print("Starting 1000 steps simulation...")
sys.stdout.flush()

initial_energy = 0.5 * proj_mass * 450.0**2
print(f"Initial Kinetic Energy: {initial_energy:.2f} J")

for chunk in range(10):
    # Run 100 steps
    dt = solver.run_substeps(
        100,  # num_substeps
        dt,  # dt_init
        0.0,  # rayleigh_beta
        0.0228,  # damage_onset_strain
        0.038,  # failure_strain
        n_plies,  # n_plies
        n_nodes_per_layer,  # n_nodes_per_layer
        0.0001,  # t_ply
        k_penalty,  # k_penalty
        0.0,  # w_h
        0.0,  # t_h
        5e-8,  # proximity_threshold
        0.0,  # rayleigh_alpha
        0,  # use_viscous (0 for false)
        0.1,  # cfl_factor
        dx,  # dx
        1.0,  # fracture_energy_multiplier
        mu_s=mu_s,
        cfl_recompute_interval=20,
    )

    # Get telemetry
    telem = solver.get_telemetry()
    proj_v = solver.proj_velocity[None]
    proj_v_z = proj_v.z

    # Calculate energy drift
    total_energy = telem["ke"] + telem["se"] + telem["proj_ke"]
    drift = (total_energy - initial_energy) / initial_energy

    print(
        f"Step {(chunk + 1) * 100:4d}: Proj Vel Z = {proj_v_z:6.2f} m/s, KE = {telem['proj_ke']:6.2f} J, Grid KE = {telem['ke']:6.2f} J, Grid SE = {telem['se']:6.2f} J, Drift = {drift * 100:6.3f}%"
    )
    sys.stdout.flush()
