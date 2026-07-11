import sys
import time
from pathlib import Path

import numpy as np

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from kevlargrid.solver.fused import fused_leapfrog_loop
from kevlargrid.solver.grid import generate_rectangular_grid
from kevlargrid.solver.taichi_solver import taichi_leapfrog_loop
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
proj_inertia_inv = np.diag([1.0/I_xx, 1.0/I_xx, 1.0/I_zz])

proj_pos = np.array([0.0, 0.0, -0.002], dtype=np.float64)
proj_vel = np.array([0.0, 0.0, 450.0], dtype=np.float64)
proj_omega = np.zeros(3, dtype=np.float64)
proj_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

k_penalty = 2.0e6
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

print("Initial JIT compilation of Taichi backend...")
sys.stdout.flush()
taichi_leapfrog_loop(
    pos, vel, grid.springs.copy(), grid.stiffnesses.copy(), grid.rest_lengths.copy(),
    failed, grid.masses.copy(), grid.tension_only.copy(), boundary_mask,
    np.zeros((grid.n_nodes, 3)), proj_pos, proj_vel, proj_mass, 0.0, 0.0,
    n_plies, n_nodes_per_layer, 0.0001, dx, k_penalty, 0.0, 5e-8, 0.038, 0.0228,
    1.5, dt, 5, 5, 0.0, 0.0, 0.0, 0.0, 1.0,
    node_initial_springs, node_spring_offsets, node_spring_ids, node_spring_signs,
    use_viscous=False, cfl_factor=0.1, mu_s=mu_s, proj_quat=proj_quat,
    proj_omega=proj_omega, proj_shape_type="cylinder", proj_radius=R, proj_length=L,
    proj_inertia_inv=proj_inertia_inv, grid_damage=grid_damage,
    contact_energy_init=0.0, friction_dissipated_init=0.0
)

print("Running 100 steps with Taichi...")
sys.stdout.flush()
t0 = time.time()
try:
    taichi_leapfrog_loop(
        pos, vel, grid.springs.copy(), grid.stiffnesses.copy(), grid.rest_lengths.copy(),
        failed, grid.masses.copy(), grid.tension_only.copy(), boundary_mask,
        np.zeros((grid.n_nodes, 3)), proj_pos, proj_vel, proj_mass, 0.0, 0.0,
        n_plies, n_nodes_per_layer, 0.0001, dx, k_penalty, 0.0, 5e-8, 0.038, 0.0228,
        1.5, dt, 100, 100, 0.0, 0.0, 0.0, 0.0, 1.0,
        node_initial_springs, node_spring_offsets, node_spring_ids, node_spring_signs,
        use_viscous=False, cfl_factor=0.1, mu_s=mu_s, proj_quat=proj_quat,
        proj_omega=proj_omega, proj_shape_type="cylinder", proj_radius=R, proj_length=L,
        proj_inertia_inv=proj_inertia_inv, grid_damage=grid_damage,
        contact_energy_init=0.0, friction_dissipated_init=0.0
    )
    t_taichi = time.time() - t0
    print(f"Taichi 100 steps finished in {t_taichi:.2f} seconds.")
except Exception as e:
    print("Taichi failed:", e)

sys.stdout.flush()

print("Initial JIT compilation of Numba backend...")
sys.stdout.flush()
fused_leapfrog_loop(
    pos, vel, grid.springs.copy(), grid.stiffnesses.copy(), grid.rest_lengths.copy(),
    failed, grid.masses.copy(), grid.tension_only.copy(), boundary_mask,
    np.zeros((grid.n_nodes, 3)), proj_pos, proj_vel, proj_mass, 0.0, 0.0,
    n_plies, n_nodes_per_layer, 0.0001, dx, k_penalty, 0.0, 5e-8, 0.038, 0.0228,
    1.5, dt, 5, 5, 0.0, 0.0, 0.0, 0.0, 1.0,
    node_initial_springs, node_spring_offsets, node_spring_ids, node_spring_signs,
    use_viscous=False, cfl_factor=0.1, mu_s=mu_s, proj_quat=proj_quat,
    proj_omega=proj_omega, proj_shape_type="cylinder", proj_radius=R, proj_length=L,
    proj_inertia_inv=proj_inertia_inv, grid_damage=grid_damage,
    contact_energy_init=0.0, friction_dissipated_init=0.0
)

print("Running 100 steps with Numba...")
sys.stdout.flush()
t0 = time.time()
try:
    fused_leapfrog_loop(
        pos, vel, grid.springs.copy(), grid.stiffnesses.copy(), grid.rest_lengths.copy(),
        failed, grid.masses.copy(), grid.tension_only.copy(), boundary_mask,
        np.zeros((grid.n_nodes, 3)), proj_pos, proj_vel, proj_mass, 0.0, 0.0,
        n_plies, n_nodes_per_layer, 0.0001, dx, k_penalty, 0.0, 5e-8, 0.038, 0.0228,
        1.5, dt, 100, 100, 0.0, 0.0, 0.0, 0.0, 1.0,
        node_initial_springs, node_spring_offsets, node_spring_ids, node_spring_signs,
        use_viscous=False, cfl_factor=0.1, mu_s=mu_s, proj_quat=proj_quat,
        proj_omega=proj_omega, proj_shape_type="cylinder", proj_radius=R, proj_length=L,
        proj_inertia_inv=proj_inertia_inv, grid_damage=grid_damage,
        contact_energy_init=0.0, friction_dissipated_init=0.0
    )
    t_numba = time.time() - t0
    print(f"Numba 100 steps finished in {t_numba:.2f} seconds.")
except Exception as e:
    print("Numba failed:", e)

sys.stdout.flush()
