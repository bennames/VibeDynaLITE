import sys
import time
from pathlib import Path

import numpy as np

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import taichi as ti

from kevlargrid.solver.grid import generate_rectangular_grid
from kevlargrid.solver.taichi_solver import TaichiSolver
from kevlargrid.solver.timestep import compute_cfl_timestep

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

# Compile a simple 1 step graph to warm up
print("Warming up JIT...")
sys.stdout.flush()
solver.get_or_compile_graph(1, True, True, 20)

for size in [1, 2, 5, 10]:
    print(f"\n--- Testing graph size: {size} steps ---")
    sys.stdout.flush()
    try:
        t0 = time.time()
        g = solver.get_or_compile_graph(size, True, True, 20)
        t_compile = time.time() - t0
        print(f"Compilation took {t_compile:.2f} seconds.")

        t0 = time.time()
        for _ in range(100 // size):
            g.run({})
        # Sync GPU to get accurate timing
        ti.sync()
        t_run = time.time() - t0
        print(f"Running 100 steps took {t_run:.2f} seconds (average {t_run / 100:.3f} s/step).")
        sys.stdout.flush()
    except Exception as e:
        print(f"Failed for size {size}: {e}")
        sys.stdout.flush()
