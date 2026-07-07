import sys
import time
import numpy as np
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from kevlargrid.solver.grid import generate_rectangular_grid
from kevlargrid.solver.timestep import compute_cfl_timestep
from kevlargrid.solver.fused import fused_leapfrog_loop

print("Initializing grid...")
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

print("Running Numba solver loop...")
sys.stdout.flush()

initial_energy = 0.5 * proj_mass * 450.0**2
print(f"Initial Kinetic Energy: {initial_energy:.2f} J")

t_sim = 0.0
damp_dissipated = 0.0
failure_dissipated = 0.0
clamp_dissipated = 0.0
contact_energy = 0.0
friction_dissipated = 0.0

t0 = time.perf_counter()

for step_chunk in range(10):
    (
        pos,
        vel,
        failed,
        proj_pos,
        proj_vel,
        damp_dissipated,
        failure_dissipated,
        clamp_dissipated,
        t_sim,
        _, _, _, _, _, _, _, _,
        contact_energy,
        friction_dissipated,
    ) = fused_leapfrog_loop(
        pos,
        vel,
        grid.springs,
        grid.stiffnesses,
        grid.rest_lengths,
        failed,
        grid.masses,
        grid.tension_only,
        boundary_mask,
        np.zeros((grid.n_nodes, 3)),
        proj_pos,
        proj_vel,
        proj_mass,
        0.0,  # blade_width
        0.0,  # edge_thickness
        n_plies,
        n_nodes_per_layer,
        0.0001,
        dx,
        k_penalty,
        0.0,   # rayleigh_alpha
        5e-8,  # rayleigh_beta
        0.038, # failure_strain
        0.0228, # damage_onset_strain
        1.5,   # fracture_energy_multiplier
        dt,
        100,   # n_steps
        100,   # save_interval
        damp_dissipated,
        failure_dissipated,
        clamp_dissipated,
        t_sim,
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
        grid_damage=grid_damage,
        contact_energy_init=contact_energy,
        friction_dissipated_init=friction_dissipated,
    )
    
    ke_nodes = 0.5 * np.sum(grid.masses * np.sum(vel**2, axis=1))
    p1 = pos[grid.springs[:, 0]]
    p2 = pos[grid.springs[:, 1]]
    lens = np.sqrt(np.sum((p2 - p1)**2, axis=1))
    strains = (lens - grid.rest_lengths) / grid.rest_lengths
    strains_eff = np.where(grid.tension_only & (strains < 0.0), 0.0, strains)
    se_springs_array = 0.5 * grid.stiffnesses * (1.0 - grid_damage) * (strains_eff * grid.rest_lengths)**2
    se_springs = float(np.sum(np.where(failed, 0.0, se_springs_array)))
    ke_proj = 0.5 * proj_mass * np.sum(proj_vel**2)
    
    total_energy = ke_nodes + se_springs + ke_proj + damp_dissipated + failure_dissipated + clamp_dissipated + contact_energy + friction_dissipated
    drift = (total_energy - initial_energy) / initial_energy
    
    print(f"Step {(step_chunk+1)*100:4d}: Proj Vel Z = {proj_vel[2]:6.2f} m/s | Proj Pos Z = {proj_pos[2]*1000:6.3f} mm | Drift = {drift*100:6.3f}%")
    sys.stdout.flush()

print(f"Total time elapsed: {time.perf_counter() - t0:.2f} s")
