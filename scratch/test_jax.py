import os

os.environ["KEVLARGRID_BACKEND"] = "jax"

import numpy as np

from kevlargrid.solver import backend
from kevlargrid.solver.fused import fused_leapfrog_loop

print("Active Backend:", backend.get_backend_name())
print("Active Device:", backend.get_active_device())

size = 20
n_nodes = size * size
positions = np.zeros((n_nodes, 3), dtype=np.float64)
velocities = np.zeros((n_nodes, 3), dtype=np.float64)

springs_list = []
for y in range(size):
    for x in range(size):
        idx = y * size + x
        if x < size - 1:
            springs_list.append([idx, idx + 1])
        if y < size - 1:
            springs_list.append([idx, idx + size])

grid_springs = np.array(springs_list, dtype=np.int32)
n_springs = len(grid_springs)
grid_stiffnesses = np.ones(n_springs, dtype=np.float64) * 1e5
grid_rest_lengths = np.ones(n_springs, dtype=np.float64) * 0.05
grid_failed = np.zeros(n_springs, dtype=bool)
grid_tension_only = np.ones(n_springs, dtype=bool)
grid_masses = np.ones(n_nodes, dtype=np.float64) * 0.02
boundary_mask = np.zeros(n_nodes, dtype=bool)
boundary_mask[[0, size - 1, n_nodes - size, n_nodes - 1]] = True

proj_pos = np.array([size * 0.025, size * 0.025, 0.01], dtype=np.float64)
proj_vel = np.array([0.0, 0.0, -10.0], dtype=np.float64)
proj_mass = 0.5
blade_width = 0.02
edge_thickness = 0.005
try:
    res = fused_leapfrog_loop(
        positions,
        velocities,
        grid_springs,
        grid_stiffnesses,
        grid_rest_lengths,
        grid_failed,
        grid_masses,
        grid_tension_only,
        boundary_mask,
        np.zeros((n_nodes, 3)),
        proj_pos,
        proj_vel,
        proj_mass,
        blade_width,
        edge_thickness,
        n_plies=1,
        n_nodes_per_layer=n_nodes,
        t_ply=0.002,
        dx=0.05,
        k_penalty=1e6,
        rayleigh_alpha=0.1,
        rayleigh_beta=0.0001,
        failure_strain=0.05,
        damage_onset_strain=0.03,
        fracture_energy_multiplier=1.5,
        dt=1e-5,
        n_steps=10,
        save_interval=10,
        damp_dissipated_init=0.0,
        failure_dissipated_init=0.0,
        clamp_dissipated_init=0.0,
        t_sim_init=0.0,
        strike_direction=0.0,
        node_initial_springs=np.zeros(n_nodes, dtype=np.int32),
        node_spring_offsets=np.zeros(n_nodes+1, dtype=np.int32),
        node_spring_ids=np.zeros(2*n_springs, dtype=np.int32),
        node_spring_signs=np.zeros(2*n_springs, dtype=np.float64),
    )
    print("JAX execution succeeded!")
except Exception:
    import traceback
    traceback.print_exc()
