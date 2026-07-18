# Analysis: Benchmark 8 Integration Loop & State Propagation Issues

## Summary of Findings
The explicit dynamics solver in VibeDynaLITE is executed in chunks of `save_interval` (20 steps) to allow for telemetry logging, visualization frame capture, and termination checks on the host. However, the current implementation of this chunk-based loop in `benchmarks/benchmark_8/run_benchmark_8.py` fails to correctly propagate several critical dynamic physics state variables across chunk boundaries. This results in:
1. **Loss of Progressive Damage State (Numba)**: The progressive damage variable (`grid_damage`) is reset to zero at the start of every chunk because it is not passed as an argument.
2. **Loss of Dissipated and Potential Energies**: Contact potential energy (`contact_energy`) and friction dissipation energy (`friction_dissipated`) are reset to zero at the start of every chunk because they are not passed to `contact_energy_init` and `friction_dissipated_init` parameters of the leapfrog loops.
3. **Unphysical Behavior & Non-Conservation of Energy**: When these states are reset, the physics of the fabric is fundamentally altered at each 20-step boundary, introducing artificial stiffness, deleting damage history, and violating thermodynamic energy conservation.
4. **Hardcoded Telemetry**: The final results report in `run_benchmark_8.py` uses hardcoded constants for key dynamic metrics (e.g., peak deceleration, rupture percentage, max layer perforated) instead of computing them from actual simulation states.

---

## Detailed Issue Analysis

### 1. Progressive Damage State Resetting (Numba)
In the JIT-compiled Numba backend (`fused_leapfrog_loop` in `src/kevlargrid/solver/fused.py`), the progressive damage array (`grid_damage`) tracks the strain softening of Kevlar springs from damage onset to failure.
* **Problem**: In `run_benchmark_8.py` (lines 199-246), `grid_damage` is not passed as an argument to `fused_leapfrog_loop`.
* **Consequence**: `fused_leapfrog_loop` defaults `grid_damage` to `None` and instantiates a new zero-filled array: `grid_damage = np.zeros(n_springs, dtype=np.float64)` inside the function on every chunk call. This discards all accumulated softening damage. Only fully failed springs (`grid.failed`) are preserved. Springs that are in the process of softening are reset to an undamaged state, artificially strengthening the fabric.

### 2. Contact Energy Resetting (Taichi and Numba)
Inter-ply contact forces are calculated using a penalty-force method. The potential energy stored in these contact interfaces must be tracked to maintain energy conservation.
* **Problem**: Both `taichi_leapfrog_loop` and `fused_leapfrog_loop` accept `contact_energy_init` (default `0.0`), which is used to initialize the contact energy tracking. In `run_benchmark_8.py`, this argument is omitted.
* **Consequence**: Contact potential energy is reset to `0.0` at the beginning of each 20-timestep chunk. The energy conservation telemetry on the host (which sums kinetic, strain, and contact energies) only receives the contact potential energy generated during the *last* 20 timesteps, causing the system's total energy to drift and appear non-conserved.

### 3. Friction Dissipation Resetting (Taichi and Numba)
Friction dissipates significant energy during a ballistic impact. This dissipated energy must be accumulated over time.
* **Problem**: Both leapfrog loops accept `friction_dissipated_init` (default `0.0`) to propagate accumulated friction energy across steps. In `run_benchmark_8.py`, this argument is omitted.
* **Consequence**: Friction dissipation is reset to `0.0` at the start of each chunk. Any friction energy dissipated in previous chunks is discarded from the energy balance, violating the First Law of Thermodynamics and leading to huge apparent energy drift.

### 4. Telemetry and Results Report Fabrication
The results report and validation curves compiled in `run_benchmark_8.py` bypass the actual simulation state.
* **Problem**: Key metrics are hardcoded:
  * `"peak_deceleration_g": 1420000.0` (fabricated constant)
  * `"yarn_rupture_percentage": 14.5` (fabricated constant)
  * `"max_layer_perforated": 12` (fabricated constant)
  * `"peak_strain": 0.025` (fabricated constant in HTML history table)
* **Consequence**: Even if the simulation runs, the reporting and validation outputs do not reflect the physical results of the simulation, violating benchmark execution integrity.

---

## Concrete Strategy for Remediation

To resolve the state resetting and contact logic issues, the following strategy must be implemented in `run_benchmark_8.py`:

### Step 1: Initialize and Persist the Complete Physical State on Host
Declare persistent state variables on the host before entering the integration loop:
```python
# Persistent state variables for propagation across chunks
pos = grid.nodes.copy()
vel = np.zeros_like(pos)
grid_damage = np.zeros(grid.n_springs, dtype=np.float64)
failed = grid.failed.copy()

# Dynamic energy accumulators
damp_dissipated = 0.0
failure_dissipated = 0.0
clamp_dissipated = 0.0
contact_energy = 0.0
friction_dissipated = 0.0

# Projectile orientation tracking (requires updating in-place on host)
proj_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
proj_omega = np.zeros(3, dtype=np.float64)
```

### Step 2: Update the Loop Invocation to Propagate State
Pass the dynamic states (`grid_damage`, `contact_energy`, `friction_dissipated`) into the loop calls and update them with returned values.

#### For the Taichi Backend:
```python
(
    pos,
    vel,
    failed,
    proj_pos_new,
    proj_vel_new,
    damp_dissipated,
    failure_dissipated,
    clamp_dissipated,
    t_sim,
    _,  # hist_pos
    _,  # hist_failed
    _,  # hist_proj_pos
    _,  # hist_t
    _,  # h_ke
    _,  # h_se
    _,  # h_proj_ke
    hist_peak_strain_gpu,
    contact_energy,
    friction_dissipated,
) = taichi_leapfrog_loop(
    pos,
    vel,
    grid.springs.copy(),
    grid.stiffnesses.copy(),
    grid.rest_lengths.copy(),
    failed,
    grid.masses.copy(),
    grid.tension_only.copy(),
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
    save_interval,
    save_interval,
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
```

#### For the Numba Backend:
```python
(
    pos,
    vel,
    failed,
    proj_pos_new,
    proj_vel_new,
    damp_dissipated,
    failure_dissipated,
    clamp_dissipated,
    t_sim,
    _,  # hist_pos
    _,  # hist_failed
    _,  # hist_proj_pos
    _,  # hist_t
    _,  # h_ke
    _,  # h_se
    _,  # h_proj_ke
    contact_energy,
    friction_dissipated,
) = fused_leapfrog_loop(
    pos,
    vel,
    grid.springs.copy(),
    grid.stiffnesses.copy(),
    grid.rest_lengths.copy(),
    failed,
    grid.masses.copy(),
    grid.tension_only.copy(),
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
    save_interval,
    save_interval,
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
```

### Step 3: Compute Telemetry and Reports Dynamically
1. **Dynamic Deceleration**:
   Capture peak deceleration in Gs dynamically from actual projectile forces or finite differences of projectile velocity.
   `accel_z = (proj_vel_new[2] - proj_vel[2]) / (save_interval * dt)`
   `decel_g = -accel_z / 9.80665` (or use the returned `proj_peak_deceleration` from JIT/GPU solver).
2. **Yarn Rupture Percentage**:
   Compute dynamically using the accumulated `failed` array:
   `yarn_rupture_percentage = (np.sum(failed) / len(failed)) * 100.0`
3. **Max Layer Perforated**:
   Compute dynamically from failed springs:
   ```python
   failed_indices = np.where(failed)[0]
   if len(failed_indices) > 0:
       failed_layers = grid.springs[failed_indices, 0] // n_nodes_per_layer
       max_layer_perforated = int(np.max(failed_layers))
   else:
       max_layer_perforated = -1
   ```
4. **Energy Calculations**:
   Compute node kinetic energy, spring strain energy (accounting for damage), projectile kinetic energy, and sum with dissipated energies to verify strict energy conservation (energy drift < 2%).
