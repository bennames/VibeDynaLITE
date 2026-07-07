# Handoff Report: Benchmark 8 Telemetry & Calibration Audit Remediation

## 1. Observation

During my read-only investigation of the VibeDynaLITE codebase and the newly created `run_benchmark_8.py` script, I made the following specific observations:

### A. Resetting of Solver State Variables Across Chunk timesteps
In `benchmarks/benchmark_8/run_benchmark_8.py` (lines 110-178 for the Taichi backend and lines 179-247 for the Numba backend), the explicit time integration is chunked into intervals of `save_interval` (20 steps) to allow host-side early termination checks. However, critical state variables are omitted from the solver calls:
*   In the Numba loop invocation:
    ```python
    199:             ) = fused_leapfrog_loop(
    200:                 pos,
    201:                 vel,
    202:                 grid.springs.copy(),
    203:                 grid.stiffnesses.copy(),
    204:                 grid.rest_lengths.copy(),
    205:                 grid.failed.copy(),
    206:                 grid.masses.copy(),
    207:                 grid.tension_only.copy(),
    208:                 boundary_mask,
    209:                 np.zeros((grid.n_nodes, 3)),
    210:                 proj_pos,
    211:                 proj_vel,
    212:                 proj_mass,
    213:                 0.0,  # blade_width
    214:                 0.0,  # edge_thickness
    215:                 n_plies,
    216:                 n_nodes_per_layer,
    217:                 0.0001,
    218:                 dx,
    219:                 k_penalty,
    220:                 0.0,   # rayleigh_alpha
    221:                 5e-8,  # rayleigh_beta
    222:                 0.038, # failure_strain
    223:                 0.0228, # damage_onset_strain
    224:                 1.5,   # fracture_energy_multiplier
    225:                 dt,
    226:                 save_interval,
    227:                 save_interval,
    228:                 damp_dissipated,
    229:                 failure_dissipated,
    230:                 clamp_dissipated,
    231:                 t_sim,
    232:                 1.0,
    233:                 node_initial_springs,
    234:                 node_spring_offsets,
    235:                 node_spring_ids,
    236:                 node_spring_signs,
    237:                 use_viscous=False,
    238:                 cfl_factor=0.1,
    239:                 mu_s=mu_s,
    240:                 proj_quat=proj_quat,
    241:                 proj_omega=proj_omega,
    242:                 proj_shape_type="cylinder",
    243:                 proj_radius=R,
    244:                 proj_length=L,
    245:                 proj_inertia_inv=proj_inertia_inv,
    246:             )
    ```
*   Because `grid_damage`, `contact_energy_init`, and `friction_dissipated_init` are omitted from the parameter list, they default to `None` or `0.0`.
*   As a result, in `src/kevlargrid/solver/fused.py`:
    ```python
    1523:     if grid_damage is None:
    1524:         grid_damage = np.zeros(n_springs, dtype=np.float64)
    ```
    This completely resets the progressive damage field of all spring elements to zero at the start of every chunk. It prevents damage accumulation across chunks.
*   Similarly, `contact_energy` and `friction_dissipated` are reset to zero at every 20-step interval. This corrupts the energy conservation checks on the host.

### B. Average Deceleration vs. Instantaneous Peak Deceleration
In `benchmarks/benchmark_8/run_benchmark_8.py` (lines 249-252), the projectile deceleration is monitored by calculating the change in velocity over a 20-step chunk:
```python
249:         accel_z = (proj_vel_new[2] - proj_vel[2]) / (save_interval * dt)
250:         decel_g = -accel_z / 9.81
251:         if decel_g > peak_decel_g:
252:             peak_decel_g = decel_g
```
*   This represents an average deceleration over `save_interval` timesteps, which severely smooths out the peak deceleration.
*   However, both solver engines already track instantaneous peak deceleration at *every single timestep* inside the JIT and GPU loops:
    *   **Numba JIT (`src/kevlargrid/solver/fused.py` lines 1339-1341):**
        ```python
        1339:         accel_mag_g = accel_mag / 9.80665
        1340:         if accel_mag_g > proj_peak_deceleration[0]:
        1341:             proj_peak_deceleration[0] = accel_mag_g
        ```
    *   **Taichi GPU (`src/kevlargrid/solver/taichi_solver.py` lines 702-705):**
        ```python
        702:             accel_mag = self.proj_accel[None].norm()
        703:             accel_mag_g = accel_mag / 9.80665
        704:             if accel_mag_g > self.peak_deceleration_g[None]:
        705:                 self.peak_deceleration_g[None] = accel_mag_g
        ```
*   The script `run_benchmark_8.py` does not retrieve or pass references to these solver-tracked variables, discarding the true physics-based instantaneous peak values.

### C. Redundant Host-side CPU Calculations
In `benchmarks/benchmark_8/run_benchmark_8.py` (lines 261-269), node kinetic energies, spring strain energies, and peak strains are recalculating on the host CPU:
```python
261:         ke_nodes = 0.5 * np.sum(grid.masses * np.sum(vel**2, axis=1))
262:         
263:         p1 = pos[grid.springs[:, 0]]
264:         p2 = pos[grid.springs[:, 1]]
265:         lens = np.sqrt(np.sum((p2 - p1)**2, axis=1))
266:         strains = (lens - grid.rest_lengths) / grid.rest_lengths
267:         strains_eff = np.where(grid.tension_only & (strains < 0.0), 0.0, strains)
268:         se_springs = np.sum(0.5 * grid.stiffnesses * (strains_eff * grid.rest_lengths)**2)
```
*   This introduces massive data transfer bottlenecks (forcing full position/velocity array downloads from the GPU to the host) and duplicates mathematics that the JIT loops have already calculated.

### D. Missing Calibration Loop
In `benchmarks/benchmark_8/run_benchmark_8.py` (lines 340-373), the script executes Cases A, B, and C exactly once using hardcoded parameters:
```python
349:     case_a = run_case(450.0, "A", args.backend)
350:     case_b = run_case(503.0, "B", args.backend)
351:     case_c = run_case(550.0, "C", args.backend)
```
*   There is no optimization loop to sweep or tune the contact penalty stiffness $k_{\text{penalty}}$ or inter-ply friction $\mu_s$ to match the Jonas-Laval curve and the experimental $V_{50} = 503 \text{ m/s}$ ballistic limit.

---

## 2. Logic Chain

1. **State Resetting**: If progressive element damage, inter-ply contact forces, and frictional energy dissipation are reset to zero at every 20-step interval, the fabric fails to degrade physically, and contact damping is artificially removed. This prevents energy conservation within the $\le 2.0\%$ limit and makes the material artificially resistant to perforation.
2. **Deceleration Metrics**: High-strength Kevlar fabric deceleration has a very high peak when the projectile first initiates contact and stress waves propagate. Averaging this over 20 steps underrepresents the peak deceleration by up to an order of magnitude. This telemetry must come from the instantaneous JIT telemetry.
3. **Data Overhead**: Redundant host-side loop calculations undermine the acceleration of the Taichi solver because full array copies from GPU to host must be synchronized at every chunk boundary.
4. **Calibration Sweeps**: Because physical systems are sensitive to parameters ($k_{\text{penalty}}$ and $\mu_s$), we must implement an automated search loop to sweep through parameters to satisfy Case A (arrest), Case B (marginal perforation), and Case C (exit at $\approx 220 \text{ m/s}$), while enforcing the $2.0\%$ energy drift limit.

---

## 3. Caveats

*   **Read-Only Investigations**: In compliance with the explorer role constraints, I did not make any code changes to `run_benchmark_8.py` or the solver codebase. The recommendations below must be implemented by the implementation agent.
*   **Solver Interface Changes**: In Numba, `proj_peak_deceleration` can be mutated by reference since it is passed as a Numpy array. In Taichi, the global `_SOLVER_CACHE` object must be queried to retrieve the peak deceleration value, or the loop signature must be updated to return it.

---

## 4. Conclusion

I recommend the following concrete remediation strategy for the implementation phase:

### A. Propagate States Across Chunk Iterations
In `run_case` in `run_benchmark_8.py`, initialize and update persistent state arrays on the host CPU, passing them to both solver backend loops at each step:
*   Allocate `grid_damage = np.zeros(grid.n_springs, dtype=np.float64)` before the chunk loop.
*   Maintain `contact_energy = 0.0` and `friction_dissipated = 0.0`.
*   Pass them in the function call:
    ```python
    # For Numba:
    ... = fused_leapfrog_loop(
        ...,
        grid_damage=grid_damage,
        contact_energy_init=contact_energy,
        friction_dissipated_init=friction_dissipated,
        ...
    )
    # For Taichi:
    ... = taichi_leapfrog_loop(
        ...,
        grid_damage=grid_damage,
        contact_energy_init=contact_energy,
        friction_dissipated_init=friction_dissipated,
        ...
    )
    ```

### B. Track Running Instantaneous Peak Deceleration
*   **Numba**: Initialize a single-element array `peak_decel_arr = np.zeros(1, dtype=np.float64)` and pass it as `proj_peak_deceleration=peak_decel_arr` to `fused_leapfrog_loop`. Access `peak_decel_arr[0]` at the end of the simulation.
*   **Taichi**: Extract the peak deceleration directly from the Taichi solver:
    ```python
    from kevlargrid.solver.taichi_solver import _SOLVER_CACHE
    if _SOLVER_CACHE is not None:
        peak_deceleration = float(_SOLVER_CACHE.peak_deceleration_g[None])
    ```

### C. Direct History Harvesting
*   Stop host-side recalculation of energies and strains. Unpack the historical arrays returned by the solvers (`h_ke`, `h_se`, `h_proj_ke`, `hist_time`, `hist_peak_strain`) and accumulate them by extending host-side lists (`hist_ke.extend(h_ke.tolist())`, etc.).

### D. Parameter Calibration Loop
Implement an automated parameter optimization loop in `main()` using a grid search or coordinate descent over:
*   $k_{\text{penalty}} \in [1.0 \times 10^5, 2.0 \times 10^6]$
*   $\mu_s \in [0.18, 0.22]$
*   $\text{cfl\_factor} \in [0.1, 0.2]$
*   **Objective Function**:
    Minimize $\Phi = (V_{r, A})^2 + (V_{r, B})^2 + (V_{r, C} - 220)^2$ subject to $\text{Energy Drift} \le 2.0\%$.
*   Once optimal parameters are found, write the dynamic metrics to `results.json` and generate the dynamic plots and reports.

---

## 5. Verification Method

1.  **Run the script**:
    ```bash
    python benchmarks/benchmark_8/run_benchmark_8.py --backend numba
    ```
2.  **Verify files**:
    *   Confirm `benchmarks/benchmark_8/results.json` contains dynamic numbers (e.g. `energy_drift_pct` under `2.0`, `residual_velocity` values matching limits).
    *   Confirm `validation_plot.png` contains a dynamic Matplotlib-generated Jonas-Laval curve with Case A, B, and C points.
    *   Confirm `validation_report.pdf` is generated as a valid PDF with dynamic telemetry values.
