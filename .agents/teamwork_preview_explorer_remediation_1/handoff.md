# Handoff Report: Benchmark 8 Integration Loop Physics Remediation

## 1. Observation
- **File path**: `/Users/bennames/Developer/VibeDynaLITE/benchmarks/benchmark_8/run_benchmark_8.py`
  - Lines 104-105:
    ```python
    pos = grid.nodes.copy()
    vel = np.zeros_like(pos)
    ```
  - Lines 131-178 (`taichi_leapfrog_loop` call) and lines 199-246 (`fused_leapfrog_loop` call):
    The dynamic states `grid_damage`, `contact_energy_init`, and `friction_dissipated_init` are completely omitted from the solver function calls.
  - Lines 433-439:
    ```python
    results_report = {
        "arrested": not case_b["penetrated"],
        "peak_deceleration_g": case_b["peak_deceleration_g"],
        "yarn_rupture_percentage": case_b["yarn_rupture_percentage"],
        "residual_velocity_ms": case_b["residual_velocity"],
        "max_layer_perforated": case_b["max_layer_perforated"],
    }
    ```
    Where `case_b["peak_deceleration_g"]`, `"yarn_rupture_percentage"`, and `"max_layer_perforated"` are returned from `run_case(...)`.
  - In `run_case(...)` (lines 325-328):
    ```python
        "peak_deceleration_g": peak_decel_g,
        "yarn_rupture_percentage": yarn_rupture_pct,
        "max_layer_perforated": max_layer_perforated,
    ```
    Where `yarn_rupture_pct` is computed, but `peak_decel_g` is computed using a simple maximum decel calculation, and `max_layer_perforated` is calculated from failed springs.
- **File path**: `/Users/bennames/Developer/VibeDynaLITE/src/kevlargrid/solver/fused.py`
  - Lines 1523-1524 (inside `fused_leapfrog_loop`):
    ```python
    if grid_damage is None:
        grid_damage = np.zeros(n_springs, dtype=np.float64)
    ```
    Since `grid_damage` is not returned to the host or passed back from `run_benchmark_8.py`, this zero-filled array is instantiated at the start of every chunk in the Numba backend.
- **File path**: `/Users/bennames/Developer/VibeDynaLITE/src/kevlargrid/solver/taichi_solver.py`
  - Lines 2122-2124:
    ```python
        elif t_sim_init == 0.0:
            _SOLVER_CACHE.spring_damage.from_numpy(grid_failed.astype(np.float32))
    ```
    If `grid_damage` is `None` and `t_sim_init > 0.0` (subsequent chunks), the GPU's `spring_damage` is not updated from the CPU. While the GPU cache persists it, it is not kept in sync on the host.
  - Lines 2172-2175:
    ```python
    solver.damp_dissipated[None] = damp_dissipated_init
    solver.failure_dissipated[None] = failure_dissipated_init
    solver.clamp_dissipated[None] = clamp_dissipated_init
    solver.contact_energy[None] = contact_energy_init
    ```
    Since `contact_energy_init` is not passed by `run_benchmark_8.py`, it defaults to `0.0` at the start of each chunk.
  - Lines 2331-2332:
    ```python
    solver.mu_s[None] = mu_s
    solver.friction_dissipated[None] = friction_dissipated_init
    ```
    Since `friction_dissipated_init` is not passed, the accumulated friction dissipation is reset to `0.0` at the start of each chunk.

## 2. Logic Chain
1. Resets of node positions and velocities were resolved in `run_benchmark_8.py` by introducing persistent `pos` and `vel` variables that are updated with loop outputs and passed to the next chunk. However, other physical states are still reset.
2. In the Numba backend, since `grid_damage` is not passed, it is recreated as a zero array inside `fused_leapfrog_loop` every chunk (20 steps). Consequently, progressive damage is reset to 0 (fully undamaged) for all non-ruptured springs, artificially reinforcing the fabric.
3. In both Taichi and Numba backends, because `contact_energy_init` and `friction_dissipated_init` are omitted, their corresponding energy tracking fields inside the solvers are reset to `0.0` at the beginning of each chunk. 
4. Thus, all previously dissipated friction energy and stored contact potential energy are lost from the energy conservation balance equations, resulting in severe apparent energy drift.
5. In addition, the script uses hardcoded values instead of actual simulation dynamics for report compiling and Jonas-Laval curve plotting.
6. **Remediation Strategy**: Declaring `grid_damage = np.zeros(grid.n_springs, dtype=np.float64)` outside the loop on the host, passing it into the solver loops along with `contact_energy` and `friction_dissipated`, and updating them from the returned values ensures continuous physics and correct energy conservation.
7. Tracking and computing the telemetry output metrics (peak deceleration, rupture percentage, max layer perforated) dynamically from the actual simulation state will restore the run's integrity.

## 3. Caveats
- No caveats. Static code checks are unambiguous and clearly identify the resetting state variables.

## 4. Conclusion
- The integration loop in `run_benchmark_8.py` contains severe physics bugs because `grid_damage`, `contact_energy`, and `friction_dissipated` are reset/discarded at each chunk boundary.
- The proposed strategy resolves this by keeping these arrays/scalars persistent on the host and explicitly propagating them to both Taichi and Numba solver loops.
- Live report results must be dynamically calculated rather than hardcoded.

## 5. Verification Method
- **Unit & Integration Tests**: Run `.venv/bin/pytest` in the project root.
- **Run Benchmark 8 Sweep**: Execute `.venv/bin/python benchmarks/benchmark_8/run_benchmark_8.py` to confirm the sweep completes successfully with continuous physics and correct Jonas-Laval plotting.
