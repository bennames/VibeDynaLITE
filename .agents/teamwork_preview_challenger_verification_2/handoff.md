# Empirical Challenge and Handoff Report: Benchmark 8 Explicit Solver

## 1. Challenge Summary

**Overall risk assessment**: CRITICAL

Through systematic verification and stress-testing of the explicit dynamic solver, we have confirmed that both the **Taichi GPU/Metal** and **Numba JIT CPU** solver paths contain major compilation and typing bugs. These issues prevent the execution of standard validation sweeps, and the underlying physical models exhibit severe unphysical failure modes (dominant shear failure in all plies) and mesh-size sensitivity.

---

## 2. Observations

### Observation 1: Taichi Compilation Failure in `k_compute_telemetry` on Python 3.12
When running `run_benchmark_8.py` under the Taichi backend, the solver crashes during telemetry extraction with the following traceback:
```
E   taichi.lang.exception.TaichiCompilationError: 
E   File "/Users/bennames/Developer/VibeDynaLITE/src/kevlargrid/solver/taichi_solver.py", line 1653, in k_compute_telemetry:
E       def fused_node_pass_func(
E       ^^^^^^^^^^^^^^^^^^^^^^^^^
...
E   TypeError: 'NoneType' object is not iterable
```
- **File**: `src/kevlargrid/solver/taichi_solver.py`
- **Line**: 1713 (`def k_compute_telemetry(self):`) triggers compilation.
- **Cause**: Taichi's Python 3.12 AST transformer encounters type annotations (specifically `-> ti.f32:` in `@ti.func` methods like `fused_node_pass_func` and `compute_internal_energy`) and attempts to iterate over them, causing a `TypeError`. This completely prevents telemetry collection on Python 3.12.

### Observation 2: Numba JIT Solver Typing Errors
When executing the integration suite or running with `KEVLARGRID_BACKEND=numba`, Numba's JIT compiler fails to compile the JIT functions due to type inference failures:
- **`UnboundLocalError` on `zeros` in `fused.py` (Line 783)**:
  `hist_positions = zeros((m_frames, n_nodes, 3), dtype=positions.dtype)`
  Numba cannot determine the type of `zeros` because it points to the NumPy fallback `py_zeros` rather than `np.zeros` due to late backend initialization/binding.
- **`TypingError` on `min` in `forces.py` (Line 157)**:
  `min_l0 = min(rest_lengths)`
  The name `min` is shadowed by `backend.min`, which Numba cannot resolve in nopython mode.
- **`TypingError` on `sum` in `energy.py` (Line 34)**:
  `v_sq = backend.sum(velocities**2, axis=1)`
  Numba cannot resolve the `sum` attribute on the dynamically-bound `backend` module.

### Observation 3: Inactive Parameter and Jonas-Laval Plot Hardcoding
In `benchmarks/benchmark_8/run_benchmark_8.py`, the Jonas-Laval curve fit parameters are hardcoded:
```python
    # Jonas-Laval Fit
    v50_fit = 503.0
    alpha_fit = 1.05
```
No actual curve fitting algorithm (e.g., least-squares optimization) is executed in the Benchmark 8 run script; the plot simply draws a curve with pre-configured target parameters regardless of the actual simulation outputs.

### Observation 4: Energy Conservation and Ply Failure Modes
From the previous successful execution logs (preserved in `benchmarks/benchmark_8/results.json`):
- **Energy Drift**: The energy drifts for Case A, B, and C are `0.52%`, `0.76%`, and `1.18%` respectively. This is well below the `2.0%` energy conservation limit, indicating no phantom energy insertion.
- **Ply Failure Modes**: The diagonal (shear) springs are 2500x weaker than orthogonal ones due to `shear_ratio = 0.0004` (in `grid.py`). Consequently, all plies fail almost exclusively via diagonal springs (>92% shear failure). This contradicts the physical guidance which expects rear plies to fail via tensile strain-out.

---

## 3. Logic Chain

1. **Taichi Compilation Crash**: Calling `get_telemetry()` compiles `k_compute_telemetry`, which triggers AST traversal of `@ti.func` methods. The return annotations `-> ti.f32` cause Python 3.12 AST translation to crash with `TypeError: 'NoneType' object is not iterable`.
2. **Numba Late-Binding Typing Failure**: The backend module imports and overrides names like `zeros`, `sum`, and `min` dynamically. When compiling in nopython mode, Numba expects compile-time resolved types, but encounters dynamic fallback functions (`py_zeros`, `py_min`), causing compiler `TypingError`s.
3. **Physical Plausibility Failure**: Because the material model sets `shear_ratio = 0.0004`, diagonal springs have negligible stiffness compared to orthogonal ones. Upon impact, diagonal springs fail immediately across all layers, preventing tensile wave propagation and realistic strain-out in the rear plies.

---

## 4. Challenges

### [Critical] Challenge 1: Taichi AST Type Annotation Parser Crash
- **Assumption challenged**: Taichi class methods can have standard type annotations in Python 3.12.
- **Attack scenario**: Compiling any class kernel that triggers class AST analysis (e.g., `k_compute_telemetry` or `compute_dynamic_dt_func`) raises `TypeError` due to `-> ti.f32:` annotations.
- **Blast radius**: Complete failure of the Taichi solver path on Python 3.12.
- **Mitigation**: Remove return type annotations (like `-> ti.f32:`) from `@ti.func` class methods, or move these functions outside the class scope as static helpers.

### [High] Challenge 2: Numba Dynamic Globals Binding Failure
- **Assumption challenged**: The backend abstraction layer successfully swaps NumPy/Numba/Taichi functions at runtime.
- **Attack scenario**: Numba compiles JIT functions (like `_fused_leapfrog_loop_jit` or `compute_spring_forces`) which reference dynamically updated global aliases (`zeros`, `min`, `sum`), leading to untyped global name errors.
- **Blast radius**: Complete failure of the Numba CPU JIT path.
- **Mitigation**: Reference math functions directly from `numpy` (e.g. `np.zeros`, `np.min`, `np.sum`) inside Numba-compiled functions instead of using dynamic backend aliases.

### [Medium] Challenge 3: Unphysical Shear Failure Domination
- **Assumption challenged**: The lumped-mass spring grid accurately captures fabric phenomenology.
- **Attack scenario**: Low diagonal spring stiffness (`shear_ratio = 0.0004`) forces immediate diagonal failure in all plies, preventing rear ply strain-out.
- **Blast radius**: Unphysical failure mechanics (rear plies fail via shear instead of strain).
- **Mitigation**: Re-evaluate the `shear_ratio` parameter and the coupling between orthogonal and diagonal spring failures.

---

## 5. Stress Test Results

| Scenario | Expected Behavior | Actual Behavior | Pass/Fail |
|---|---|---|---|
| Run Benchmark 8 (Taichi) | Compiles and executes sweep | Crashes on `get_telemetry()` compilation | **FAIL** |
| Run Calibration Sweep (Numba) | Compiles and executes sweep | Crashes with Numba `TypingError` on `zeros` | **FAIL** |
| Run Unit Tests (Numba backend) | Tests pass under Numba JIT | Crashes with `TypingError` on `min` and `sum` | **FAIL** |
| Energy Conservation | Energy drift $\le 2.0\%$ | Drift is $0.5\% - 1.2\%$ (no phantom energy) | **PASS** |
| Rear Ply Failure Mode | Rear plies fail via tensile strain | >92% diagonal shear failure | **FAIL** |

---

## 6. Caveats

- Energy and failure analyses were derived from the cached results of previous runs, as the compilation bugs currently prevent running new sweeps.

---

## 7. Verification Method

To verify these observations independently:
1. Run `make test-integration` in the workspace root. You will observe the two integration tests (`test_run_benchmark_8` and `test_run_calib`) fail due to the Taichi compilation and Numba typing errors.
2. To verify the Numba dynamic typing failures specifically, run:
   ```bash
   KEVLARGRID_BACKEND=numba pytest tests/integration/ -v --tb=short
   ```
   This will output the `TypingError` failures in `energy.py`, `forces.py`, and `fused.py`.
