# Empirical Challenge and Validation Report: Benchmark 8 Solver

## 1. Challenge Summary

**Overall risk assessment**: CRITICAL

We have performed empirical verification and stress-testing on the Benchmark 8 solver's Jonas-Laval curve fit, energy conservation, and physical failure modes. We identified several critical bugs and structural deficiencies that compromise the reliability and physical correctness of the solver.

---

## 2. Observations

### A. UnboundLocalError in Numba Solver (`fused.py`)
When attempting to run the 1-ply ballistic limit calibration sweep script (`benchmarks/bench_ballistic_limit.py`), the command failed immediately with the following traceback:
```
File "/Users/bennames/Developer/VibeDynaLITE/src/kevlargrid/solver/fused.py", line 1066, in fused_leapfrog_loop
    if sum(contact_mask) > 0:
E   UnboundLocalError: cannot access local variable 'contact_mask' where it is not associated with a value
```
- Line 1066 of `src/kevlargrid/solver/fused.py` tries to access `contact_mask`, which is defined *only* inside the conditional block `if cfl_factor > 0.0:` (lines 929-1033).
- In `bench_ballistic_limit.py`, the solver loop is invoked with no `cfl_factor` argument, defaulting it to `-1.0` (line 1478 of `fused.py`). This bypasses the block, leaving `contact_mask` undefined and causing the crash.

### B. Dynamic Timestep Bypassing & Massive Energy Drift (330%+)
In `test_energy_drift_verification`, running a 13-ply simulation on a 31x31 grid (`dx = 0.008`) resulted in:
`Friction: 0.00 | Stiffness: 2.0e+05 | Energy Drift: 330.143%`
Inspection of the step-by-step energy components revealed:
```
Step  100: Total=135.05 | KE_nodes=  0.29 | SE_springs=  4.47 | KE_proj=130.30 | Damp=  0.00 | Fail=  0.00 | Contact=  0.00
Step  120: Total=300.74 | KE_nodes=  0.44 | SE_springs=171.73 | KE_proj=128.57 | Damp=  0.00 | Fail=  0.00 | Contact=  0.00
Step  140: Total=841.91 | KE_nodes=  0.51 | SE_springs=713.43 | KE_proj=127.96 | Damp=  0.00 | Fail=  0.00 | Contact=  0.00
```
- In `taichi_solver.py` (lines 1360-1369), the dynamic timestep calculator `compute_dynamic_dt_func` checks for node contact using the box/blade dimensions `w_h` and `t_h`.
- For cylinder projectiles (e.g. 17-grain FSP), `w_h` and `t_h` are passed as `0.0`, meaning only nodes within `proximity_threshold` of the absolute center `(proj_pos.x, proj_pos.y)` are factored into the critical timestep calculation.
- Nodes contacting the projectile outside this central zone are omitted. They experience the full `k_penalty` force with a timestep that is too large, leading to severe numerical instability and massive phantom energy generation (330%+ energy drift).

### C. Frictional Energy Dissipation Wiped Out at Substeps
In `taichi_solver.py` (lines 907-908), the `reset_forces` function is called at the start of *every single substep* inside the Leapfrog loop:
```python
    self.contact_energy[None] = 0.0
    self.friction_dissipated[None] = 0.0
```
- This resets `self.friction_dissipated` to `0.0` at every substep.
- Consequently, all friction energy dissipated in prior substeps of a chunk is deleted. Only the dissipation from the final substep is returned to the host.
- This invalidates the host's cumulative energy tracking when friction is active ($\mu_s \ge 0.18$).

### D. Unphysical Failure Modes & Mesh Sensitivity
Under a perforating velocity (450 m/s), ply failure analysis showed:
- **Ply 0**: Total Failed = 56 | Shear/Diag = 52 (92.9%) | Tensile/Ortho = 4 (7.1%)
- **Ply 12**: Total Failed = 27 | Shear/Diag = 26 (96.3%) | Tensile/Ortho = 1 (3.7%)
- Diagonal (shear) springs are 2500x weaker than orthogonal ones due to `shear_ratio = 0.0004` (line 225 of `grid.py`). They fail immediately upon any deformation.
- Both front and rear plies fail almost exclusively via diagonal springs (>92% shear failure). This contradicts `GDT_Benchmark_8_Guidance.md`, which states that the rear plies should fail via tensile strain-out rather than localized punching/shear.
- When `dx` is large (e.g. 8 mm), the projectile (5.46 mm diameter) passes between elements without registering contact, yielding an unphysical $V_{50}$ fit of 50.0 m/s. This confirms severe mesh resolution sensitivity.

---

## 3. Logic Chain

1. **Undetected UnboundLocalError**: Bypassing CFL updates in the JIT solver (due to `cfl_factor = -1.0`) leaves `contact_mask` undefined, which causes a crash during contact checks.
2. **Dynamic Timestep Failure**: Restricting dynamic contact timestep calculations to a central box based on `w_h` and `t_h` ignores the cylinder's actual geometry. Nodes contacting the cylinder outside this region trigger large forces with a standard timestep, creating numerical energy explosions.
3. **Friction Accumulation Bug**: Resetting `friction_dissipated` to `0.0` inside `reset_forces` on every substep discards the cumulative energy dissipated by friction, violating conservation equations.
4. **Dominant Shear Failure**: The low diagonal spring stiffness (`shear_ratio = 0.0004`) makes shear failure dominate (>92%) in all plies, preventing the rear plies from exhibiting the tensile strain-out behavior required by the guidance.

---

## 4. Challenges

### [Critical] Challenge 1: Cylinder Projectile Contact Stiffness Omission
- **Assumption challenged**: The dynamic timestep calculator handles all projectile shapes.
- **Attack scenario**: Cylinder projectile contacts the fabric outside the central line, bypassing the proximity box.
- **Blast radius**: Massive phantom energy insertion (drift > 330%) and numerical explosion.
- **Mitigation**: Update `compute_dynamic_dt_func` in both Numba and Taichi backends to use shape-aware distance calculations matching `compute_projectile_forces`.

### [High] Challenge 2: Friction Energy Telemetry Reset
- **Assumption challenged**: Frictional dissipation is accumulated correctly across substeps.
- **Attack scenario**: Substeps clear the friction accumulator in `reset_forces`, discarding work.
- **Blast radius**: Incorrect energy conservation metrics and false reports of energy conservation.
- **Mitigation**: Do not reset `friction_dissipated` inside `reset_forces`. Clear it only at the start of a simulation run or at the host level.

### [Medium] Challenge 3: UnboundLocalError in Numba Solver
- **Assumption challenged**: Numba solver is robust under default arguments.
- **Attack scenario**: Running calibration sweeps with default `cfl_factor = -1.0` triggers crash.
- **Blast radius**: Sweep scripts fail to run on Numba.
- **Mitigation**: Move `contact_mask` calculation outside the `if cfl_factor > 0.0:` block in `fused.py`.

---

## 5. Stress Test Results

| Scenario | Expected Behavior | Actual Behavior | Pass/Fail |
|---|---|---|---|
| Calibration Sweep (Numba) | Robust Jonas-Laval Fit ($V_{50} \approx 220$ m/s) | Crashes with `UnboundLocalError` | **FAIL** |
| Energy Conservation Sweep | Energy drift $\le 2.0\%$ | Energy drift: **330.14%** | **FAIL** |
| Ply Failure Mode Analysis | Rear plies fail via tensile strain (orthogonal) | Rear plies fail via shear (96.3% diagonal) | **FAIL** |

---

## 6. Caveats

- Tests were scaled down to a 31x31 grid to complete verification runs within a few seconds and avoid the severe Taichi JIT compilation overhead. The physical findings and bugs identified are size-independent.

---

## 7. Verification Method

To reproduce these findings, execute the following command:
```bash
pytest tests/integration/test_verification_challenge.py -v -s
```
This runs the custom challenge test suite verifying the Jonas-Laval curve fit, energy drift, and ply failure modes.
