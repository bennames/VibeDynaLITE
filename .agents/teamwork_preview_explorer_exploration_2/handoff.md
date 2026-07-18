# Exploration Report — VibeDynaLITE Solver Physics

This report details the findings from an investigation of the VibeDynaLITE explicit dynamics solver, focusing on inter-ply/yarn contact forces and friction, energy conservation and logging, and element failure and deletion.

---

## 1. Observations

### A. Contact and Friction Modeling
1. **Inter-ply Contact**: The inter-ply contact force is computed in `src/kevlargrid/solver/forces.py` (lines 249–327) inside `compute_interply_contact_forces`. It uses a penalty stiffness formulation to resolve vertical interpenetration between corresponding nodes across adjacent plies:
   ```python
   # Lines 295–302
   z_n = positions[start_idx:end_idx, 2]
   z_n1 = positions[end_idx : end_idx + n_nodes_per_layer, 2]
   
   # Penetration depth using absolute distance formulation
   gap = np.abs(z_n - z_n1)
   penetration = t_ply - gap
   penetrating = penetration > 0.0
   ...
   # Force magnitude
   f_mag = where(penetrating, k_penalty * penetration, 0.0)
   direction = where(z_n > z_n1, 1.0, -1.0)
   
   forces_n = stack_z(f_mag * direction)
   forces_n1 = stack_z(-f_mag * direction)
   ```
2. **Tangent Friction**: There is no friction model (such as Coulomb dry friction or friction coefficient $\mu$) implemented for inter-ply contact. The contact force vector is purely vertical (along the Z-axis).
3. **Projectile Contact**: Projectile-to-mesh contact forces are computed in `src/kevlargrid/solver/fused.py` (lines 1068–1176) and `src/kevlargrid/solver/taichi_solver.py` (lines 953–1062). In both implementations, the force is calculated along the normal direction of the projectile surface:
   ```python
   # src/kevlargrid/solver/fused.py lines 1148-1156
   f_mag = k_penalty * delta + proj_c_damping * delta_dot
   if f_mag < 0.0:
       f_mag = 0.0
   ...
   F_contact = f_mag * n_world * node_scale_factor
   ```
   No tangential friction force resisting sliding along the projectile surface is modeled.
4. **Yarn-to-Yarn Contact**: There is no yarn-to-yarn (inter-yarn) contact detection or sliding friction computation modeled within a single ply. The mesh nodes are connected by orthogonal and diagonal springs, but do not interact through internal contact forces.

### B. Energy Calculations and Balance
1. **Kinetic and Strain Energy**: Node kinetic energy is computed in `src/kevlargrid/solver/energy.py` via `compute_kinetic_energy(velocities, masses) = sum(0.5 * masses * v^2)`. Elastic strain energy is computed via `compute_strain_energy` (line 70):
   ```python
   se_springs = 0.5 * eff_k * (backend.maximum(0.0, strains) * rest_lengths) ** 2
   ```
   Note that `backend.maximum(0.0, strains)` zeroes out compressive strain energy for all springs.
2. **Rayleigh Damping Dissipation**: Mass-proportional damping energy is computed in `src/kevlargrid/solver/fused.py` (line 1187) via:
   ```python
   p_mass_damp = sum(f_mass_damp * v_half)
   damp_dissipated += -p_mass_damp * dt
   ```
   However, stiffness-proportional Rayleigh damping dissipation (dependent on `rayleigh_beta`) is not accumulated or integrated in the timestep loop.
3. **Progressive Failure Dissipated Energy**: 
   - **Taichi JIT Backend**: Dissipated energy for damaging springs at intermediate peak strains is computed in `src/kevlargrid/solver/taichi_solver.py` (line 1182) via:
     ```python
     w_diss = (k * L0**2 / (6.0 * denom_safe)) * (x_peak**3 - x_onset**3)
     ```
   - **Numba JIT Backend**: Dissipated energy is computed in `src/kevlargrid/solver/fused.py` (line 572) via:
     ```python
     w_input = (k * L0**2 / 6.0) * (x**2 + x * x_onset + x_onset**2)
     se_actual = 0.5 * effective_k * (x * L0) ** 2
     total_diss += fracture_energy_multiplier * (w_input - se_actual)
     ```
   - **Python Fallback Backend**: Dissipated failure energy is completely ignored:
     ```python
     failure_dissipated = 0.0
     ```
4. **Other Dissipated Energy**: Potential energy of projectile-to-mesh contact is not computed or added to the total energy balance (only inter-ply contact energy `contact_energy` is added to total SE in the telemetry logging).

### C. Failure Criteria and Deletion
1. **Failure Strain & Scaling**: Nominal failure strain $\varepsilon_{fail}$ is checked in `check_failures` and `check_progressive_damage`. Bazant strain regularization is defined in `src/kevlargrid/solver/failure.py`:
   ```python
   def scale_failure_strain(epsilon_0: float, dx: float) -> float:
       if dx < 0.001:
           return epsilon_0 * np.sqrt(0.01 / dx)
       return epsilon_0
   ```
2. **Bazant Regularization Omission**: Bazant scaling is applied in the wrapper function of the Numba solver (`src/kevlargrid/solver/fused.py` line 1417), but is completely omitted in the Taichi solver (`src/kevlargrid/solver/taichi_solver.py` lines 2098–2105) where unscaled `failure_strain` is passed directly.
3. **Deactivation/Deletion**: When a spring fails, it is marked as `failed[i] = True`. Its effective stiffness `effective_k[i]` is set to `0.0`, resulting in zero force. Springs are never removed from the connectivity array.
4. **Lagrangian Inertial Drift**: Contact forces are scaled by the ratio of active connected springs to initial connected springs:
   ```python
   node_scale_factor = float(active_counts[i]) / float(node_initial_springs[i])
   ```
   If a node becomes completely detached (`active_counts[i] == 0`), `node_scale_factor` becomes `0.0`, zeroing out both projectile and inter-ply contact forces on that node.

---

## 2. Logic Chain

1. **Friction Absence**:
   - *Observation*: Grepping for "friction" in `src/` yielded no results, and `compute_interply_contact_forces` resolves penetration strictly along the Z-axis without any transverse component.
   - *Reasoning*: The solver is missing dry static and dynamic friction ($\mu \ge 0.18$) at both ply-to-ply interfaces and projectile-to-mesh interfaces.
   - *Conclusion*: A friction force model must be added to the contact force calculators.

2. **Compressive Energy Mismatch**:
   - *Observation*: `compute_strain_energy` zeroes out compressive strain energy for all springs, but diagonal shear springs are not `tension_only` in force computation.
   - *Reasoning*: Under compression, diagonal springs do work (carry force) but their strain energy is tracked as zero, causing energy conservation drift.
   - *Conclusion*: Compressive strain energy tracking must be allowed for springs that are not marked as `tension_only`.

3. **Stiffness Damping Dissipation Gap**:
   - *Observation*: Rayleigh damping forces are calculated using `rayleigh_beta * effective_k * v_proj`, but the dissipated energy is never integrated or added to `damp_dissipated`.
   - *Reasoning*: Energy dissipated by stiffness-proportional damping is lost from the energy balance, leading to energy conservation drift when $\beta > 0$.
   - *Conclusion*: Stiffness damping power must be integrated and accumulated.

4. **Numba Failure Energy Tracking Bug**:
   - *Observation*: Numba JIT uses `w_input = (k * L0**2 / 6.0) * (x**2 + x * x_onset + x_onset**2)` for intermediate strains, whereas Taichi uses `w_diss = (k * L0**2 / (6.0 * h)) * (x_peak**3 - x_onset**3)`.
   - *Reasoning*: The correct dissipated energy is $E_{dissipated}(x_{peak}) = E_{elastic} + W_{damage}(x_{peak}) - SE(x_{peak}) = \frac{k \, L_0^2}{6 \, h} (x_{peak}^3 - x_{onset}^3)$. Numba's formulation assumes a fictitious damage curve ending at $x$, making it physically incorrect for intermediate damage levels.
   - *Conclusion*: Numba's formulation must be corrected to match Taichi's formula.

5. **Python Fallback Failure Deactivation Bug**:
   - *Observation*: The Python fallback branch of `_fused_leapfrog_loop_jit` does not update `grid_failed` and sets `failure_dissipated = 0.0`.
   - *Reasoning*: Failed elements are never flagged as failed in Python fallback, and no failure energy dissipation is tracked.
   - *Conclusion*: Python fallback needs to update `grid_failed` and compute failure energy.

6. **Taichi Bazant Scaling Omission**:
   - *Observation*: `taichi_leapfrog_loop` receives the raw `failure_strain` instead of `failure_strain_eff`.
   - *Reasoning*: Taichi GPU simulations on fine meshes ($dx < 1$ mm) do not scale the failure strain, leading to mesh dependency.
   - *Conclusion*: Apply Bazant regularization inside the Taichi wrapper.

---

## 3. Caveats

- **No physical validation of friction coefficients**: Because friction is not implemented, we could not evaluate the physical response of the model under friction calibration.
- **Rayleigh beta usage**: If Rayleigh beta is set to 0.0 (e.g. only mass-proportional damping), the stiffness damping energy drift does not occur.

---

## 4. Conclusion

The VibeDynaLITE solver physics has several critical gaps:
1. **Friction** is entirely missing from both backends.
2. **Energy Conservation** drifts due to (a) untracked stiffness-proportional damping energy, (b) zeroed compressive energy of shear springs, (c) incorrect Numba failure energy tracking, and (d) untracked failure energy in Python fallback.
3. **Bazant Regularization** is bypassed in the Taichi backend.

Structured Recommendations for Implementation:
- **Friction**: Implement Coulomb static friction force $F_{fric} \le \mu F_N$ along the tangent plane of contact (both ply-to-ply sliding and projectile-to-mesh sliding).
- **Energy**: Integrate stiffness-damping power, include compressive strain energy for non-tension-only springs, and align JIT energy formulas.
- **Taichi Regularization**: Compute `failure_strain_eff` in the Taichi wrapper before passing it to the solver class.

---

## 5. Verification Method

- **Unit Tests**: Run `pytest tests/unit/ -v`.
- **Damping & Energy Check**: Run the energy conservation test `pytest tests/unit/test_energy.py`.
- **Stability Check**: Verify that running the solver under small $dx < 1$ mm yields scaled failure strains on both backends.
- **Validation**: Invalidation conditions would include any test failures or energy balance drift exceeding 2.0% post-implementation.
