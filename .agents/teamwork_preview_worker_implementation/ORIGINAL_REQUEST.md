## 2026-07-10T03:00:46Z
You are teamwork_preview_worker. Your working directory is `/Users/bennames/Developer/VibeDynaLITE/.agents/teamwork_preview_worker_implementation/`.
Your mission is to implement the following changes in VibeDynaLITE:
1. Fix the configuration lookup mismatch in `src/kevlargrid/gui/app.py`:
   Line 945: Change `cfg["material"].get("material_name", "")` to `cfg["material"].get("name", "")`.
2. Fix `src/kevlargrid/gui/viewport3d.py` reset method:
   Ensure that if `structure_type == "metallic_sheet"`, `self.n_nodes_per_layer` is set to `len(grid.nodes) // n_plies` (to prevent invalid layer visibility check/culling).
3. Fix the stable timestep (CFL) overestimation in `src/kevlargrid/solver/fused.py`:
   Line 2028: Change `omega_spring = sqrt(2.0 * k_0 / mass_min)` to `omega_spring = sqrt(4.0 * k_0 / mass_min)`.
4. Fix the crossover and energy bookkeeping issues in `src/kevlargrid/solver/fused.py` cohesive forces loop:
   - For each tiebreak spring, compute the relative centroid direction `dx_c = C1[0] - C0[0]`, etc. (where `e0 = n0 // 4`, `e1 = n1 // 4`, and `C0`, `C1` are centroids of the elements computed from the current node `positions`).
   - Check if they are separating: `is_tension = (dx_s * dx_c + dy_s * dy_c + dz_s * dz_c) >= 0.0`.
   - If `is_tension`, execute the damage updating logic. Force magnitude is `(1.0 - d) * k_0 * delta`.
   - If not `is_tension` (compression/penetration), bypass damage accumulation (do not update damage), and apply penalty force with undamaged stiffness: `f_mag = k_0 * delta`.
   - If a spring fails in this step (`d >= 1.0`), set `spring_failed[i] = 1`, and add the remaining energy `0.5 * (1.0 - d_old) * k_0 * delta * delta` to `failure_dissipated` (to avoid energy drop).
5. Run local tests to verify your implementation:
   - Run `pytest tests/unit/test_metallic_sheet.py` and `pytest tests/gui/test_config_roundtrip.py`.
   - Run `python scratch/test_czm_simulation.py` and verify that the simulation runs to completion (100 steps) and energy drift is small (<2.0%).

MANDATORY INTEGRITY WARNING: DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Please document your code changes in `/Users/bennames/Developer/VibeDynaLITE/.agents/teamwork_preview_worker_implementation/changes.md` and write a final handoff report to `/Users/bennames/Developer/VibeDynaLITE/.agents/teamwork_preview_worker_implementation/handoff.md` detailing the test/command output and verifying that everything builds and passes.
After you finish, send a message to recipient "parent" (conversation ID: 26d6399a-b329-4b4e-a3c5-c12ca7308bc3) summarizing your work.
