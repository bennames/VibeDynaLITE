# Synthesis of Codebase Exploration for Benchmark 8

## Consensus Findings
1. **Solver Backend**: Explicit dynamics solver uses Leapfrog Verlet integration, implemented in `src/kevlargrid/solver/fused.py` (Numba CPU) and `src/kevlargrid/solver/taichi_solver.py` (Taichi GPU).
2. **Mesh Setup (13 Plies)**: Handled by `generate_rectangular_grid` in Checkout Mode B (when `t_ply` is set and `n_plies > 1`). Stacks layers in Z-axis and offsets node indices so they are decoupled. Clamping works by setting `boundary_mask = 1` for boundary nodes.
3. **Missing Contact Physics (Friction)**: Friction is entirely missing. Inter-ply contact is Z-axis only; projectile contact is normal-only.
4. **Energy Conservation Drift**: Energy calculations have gaps:
   - Stiffness-proportional damping energy is untracked.
   - Compressive strain energy is zeroed for all springs, though diagonal shear springs are not tension-only.
   - Numba has a discrepancy in progressive damage energy formulation.
   - Python fallback has no failure energy tracking.
5. **Taichi Bazant Regularization**: Bypassed in the Taichi backend wrapper.

## Calibration and Runner Plan
- Target parameters: $k_{\text{penalty}}$, friction $\mu_s \ge 0.18$, and CFL safety factor.
- Objective function: Minimizes error on Case A, B, C residual velocities while penalizing energy conservation drift $>2\%$.
- Output requirements: `results.json`, `validation_plot.png`, and `validation_report.pdf` at `benchmarks/benchmark_8/`.
