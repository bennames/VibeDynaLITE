## 2026-06-27T03:44:00Z
Please perform the following implementation, calibration, and reporting tasks for Benchmark 8:

1. **Solver Code Improvements**:
   Modify the solver files (under `src/kevlargrid/solver/` and `src/kevlargrid/io/` if needed) to:
   - Implement velocity-regularized Coulomb friction for inter-ply and projectile contact. Friction coefficient must be tunable ($\mu_s \ge 0.18$). Track and accumulate frictional energy dissipation and include it in the total energy balance.
   - Fix energy conservation gaps: track stiffness damping dissipation energy, allow compressive strain energy for non-tension-only diagonal springs, align JIT damage energy formulations, and add failure energy tracking to the Python fallback.
   - Correct the Taichi backend wrapper to apply Bazant strain regularization just like Numba.

2. **Benchmark 8 Setup & Runner**:
   Create `benchmarks/benchmark_8/run_benchmark_8.py` to set up 13-ply dry Kevlar 29 Style 713 with clamped boundaries, rigid 17-grain FSP (5.46mm, 1.10g cylinder), layer gaps (0.1mm), and resolution ($dx \approx 1.82\text{ mm}$).
   - Execute Cases A (450 m/s), B (503 m/s), C (550 m/s).
   - Calibrate parameters ($\mu_s \ge 0.18$, penalty stiffness, CFL factor) to meet residual velocity exit bounds (Case A stops, Case B barely perforates $<25$ m/s, Case C exits at $220 \pm 20$ m/s, energy drift $\le 2\%$).
   - Save validation data in `results.json`, plot residual velocity curve (with Lambert-Jonas fit) to `validation_plot.png`, and compile validation report `validation_report.pdf` at `benchmarks/benchmark_8/`.
