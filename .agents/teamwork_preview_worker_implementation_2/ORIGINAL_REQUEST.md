## 2026-06-27T04:15:32Z
Please perform the following implementation, calibration, and reporting tasks for Benchmark 8:

1. **Solver Code Improvements**:
   Modify the solver files (under `src/kevlargrid/solver/` and `src/kevlargrid/io/` if needed) to:
   - Implement velocity-regularized Coulomb friction for inter-ply and projectile contact. Friction coefficient must be tunable ($\mu_s \ge 0.18$). Track and accumulate frictional energy dissipation and include it in the total energy balance.
   - Fix energy conservation gaps: track stiffness damping dissipation energy, allow compressive strain energy for non-tension-only diagonal springs, align JIT damage energy formulations, and add failure energy tracking to the Python fallback.
   - Correct the Taichi backend wrapper to apply Bazant strain regularization just like Numba.

2. **Benchmark 8 Setup & Runner**:
   Create a dedicated script `benchmarks/benchmark_8/run_benchmark_8.py` that:
   - Sets up the 13-ply dry Kevlar 29 Style 713 model (areal density: 475 g/m^2, density: 1,440 kg/m^3, longitudinal modulus: 70.5 GPa, failure strain: 3.6% - 4.0%).
   - Patch dimensions: 250 mm x 250 mm, clamped boundaries, rigid 17-grain FSP (1.10g, 5.46mm).
   - Mesh resolution: at least 3-4 nodes spanning the projectile diameter ($dx \approx 1.82$ mm).
   - Separates layers with a minor gap (0.1 mm) and distinct node IDs.
   - Executes Case A (Vi = 450 m/s), Case B (Vi = 503 m/s), and Case C (Vi = 550 m/s).
   - Performs a programmatic parameter calibration loop over contact stiffness, friction coefficient $\mu_s \ge 0.18$, and CFL safety factor to ensure Case A arrests ($Vr = 0$), Case B barely perforates ($Vr < 25$), Case C exits at $220 \pm 20$ m/s, and energy drift is $\le 2\%$.
   - Automatically saves raw validation data in `results.json`, plots the residual velocity validation curve (with Lambert-Jonas fit) to `validation_plot.png`, and compiles a validation report `validation_report.pdf` at `benchmarks/benchmark_8/`.

3. **Verify Compliance**:
   - Run the unit tests (`pytest tests/`) to ensure no regressions.
   - Run benchmarks to verify stability.
   - Document your changes and results in `handoff.md` under your working directory, including passing build/test command logs and verification of output layouts.

Read PROJECT.md and all guidance files before starting. Send a message to the parent once complete.
