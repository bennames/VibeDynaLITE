## 2026-06-28T09:57:36Z
You are the Final Solver Sweep and Verification Worker. Your working directory is `/Users/bennames/Developer/VibeDynaLITE/.agents/worker_final_sweep`.

Objective:
You must tune the physical parameters in `run_benchmark_8.py` to ensure Case A arrests and other targets are met, run the validation sweep, run the tests, and commit and push.

Steps:
1. Examine `/Users/bennames/Developer/VibeDynaLITE/benchmarks/benchmark_8/run_benchmark_8.py`.
2. Change the `shear_ratio` in `material_kev29` (line 142) from `0.0004` to `0.002` (which is the actual standard Kevlar 29 material value in `library.py`). This restores the correct fabric diagonal stiffness to distribute transverse wave loads properly.
3. Verify that `cfl_factor = 0.1` and `k_penalty = 2.0e6` are set in `run_benchmark_8.py`.
4. Run Case A (450 m/s) with `TAICHI_FORCE_CPU=1` to check if the projectile is successfully arrested (residual velocity = 0) and the system energy conservation drift is <= 2%. You can do this by executing:
   ```bash
   TAICHI_FORCE_CPU=1 .venv/bin/python benchmarks/benchmark_8/run_benchmark_8.py --backend taichi
   ```
   (Wait, this will run all three cases. If Case A arrests and meets the targets, let the sweep complete. If not, check other parameters or tune `k_penalty` slightly).
5. Verify that the sweep successfully completes and generates:
   - `benchmarks/benchmark_8/results.json`
   - `benchmarks/benchmark_8/validation_plot.png`
   - `benchmarks/benchmark_8/validation_report.pdf`
6. Inspect `benchmarks/benchmark_8/results.json` and confirm:
   - Case A (450 m/s) is arrested (residual velocity = 0).
   - Case B (503 m/s) residual velocity is < 25 m/s.
   - Case C (550 m/s) residual velocity is 220 +/- 20 m/s.
   - Energy conservation drift is <= 2% for each case.
7. Verify file signatures for PNG and PDF headers.
8. Run the rest of the unit test suite to confirm all other tests pass:
   ```bash
   .venv/bin/pytest tests/ -k "not test_run_benchmark_8"
   ```
9. Git commit and push all modified files, source files, and results. If push fails or times out, make sure the commit is saved locally and report this.

MANDATORY INTEGRITY WARNING:
> DO NOT CHEAT. All implementations must be genuine. DO NOT
> hardcode test results, create dummy/facade implementations, or
> circumvent the intended task. A Forensic Auditor will independently
> verify your work. Integrity violations WILL be detected and your
> work WILL be rejected.

Please write a detailed handoff report to `/Users/bennames/Developer/VibeDynaLITE/.agents/worker_final_sweep/handoff.md` and message the parent conversation ID when done.
