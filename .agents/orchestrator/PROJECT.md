# Project: VibeDynaLITE Benchmark 8 Ballistic Limit Validation

## Architecture
- KevlarGrid Solver: JIT-compiled (Numba) explicit dynamic solver (`fused_leapfrog_loop` or similar).
- Material: Dry Kevlar 29 Style 713 (areal density 475 g/m^2, density 1440 kg/m^3, longitudinal elastic modulus 70.5 GPa, failure strain 3.6% - 4.0%).
- Target configuration: 13 plies, 250 mm x 250 mm square fabric patch, fully clamped edges.
- Projectile: Rigid 17-grain FSP (1.10g, 5.46mm diameter).
- Mesh resolution: At least 3-4 nodes spanning projectile diameter (mesh size <= 1.37 mm).
- Ply interaction: Separate layers separated by a minor gap (e.g. 0.1 mm), distinct nodes, inter-ply contact force with dry static friction (mu >= 0.18).

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Exploration & Architecture | Examine existing codebase solver (fused loop, multi-ply support, contact penalty, friction, energy tracking). | None | DONE |
| M2 | Benchmark 8 Model Setup | Define geometry setup for 13 plies, gap, clamped edges, and mesh resolution, plus rigid projectile representation. | M1 | IN_PROGRESS |
| M3 | Automated Validation Runner | Create the runner to execute Cases A (450 m/s), B (503 m/s), C (550 m/s), measuring residual velocities and energy conservation. | M2 | PLANNED |
| M4 | Solver Parameter Calibration | Tune parameters (contact stiffness, friction coefficient mu >= 0.18, CFL safety factor) programmatically if limits drift. | M3 | PLANNED |
| M5 | Automated Reporting | Generate results.json, Lambert-Jonas fit curve plot (PNG), and PDF validation report. | M4 | PLANNED |
| M6 | Integrity & Verification | Run Forensic Auditor for integrity verification; execute local test suite and physical benchmarks; commit and push, verify CI/CD. | M5 | PLANNED |

## Interface Contracts
- The benchmark setup and runner will run standalone under `benchmarks/benchmark_8/` and output `results.json`, `validation_report.pdf`, and `validation_plot.png`.

## Code Layout
- `benchmarks/benchmark_8/`: Benchmark 8 root directory.
- `benchmarks/benchmark_8/run_benchmark_8.py`: Main validation, runner, and calibration script.
- `benchmarks/benchmark_8/results.json`: Output data for simulated cases.
- `benchmarks/benchmark_8/validation_plot.png`: Residual velocity validation plot.
- `benchmarks/benchmark_8/validation_report.pdf`: Summary report.
