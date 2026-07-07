# Handoff Report: Codebase Exploration for Benchmark 8 Validation

## 1. Observation

During my investigation of the VibeDynaLITE codebase, I observed the following files, function signatures, and layouts:

### A. Benchmark Runners and Test Scripts
*   **Performance Scaling Benchmark:** In `benchmarks/bench_solver.py` (lines 29-181), the solver performance is profiled across different grid resolutions and architectures:
    ```python
    def run_benchmark(arch_name: str, size: int, mode: str, n_steps: int = 50) -> float
    ```
    This sweeps through CPU, GPU (Taichi), and Numba CPU backends, generating `benchmarks/results.json` and `benchmarks/performance_comparison.png`.
*   **Ballistic Limit Validation Sweep:** In `benchmarks/bench_ballistic_limit.py` (lines 21-117), strike velocities from 100 m/s to 450 m/s are swept against 1-ply Kevlar 29. Residual velocities are extracted and fitted using a grid search to the classical Jonas-Laval curve (lines 119-138):
    ```python
    def fit_jonas_laval(v_strike: np.ndarray, v_residual: np.ndarray) -> tuple[float, float]
    ```
    The validation results are plotted in `benchmarks/ballistic_limit_validation.png`.
*   **Physics Verification Tests:** In `tests/integration/test_physics_benchmarks.py`, a set of physical verification test cases exists:
    *   `test_cfl_stability_limit()`: Verifies stability threshold limits on perturbed grids (lines 25-269).
    *   `test_1d_stress_wave_propagation_and_reflection()`: Verifies wave speed $c = dx \sqrt{k/m}$ and boundary reflection doubling (lines 270-409).
    *   `test_smith_yarn_impact_theory()`: Overhauled in Sprint 3 to interpolate transverse shock kink wavefront and strain against Smith's 1958 analytical relations (lines 410-541).
    *   `test_prestrained_string_static_deflection()`: Verifies static out-of-plane deflection under a transverse point force (lines 542-661).
    *   `test_damping_decay_rate()`: Verifies Rayleigh proportional damping envelope decay rate (lines 662-855).
    *   `test_progressive_failure_and_fracture_energy()`: Verifies softening strain and JIT-integrated fracture energy matches analytical integral (lines 856-961).
    *   `test_thermodynamic_monotonicity()`: Confirms First Law total energy conservation and Second Law physical energy decay (lines 962-1084).
    *   `test_ballistic_limit_v50()`: Replicates a 1-ply Kevlar 29 impact sweep verifying arrest at 150 m/s and penetration at 400 m/s (lines 1085-1250).

### B. Data Reporting Structure
*   **CSV Exporter:** In `src/kevlargrid/io/export/csv_writer.py` (lines 13-69), time-series summary telemetry is saved:
    ```python
    def export_to_csv(history: list[dict[str, Any]], filepath: str) -> None
    ```
    The headers are: `Time (s)`, `Peak Strain`, `Kinetic Energy (J)`, `Strain Energy (J)`, `Damping Energy (J)`, `Projectile Z Position (m)`, `Projectile Velocity (m/s)`, `Rupture Energy (J)`, `Clamping Energy (J)`.
*   **HDF5 trajectory archiver:** In `src/kevlargrid/io/export/h5_writer.py` (lines 16-115), binary spatial states are archived:
    ```python
    def export_to_h5(config: dict[str, Any], results_report: dict[str, Any], history: list[dict[str, Any]], filepath: str) -> None
    ```
    Saves metadata/config_json, results outcome attributes (`arrested`, `peak_deceleration_g`, `yarn_rupture_percentage`, `residual_velocity_ms`, `energy_dissipation_efficiency`, `max_layer_perforated`), and compressed `time_history` datasets (`time`, `positions`, `spring_failures`, `projectile_pos`, `energies` matrix, `failure_dissipated`, `clamp_dissipated`).
*   **HTML/PDF report builders:** In `src/kevlargrid/io/export/report_builder.py` (lines 345-378), standalone summaries are generated:
    ```python
    def generate_plots_base64(history: list[dict[str, Any]]) -> tuple[str, str]
    def generate_report_html(config: dict[str, Any], results_report: dict[str, Any], history: list[dict[str, Any]]) -> str
    def generate_pdf_report(config: dict[str, Any], results_report: dict[str, Any], history: list[dict[str, Any]], filepath: str) -> None
    ```
    It uses `matplotlib.pyplot` headlessly to output Base64-encoded strain and energy plots and compiles the final report using `weasyprint` (with a transparent fallback to HTML if `weasyprint` is missing).

### C. Contact Force & Friction Modeling
*   **Z-Axis Contact Force Only:** In `src/kevlargrid/solver/forces.py` (lines 249-327) and `src/kevlargrid/solver/taichi_solver.py` (lines 922-936), the inter-ply contact force is computed purely along the Z-axis:
    ```python
    # In forces.py:
    gap = np.abs(z_n - z_n1)
    penetration = t_ply - gap
    f_mag = where(penetrating, k_penalty * penetration, 0.0)
    direction = where(z_n > z_n1, 1.0, -1.0)
    forces_n = stack_z(f_mag * direction)
    ```
    *Observation:* There is currently no dry friction coefficient or lateral shear force calculation modeled in the inter-ply or projectile contact algorithms in the solver.

---

## 2. Logic Chain

1. **Friction Modeling Gap:** The project rules require that inter-ply contact incorporates dry static friction ($\mu_s \ge 0.18$). Currently, the inter-ply contact force implementation in `forces.py` (lines 249-327) and `taichi_solver.py` (lines 922-936) only acts in the Z direction based on penalty stiffness $k_{\text{penalty}}$ and Z-penetration. No friction force is applied in the X or Y directions.
2. **Dynamic Energy conservation:** The solver uses a leapfrog Verlet scheme. When two plies contact, the normal penalty force $F_N = k_{\text{penalty}} \cdot \delta_z$ acts as a spring. If $k_{\text{penalty}}$ is too high or the timestep is too large, it acts as a stiff explosive spring, causing numerical instabilities or phantom energy additions ($>2\%$ drift), which violates the physical energy conservation constraint.
3. **CFL timestepping:** Kevlar has a high wave speed ($c \approx 7,000$ m/s). An explicit solver requires a timestep smaller than the critical limit ($\Delta t_{\text{crit}} = \frac{dx}{c}$) to prevent stress waves from skipping elements and causing unphysical node strain failures. The CFL safety factor adjusts this.

---

## 3. Caveats

*   **Mocked WeasyPrint:** If WeasyPrint is missing (as in some CLI environments), the PDF generator falls back to HTML. The automated runner will need to accommodate this fallback behavior.
*   **Friction Integration:** Since friction is currently missing, the `teamwork_preview_worker` must implement a dry friction contact force algorithm before calibration can succeed.

---

## 4. Conclusion

To successfully run and validate Benchmark 8, the following components must be designed and implemented by the implementing worker:

### A. Friction Force & Frictional Energy Dissipation Formulation
To meet the dry static friction constraint ($\mu_s \ge 0.18$), we propose a velocity-regularized Coulomb friction model in both CPU and GPU solvers:
1.  **Normal Contact Force:** $F_N = k_{\text{penalty}} \cdot \delta_{\text{penetration}}$ (along Z).
2.  **Relative Lateral Velocity:** $\Delta \mathbf{v}_{xy} = (v_{u, x} - v_{v, x}, v_{u, y} - v_{v, y}, 0)$.
3.  **Regularized Friction Force Vector:**
    $$\mathbf{F}_f = -\mu_s \cdot F_N \cdot \frac{\Delta \mathbf{v}_{xy}}{\sqrt{\|\Delta \mathbf{v}_{xy}\|^2 + \epsilon_{\text{reg}}^2}}$$
    where $\epsilon_{\text{reg}} \approx 10^{-3} \text{ m/s}$ is a regularization velocity to avoid numerical chatter and division by zero.
4.  **Opposite Force Application:** Apply $\mathbf{F}_{f, u} = \mathbf{F}_f$ to node $u$ (layer $n$) and $\mathbf{F}_{f, v} = -\mathbf{F}_f$ to node $v$ (layer $n+1$).
5.  **Frictional Energy Dissipation Tracking:**
    $$\Delta E_{\text{friction}} = \sum \|\mathbf{F}_f\| \cdot \|\Delta \mathbf{v}_{xy}\| \cdot \Delta t$$
    This must be added to the total energy balance to maintain conservation within $\le 2\%$.

### B. Programmatic Parameter Calibration Strategy
If limits drift or Cases A, B, and C fail validation bounds, the worker script should run an optimization sweep (e.g. grid search or coordinate descent) over $k_{\text{penalty}} \in [5\cdot 10^4, 2\cdot 10^6]$ N/m, $\mu_s \in [0.18, 0.25]$, and $\text{cfl\_factor} \in [0.1, 0.8]$.
*   **Objective Function:**
    $$\text{Minimize } \Phi = (V_{r, A})^2 + \max(0, V_{r, B} - 25)^2 + (V_{r, C} - 220)^2 + 10^5 \cdot \max(0, \Delta E_{\text{drift}} - 0.02)$$
    This objective enforces:
    1.  **Arrest at 450 m/s** (residual velocity $V_{r, A} = 0$).
    2.  **Marginal perforation at 503 m/s** (residual velocity $V_{r, B} < 25$ m/s).
    3.  **Residual target velocity of 220 m/s at 550 m/s** ($V_{r, C} \approx 220$).
    4.  **Energy conservation drift $\le 2.0\%$** (penalizes phantom energy generation).

---

## 5. Verification Method

To verify these findings and the final implementation:
1.  **Run the local test suite:**
    ```bash
    pytest tests/
    ```
    This verifies that all existing wave speed, progressive damage, and 1-ply ballistic limit tests pass.
2.  **Inspect the exported files:** Check that `benchmarks/benchmark_8/results.json` contains valid arrays and floats matching the schema, and that the validation plots and PDFs compile correctly.
3.  **Check Energy Conservation:** Verify in the output logs of Case C that `total_energy` does not drift by more than 2% during impact.
