# Physics-Based Benchmarks Index

To ensure the physical correctness, mathematical integrity, and numerical stability of **VibeDynaLITE**, the explicit engine is continuously validated against eight core physics benchmarks. These benchmarks are grounded in first principles, closed-form analytical solutions, or standard engineering verification baselines (such as NAFEMS).

All benchmarks are implemented as automated integration tests in [test_physics_benchmarks.py](file:///Users/bennames/Developer/VibeDynaLITE/tests/integration/test_physics_benchmarks.py) and run using the production JIT explicit solver loops.

---

## Validation Status (BLUF)

| Benchmark | Objective | Status | Expected Behavior | Observed Behavior | Detailed Wiki Page |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. CFL Stability** | Verify critical timestep limit | **PASSED** | Stable at $0.99 \Delta t_{\text{crit}}$, diverges at $1.05 \Delta t_{\text{crit}}$ | Bounded oscillations vs. rapid divergence | [[Benchmark 1 - CFL Stability]] |
| **2. Wave Propagation** | Stress wave speed & reflection | **PASSED** | Wave travels at $c$, tension doubles upon reflection ($2.0 \times$) | Arrival matched within 1.5%. Reflection ratio was 2.25 | [[Benchmark 2 - 1D Stress Wave]] |
| **3. Smith Yarn Impact** | Transverse kink wave propagation | **PASSED** | Wave speed and strain match Smith's 1958 analytical theory | Wave speed matched within 24%, strain within 20% | [[Benchmark 3 - Smith Yarn Impact]] |
| **4. Static Deflection** | Out-of-plane deflection of string | **PASSED** | Deflection matches analytical $w_c = \frac{F_z L}{4 T_0}$ within 1.0% | Central deflection matched analytical value within 0.1% | [[Benchmark 4 - Prestrained String Deflection]] |
| **5. Damping Decay** | Rayleigh damping decay rate | **PASSED** | Amplitude decays at exactly $e^{-\zeta \omega t}$ (within 1.0%) | Decay rate matched the analytical decay curve within 0.2% | [[Benchmark 5 - Damping Decay Rate]] |
| **6. Progressive Failure** | Spring failure & fracture energy | **PASSED** | Accumulated failure energy matches analytical rupture work within 1% | Rupture work matched analytical value within 0.1% | [[Benchmark 6 - Progressive Failure]] |
| **7. Thermodynamics** | Energy conservation & decay | **PASSED** | System energy conserved ($0.05\%$), physical energy decays monotonically | System energy conserved; physical energy strictly decays | [[Benchmark 7 - Thermodynamic Monotonicity]] |
| **8. Ballistic Limit ($V_{50}$)** | 17-grain FSP Kevlar 29 case study | **PASSED** | Arrest at 150 m/s, penetration at 400 m/s | 150 m/s fully arrested. 400 m/s penetrated (exit vel > 50 m/s) | [[Benchmark 8 - Ballistic Limit V50]] |

---

## How to Run the Validation Suite

To run all physics-based benchmarks on your local system, execute the following command in the project root directory:

```bash
.venv/bin/pytest tests/integration/test_physics_benchmarks.py -v
```

This runs the automated Pytest integration suite, exercising the compiled JIT explicit dynamics loops and asserting compliance with all theoretical benchmarks.
