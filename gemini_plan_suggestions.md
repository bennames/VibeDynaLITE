# 🚀 VibeDynaLITE: Systemic Architecture Overhaul & Physics Hardening Plan

## 📋 Executive Summary
VibeDynaLITE aims to be a high-fidelity explicit dynamics solver for anisotropic, high-strain-rate woven fabrics (Kevlar). While the repository demonstrates strong modularity and a mature approach to physics benchmarks, the current codebase architecture contains deep systemic flaws. 

These flaws currently compromise both **computational performance** (exhibited by severe Taichi bottlenecks caused by PCIe bus saturation) and **physical trustability** (exhibited by unvalidated mass-scaling and instantaneous numerical shockwaves). 

This document outlines a strict, phased execution plan to commit to a purely GPU-resident architecture, harden the physics, and implement production-grade simulation guardrails.

**Agent Instruction:** Read this entire document carefully. You will be executing a phased remediation plan to transition the codebase to a Taichi-exclusive, memory-resident, Headless-First architecture while implementing true Continuum Damage Mechanics and analytical test anchors.

---

## 🛠️ Phase 1: Architecture & Performance (The Taichi Overhaul)
**Status:** 🔴 CRITICAL | **Target Files:** `backend.py`, `detect_backend.py`, `taichi_solver.py`, `worker.py`, `fused.py`, `tests/solver/*`

### The Problem
The codebase is currently suffering from a "Split-Brain Backend Parity Trap." By forcing the Taichi GPU compiler to share an API signature with JAX and NumPy, we have forced it into fatal anti-patterns:
1. **The Data Yo-Yo (PCIe Bus Saturation):** The overarching state is held in Python/NumPy, requiring `from_numpy()` and `to_numpy()` at every timestep. This means 99% of execution time is spent moving data across the PCIe bus instead of doing physics.
2. **Python-Scope Looping:** Python loops are triggering Taichi kernels, causing massive dispatch overhead per node.
3. **The GIL Bottleneck:** The UI, I/O, and physics worker are thread-coupled, meaning the explicit dynamics GPU solver is constantly waiting on the Python Global Interpreter Lock.

### 📝 Action Items

- [ ] **1.1 Deprecate Alternative Backends:**
  - Delete `detect_backend.py`, `test_jax_jit_cfl.py`, `test_jit_compat.py`, and `test_parity.py`.
  - Remove all JAX, Numba, and pure NumPy solver integration loops from `backend.py`. Commit 100% to Taichi to leverage its mutable `SNode` trees and imperative kernels.
- [ ] **1.2 Enforce Absolute Taichi Data Residency:**
  - Refactor `taichi_solver.py`. The solver class must allocate `ti.field` objects ONCE during `__init__`.
  - The Python host must **never** hold the primary state arrays. Use `.from_numpy()` only during initialization. **Never call `.to_numpy()` inside the physics loop.**
- [ ] **1.3 Implement the "Mega-Kernel" Pattern:**
  - Refactor `forces.py`, `damping.py`, and `failure.py` into strictly typed `@ti.func` helper functions.
  - Create a single, monolithic `@ti.kernel` inside `taichi_solver.py` (or `fused.py`) that executes the entire timestep. The `for i in self.pos:` loop must be the *first* thing inside the kernel, guaranteeing it compiles and runs entirely on the GPU.
- [ ] **1.4 Decouple I/O and Enforce Sub-Stepping (Headless-First):**
  - Do not pull data back to Python every timestep. Implement an inner loop inside the Taichi Mega-Kernel to run `N` timesteps purely on the GPU before yielding.
  - Rip out `worker.py` threading. Transition to a **Multi-Process Architecture**: Process A (Taichi GPU Solver) writes to a shared-memory buffer at regular intervals; Process B (UI/IO) reads the buffer asynchronously without blocking the solver.

**Code Pattern Reference for AI Agent:**
```python
# REQUIRED MEGA-KERNEL PATTERN
@ti.func
def compute_spring_forces_and_damage(i):
    # Physics math here (runs on device)
    pass

@ti.kernel
def advance_substeps(self, dt: ti.f32, num_substeps: ti.i32):
    for step in range(num_substeps):
        for i in self.pos: # This loop MUST be the outermost logic inside the kernel
            f = compute_spring_forces_and_damage(i)
            # Update kinematics natively in VRAM
            self.vel[i] += (f / self.mass[i]) * dt
            self.pos[i] += self.vel[i] * dt
```

---

## 🛑 Phase 2: Physics & Mechanics (The Trustability Core)
**Status:** 🟠 HIGH PRIORITY | **Target Files:** `failure.py`, `energy.py`, `boundary.py`, `timestep.py`

### The Problem
The current physical approximations in the solver will cause it to fail rigorous ballistic validation (e.g., V50 testing).
1. **Instantaneous Spring Failure:** Binary failure (`if strain > limit: delete`) releases all stored strain energy in a single timestep, creating a high-frequency acoustic shockwave ("ringing") that causes non-physical "zippering" failures in adjacent elements.
2. **Unvalidated Mass Scaling:** Mass scaling artificially lowers wave speed ($c = \sqrt{E/\rho}$), which fundamentally corrupts high-speed ballistic momentum transfer.
3. **Static Boundaries:** Static viscous dashpots will reflect stress waves back into the grid as the fabric yields and its acoustic impedance changes.

### 📝 Action Items

- [ ] **2.1 Implement Continuum Damage Mechanics (CDM):**
  - Rewrite `failure.py`. Remove binary element deletion.
  - Introduce a damage variable `D` in range `[0, 1]` governed by the material's Fracture Energy (`Gc`). 
  - When failure strain is reached, ramp up `D` over time, smoothly degrading stiffness (`E_eff = (1-D) * E`). This dissipates energy physically and prevents numerical shocks.
- [ ] **2.2 Enforce Mass-Scaling Guardrails:**
  - Disable mass scaling by default for all high-strain-rate or ballistic configurations.
  - Update `energy.py` to continuously calculate and track `E_artificial_kinetic` introduced by mass scaling.
  - Implement a hard halt: if `E_artificial_kinetic` exceeds 2% of the total internal energy during a run, raise a `PhysicsViolationError` and abort the simulation.
- [ ] **2.3 Dynamic Impedance Matching at Boundaries:**
  - Update `boundary.py`. Instead of static coefficients, boundary dashpots must dynamically adjust their damping coefficients at every step to match the local acoustic impedance ($Z = \rho c A$) of the yielding Kevlar fabric.
- [ ] **2.4 Dynamic Auto-CFL Enforcement:**
  - Update `timestep.py` and the UI configs. Users should **never** define a static timestep `dt`.
  - The UI should only accept a "CFL Safety Factor" (e.g., 0.8). The solver must continuously and rigidly calculate the critical timestep internally based on the highest localized wave speed and smallest deformed element size.

---

## 🖥️ Phase 3: UI, UX, I/O, and CI Guardrails
**Status:** 🟡 MEDIUM PRIORITY | **Target Files:** `configs/*`, `csv_export.py`, `test_physics_benchmarks.py`, `utils/units.py`

### The Problem
The tool's user inputs and data outputs are brittle and prone to "footguns." JSON doesn't support comments (which are mandatory for citing material properties). Exporting nodal arrays to CSV will crash local file systems. Furthermore, the CI tests check for code execution rather than rigorous mathematical truths.

### 📝 Action Items

- [ ] **3.1 Migrate to TOML & Strict Units:**
  - Convert all `.json` configuration files (e.g., `kevlar29_sizing.json`) to `.toml`.
  - Require inline citations/comments for all material properties (e.g., `# Modulus sourced from DuPont 1998 Tech Guide`).
  - Integrate `utils/units.py` tightly into the TOML parser. The solver itself should remain unit-agnostic (using a strict base like `kg-mm-ms`), but the parser must enforce strict dimensional analysis on user inputs before passing them to the solver.
- [ ] **3.2 Deprecate Full-Grid CSV Exports:**
  - Remove full-grid array exports from `csv_export.py` and `io/export/csv_writer.py`.
  - CSV exports should be restricted entirely to 1D time-series scalars (e.g., Total Kinetic Energy, Max Deflection vs. Time).
  - Force all full-grid state outputs to use binary formats (e.g., HDF5 via `h5_writer.py`).
- [ ] **3.3 Analytical Anchoring in CI Pipelines:**
  - Overhaul `test_physics_benchmarks.py`. CI must not just check for crashes or regressions against previous CSV outputs.
  - For `Benchmark 3 - Smith Yarn Impact`, hardcode the analytical *Cole-Smith mathematical equations* for transverse wave velocity.
  - Set CI assertions to fail if the solver's simulated wavefront velocity deviates by $> 2\%$ from the first-principles analytical solution.