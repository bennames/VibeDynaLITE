# KevlarGrid Explicit Solver v2.0 — Expanded PRD

## 1. Project Overview

**Objective:** Build a 2D/3D explicit mass-spring simulation tool with a desktop GUI to analyze the dynamic impact of a high-energy projectile (~9000 J) against woven Kevlar barriers.

**Goal:** Determine and validate the minimum number of Kevlar plies required to contain a propeller blade fragment without exceeding the fabric's ultimate strain failure threshold.

**Primary Users:** Small team of engineers using local desktop installations.

**Language:** Python 3.11+

---

## 2. Technical Stack & Architecture

### 2.1 Core Compute

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| **Primary compute** | **Taichi** (`ti.kernel`, `ti.func`, `SNode`) | JIT compiler targeting Apple Silicon (Metal), Vulkan, CUDA, and CPU. High-performance mutable memory layouts. |
| **Fallback compute** | **Numba** (`@njit`, `@numba.jit`) | JIT compiled parallel CPU backend for direct performance benchmarking and CPU environments. |
| **Array library** | **NumPy** | Data interchange, pre/post-processing, I/O. |
| **Serialization** | **HDF5** (`h5py`) + **JSON** | HDF5 for large result arrays; JSON for configuration. |

> **IMPORTANT:** The solver supports multiple backends. A runtime configuration selects either the high-performance GPU-resident Taichi backend or the parallelized CPU-resident Numba backend.

### 2.2 GUI Framework: DearPyGui

**Recommendation: DearPyGui** — selected for the following reasons:

| Criterion | DearPyGui | PyQt | CustomTkinter |
|-----------|-----------|------|---------------|
| Real-time plotting | ✅ Native, GPU-accelerated | ⚠️ Requires embedding matplotlib | ❌ Very limited |
| 3D viewport | ✅ Built-in 3D drawing API | ⚠️ Requires VTK/PyVista widget | ❌ Not feasible |
| Responsiveness during simulation | ✅ Non-blocking render loop | ⚠️ Requires careful threading | ⚠️ GIL-bound |
| Cross-platform | ✅ macOS, Windows, Linux | ✅ | ✅ |
| Learning curve | Moderate | Steep | Low |

DearPyGui's immediate-mode rendering and built-in plot/3D primitives make it the best fit for real-time strain telemetry and the 3D impact animation viewport.

### 2.3 Visualization Stack

| Purpose | Library |
|---------|---------|
| Real-time 2D plots (strain vs. time, energy balance) | DearPyGui built-in plots |
| 3D impact animation viewport | DearPyGui 3D drawlist + custom mesh renderer |
| Offline high-quality rendering / video export | **PyVista** (VTK wrapper) for frame export → **FFmpeg** for MP4/GIF assembly |
| PDF/HTML report generation | **Jinja2** templates + **matplotlib** for static figures |

---

## 3. Physics & Mathematics Engine — Detailed Specification

### 3.1 Governing Equation

Each node `i` obeys Newton's Second Law:

```
m_i * a_i = Σ F_spring(i,j) + F_contact(i) + F_damping(i)
```

where the sum is over all connected springs `j`, `F_contact` is the projectile-fabric contact force, and `F_damping` is the damping force.

**Time integration:** Central difference (Leapfrog / Störmer-Verlet):

```
v(t + dt/2) = v(t - dt/2) + a(t) * dt
x(t + dt)   = x(t) + v(t + dt/2) * dt
```

### 3.2 Grid Topology

The fabric is modeled as a rectangular grid of nodes connected by springs:

- **Orthogonal springs:** Along warp (x) and weft (y) directions. Stiffness `k_ortho` derived from material tensile modulus.
- **Diagonal (shear) springs:** Along ±45° diagonals. Stiffness `k_shear` auto-derived from material shear modulus:

```
k_shear = k_ortho * (G / E)
```

where `G` is the in-plane shear modulus and `E` is the tensile modulus.

> **NOTE:** Each node has **8 connections** (4 orthogonal + 4 diagonal) in the interior. Boundary/corner nodes have fewer.

### 3.3 Spring Force Model

Spring force uses true 3D Euclidean distance:

```
L_current = sqrt((x_j - x_i)² + (y_j - y_i)² + (z_j - z_i)²)
strain     = (L_current - L_rest) / L_rest
F_spring   = k * strain * L_rest   (directed along the spring axis)
```

- Springs are **tension-only** for the orthogonal direction (fabric cannot resist compression in-plane) — configurable toggle.
- Diagonal springs carry shear loads in both tension and compression.

### 3.4 Damping Model

Two damping models are available, selectable in the GUI:

**Default — Viscous Damping (Simple):**
```
F_damping(i) = -c * v(i)
```
where `c` is a user-configurable damping coefficient. Recommended default: `c = 0.01 * m_i * ω_max` where `ω_max` is the maximum natural frequency of the system.

**Advanced — Rayleigh Damping:**
```
F_damping = -(α * M + β * K) * v
```
where `α` (mass-proportional) and `β` (stiffness-proportional) are user inputs. The GUI should provide helper text explaining typical ranges.

### 3.5 Failure Criteria

**Binary failure (MVP):**
- If `strain > ε_fail`, then `k → 0` permanently for that spring.
- The spring is flagged as "deleted" and rendered in red/transparent in the 3D viewport.
- Node mass is **not** removed — disconnected nodes retain mass but become free-flying.

**Progressive damage (Future Enhancement):**
- After exceeding a damage initiation threshold `ε_damage < ε_fail`, stiffness degrades:
  ```
  k_damaged = k * (1 - D)
  D = (strain - ε_damage) / (ε_fail - ε_damage)   clamped to [0, 1]
  ```
- When `D = 1`, the spring is fully deleted.

### 3.6 Dynamic Boundary Sizing ("Infinite Boundary")

To prevent artificial wave reflection, the solver auto-sizes the minimum Kevlar barrier radius:

```
R_min = c_transverse * t_sim * safety_factor
```

where:
- `c_transverse = sqrt(T / (ρ * A))` — transverse wave speed
- `T` — pretension (if any), or initial stiffness-derived equivalent
- `ρ * A` — linear mass density
- `t_sim` — estimated simulation duration
- `safety_factor` — default 1.5, user-configurable

The GUI toggle:
- **"Infinite Boundary" ON:** Auto-computes `R_min`, overrides user grid size if it's too small.
- **"Infinite Boundary" OFF:** User specifies a clamped boundary size directly.

### 3.7 CFL Timestep Control

**Fixed CFL-based timestep (MVP):**

```
dt = CFL * (dx / c_max)
```

where:
- `CFL` — Courant number (safety factor), default `0.8`, range `(0, 1)`
- `dx` — minimum spring rest length in the grid
- `c_max` — maximum wave speed: `c_max = sqrt(k_max / m_min * dx)`

The GUI must display:
- Computed `dt` value
- Estimated total timesteps for the simulation duration
- Warning if user tries to override with an unstable `dt`

**Adaptive timestep (Future Enhancement):**
- Recompute `c_max` locally after each failure event (spring deletion changes local stiffness).
- Allow `dt` to increase if the grid softens, or decrease if new stiff regions emerge.

### 3.8 Dual-Mode Solver Architecture

#### Mode A: Sizing Mode (Fast Iteration)

- Single 2D grid in the x-y plane with out-of-plane (z) deflection.
- Number of plies `N_plies` acts as a **scalar multiplier**:
  - Node mass: `m_node = N_plies * ρ_areal * dx * dy`
  - Spring stiffness: `k = N_plies * E * A_cross / L_rest`
- **Output:** Minimum `N_plies` such that `max(strain) < ε_fail` at all times.
- **Auto-sweep:** Option to automatically run a bisection search over `N_plies` range `[1, N_max]`.

#### Mode B: Checkout Mode (High Fidelity)

- `N_plies` discrete 2D grids stacked along the Z-axis with initial spacing `t_ply`.
- **Inter-ply contact — Penalty method (MVP):**
  ```
  F_contact(i) = k_penalty * max(0, z_layer_n(i) - z_layer_n+1(i) + t_ply)
  ```
  - `k_penalty` — user-configurable, default `10 * k_ortho`
  - Applied node-to-node between corresponding nodes on adjacent layers.
- **Kinematic contact (Future Enhancement):**
  - After penalty contact detection, correct positions to enforce zero interpenetration.
  - More physically accurate but requires iterative solve per timestep.
- **Output:** Per-layer failure maps, sequential failure progression, total energy absorbed per layer.

### 3.9 Projectile Model

**Rigid body (MVP):**
- Defined by mass `m_proj` and initial velocity `v_0` → KE = ½ m v².
- **Blade footprint:** Rectangular contact area defined by `blade_width` and `edge_thickness`.
  - Mapped to the set of fabric nodes whose initial positions fall within the footprint rectangle.
  - As the blade decelerates, the contact zone **evolves**: nodes that come within a proximity threshold of the blade surface are added to the contact set.
- **Contact force distribution:** The total blade force is distributed to contact nodes weighted by proximity:
  ```
  w_i = 1 / max(d_i, d_min)    (inverse distance weighting)
  F_i = F_total * (w_i / Σ w_j)
  ```
- **Blade equation of motion:**
  ```
  m_proj * a_proj = -Σ F_contact(i)   (reaction from fabric)
  ```
  The blade is tracked as a single rigid body with position `z_proj(t)` and velocity `v_proj(t)`.

**Penetration detection:**
- If the blade z-position passes through the last ply and all springs in the contact zone are deleted → **full penetration**.
- Report residual KE and exit velocity.

### 3.10 Strain-Rate Sensitivity (Future Enhancement)

Kevlar exhibits rate-dependent stiffening at high strain rates (>100 s⁻¹):

```
k_dynamic = k_static * (1 + C * ln(strain_rate / strain_rate_ref))
```

where `C ≈ 0.01–0.03` for Kevlar. This is toggled as an optional advanced parameter.

---

## 4. Required Inputs & Data Management

### 4.1 Built-in Material Library

See [material_properties.md](material_properties.md) for the complete property table with cited sources.

| Material | E (GPa) | ε_fail (%) | σ_ult (GPa) | G/E | Areal Density (kg/m²) | Fabric Style |
|----------|---------|------------|-------------|-----|-----------------------|-------------|
| Kevlar 29 (Heavy Ballistic) | 71.0 | 3.6 | 2.92 | 0.0004 | 0.47 | Style 745 |
| Kevlar 49 (High Modulus) | 112.4 | 2.4 | 3.00 | 0.0003 | 0.23 | Style 328 |
| Kevlar KM2 (High Performance) | 84.62 | 3.55 | 3.40 | 0.0004 | 0.180 | Style 706 |
| **Custom** | User | User | User | User | User | — |

Sources: DuPont Kevlar Technical Guide; Cheng, Chen & Weerasooriya (2005), *J. Eng. Mater. Technol.* 127(2), 197–203.

> **NOTE:** KM2 values use "from fabric" measurements (not virgin fiber) to account for weaving degradation per Cheng et al. (2005).

### 4.2 Projectile Configuration

| Parameter | Unit | Default | Notes |
|-----------|------|---------|-------|
| Projectile mass | kg | — | Required input |
| Impact velocity | m/s | — | Required input; GUI shows computed KE |
| Blade width | mm | — | Width of the cutting edge footprint |
| Edge thickness | mm | — | Thickness of the blade edge (sharp vs. blunt) |

### 4.3 Simulation Configuration

| Parameter | Unit | Default | Notes |
|-----------|------|---------|-------|
| Simulation duration | ms | Auto | Auto-estimated from `R_min / c_transverse` |
| CFL safety factor | — | 0.8 | Range (0, 1) |
| Boundary mode | — | Infinite | Toggle: Infinite / Fixed |
| Fixed boundary radius | m | — | Only if boundary mode = Fixed |
| Grid spacing `dx` | mm | Auto | Auto from blade edge thickness or user override |
| Solver mode | — | A | Toggle: A (Sizing) / B (Checkout) |
| Number of plies | — | 1 | Mode A: scalar; Mode B: number of discrete layers |
| Ply spacing | mm | 0.5 | Mode B only |
| Damping model | — | Viscous | Toggle: Viscous / Rayleigh |
| Damping coefficient `c` | — | Auto | Auto-computed from `0.01 * m * ω_max` |
| Rayleigh α | — | 0.0 | Advanced: mass-proportional |
| Rayleigh β | — | 0.0 | Advanced: stiffness-proportional |
| Penalty stiffness | — | 10× k_ortho | Mode B only |

### 4.4 Data Persistence

**Configuration files:** JSON format with schema validation.

```json
{
  "version": "2.0",
  "material": { "name": "Kevlar KM2", "k": 12345, "eps_fail": 0.036, "..." : "..." },
  "projectile": { "mass": 0.5, "velocity": 190, "blade_width": 50, "edge_thickness": 2 },
  "simulation": { "mode": "A", "plies": 8, "cfl": 0.8, "boundary": "infinite", "..." : "..." },
  "damping": { "model": "viscous", "c": 0.01 }
}
```

**Result files:** HDF5 with the following datasets:

| Dataset | Shape | Description |
|---------|-------|-------------|
| `/time` | `(N_steps,)` | Time array |
| `/nodes/position` | `(N_steps, N_nodes, 3)` | x, y, z per node per step |
| `/nodes/velocity` | `(N_steps, N_nodes, 3)` | vx, vy, vz per node per step |
| `/springs/strain` | `(N_steps, N_springs)` | Strain per spring per step |
| `/springs/failed` | `(N_springs,)` | Boolean failure flags |
| `/springs/fail_time` | `(N_springs,)` | Time of failure (NaN if intact) |
| `/projectile/position` | `(N_steps,)` | z-position of blade |
| `/projectile/velocity` | `(N_steps,)` | z-velocity of blade |
| `/energy/kinetic` | `(N_steps,)` | Projectile KE |
| `/energy/strain` | `(N_steps,)` | Total fabric strain energy |
| `/energy/total` | `(N_steps,)` | Sum (should be constant) |

> **NOTE:** For large simulations, store every `N`th frame (user-configurable snapshot interval) rather than every timestep.

---

## 5. Required GUI Outputs & Telemetry

### 5.1 Status Dashboard

- **Pass/Fail indicator:** Large, color-coded badge.
  - 🟢 **PASS** — Max strain never exceeded `ε_fail` in any spring; projectile arrested.
  - 🔴 **FAIL** — Full penetration occurred.
  - 🟡 **MARGINAL** — No penetration, but max strain reached >90% of `ε_fail`.
- **Key metrics table:**
  - Max strain observed and its location (node ID, layer)
  - Peak deflection (mm)
  - Time to arrest or penetration (ms)
  - Residual projectile velocity (m/s) — 0 if arrested
  - Number of failed springs / total springs (%)
  - Energy absorbed by fabric (J) / initial KE (J)

### 5.2 Strain Telemetry Plot

- **X-axis:** Time (ms)
- **Y-axis:** Maximum strain in the grid (dimensionless)
- **Overlay:** Horizontal line at `ε_fail` threshold
- **Real-time update:** Plot updates every `N` timesteps during simulation (configurable refresh rate)

### 5.3 Energy Balance Plot

- **X-axis:** Time (ms)
- **Y-axis:** Energy (J)
- **Traces:**
  - Projectile Kinetic Energy (blue)
  - Fabric Strain Energy (orange)
  - Damping dissipation (gray, if damping enabled)
  - Total Energy (dashed black — should be flat)
- **Validation:** If total energy drift exceeds a threshold (e.g., 1% of initial KE), display a ⚠️ warning

### 5.4 3D Viewport

- **Elements rendered:**
  - Fabric nodes as points, color-coded by strain (colormap: blue → green → yellow → red)
  - Springs as lines, color-coded by strain; failed springs in dark red / transparent
  - Projectile blade as a rectangular solid, color-coded by velocity
  - Boundary ring / rectangle (if clamped)
- **Interactivity:**
  - Orbit / pan / zoom with mouse
  - Play / pause / step / scrub timeline
  - Playback speed control (0.1× to 10×)
  - Toggle layer visibility (Mode B)
- **Animation export:** Record viewport to MP4/GIF via offscreen rendering

### 5.5 Report Generation

**HTML report** (primary) and **PDF** (via `weasyprint` or `pdfkit`) containing:

1. Configuration summary table
2. Pass/Fail result with key metrics
3. Strain vs. time plot (static image)
4. Energy balance plot (static image)
5. Failure map — top-down view showing failed springs at final timestep
6. Peak deflection contour plot
7. (Mode B) Per-layer failure progression snapshots

---

## 6. Sprint Breakdown

### Sprint 0: Project Scaffolding & CI (1 week)

See [docs/sprint0_tasks.md](sprint0_tasks.md) for the detailed task list.

### Sprint 1: Core Physics Engine — Single Grid (2 weeks)

**Goal:** Implement the explicit solver for a single 2D grid (Mode A, no GUI).

| ID | User Story | Acceptance Criteria |
|----|-----------|---------------------|
| S1.1 | Grid generator with orthogonal and diagonal springs. | Correct node/spring counts for 5×5, 10×10, 100×100 grids. |
| S1.2 | Spring force computation using 3D Euclidean distance. | Single spring stretched by 10% returns correct force. |
| S1.3 | Central difference time integration. | Single node under constant force accelerates linearly. |
| S1.4 | CFL-based timestep computation. | For known k, m, dx, dt matches hand calculation. |
| S1.5 | Binary spring failure. | Spring fails at correct strain; subsequent force = 0. |
| S1.6 | Viscous damping. | Undamped conserves energy; damped loses monotonically. |
| S1.7 | Energy tracking. | Total energy drift < 0.1% over 10,000 steps (undamped). |
| S1.8 | Ply-count scalar multiplier (Mode A). | Wave speed unchanged by N_plies scaling. |

**Validation:** 1D wave propagation — wave speed matches analytical within 2%.

### Sprint 2: Projectile Contact & Termination (2 weeks)

**Goal:** Rigid projectile model, evolving contact, simulation termination.

| ID | User Story | Acceptance Criteria |
|----|-----------|---------------------|
| S2.1 | Rigid projectile with mass and velocity. | Free projectile moves at constant velocity. |
| S2.2 | Blade footprint mapping to fabric nodes. | 50mm × 2mm blade on 1mm grid → ~100 contact nodes. |
| S2.3 | Evolving contact detection. | Contact zone grows as blade deforms fabric. |
| S2.4 | Inverse-distance-weighted force distribution. | Equidistant nodes get equal force. |
| S2.5 | Projectile deceleration from fabric reaction. | Projectile decelerates and stops with strong fabric. |
| S2.6 | Auto-detect simulation termination. | Stops on arrest, penetration, or t_max. |
| S2.7 | Penetration reporting. | Reports residual KE and exit velocity. |

**Validation:** Energy balance closure within 1% tolerance.

### Sprint 3: Infinite Boundary & Mode B Multi-Ply (2 weeks)

**Goal:** Dynamic boundary sizing and multi-ply stacked simulation.

| ID | User Story | Acceptance Criteria |
|----|-----------|---------------------|
| S3.1 | Transverse wave speed computation. | Matches hand-calculation. |
| S3.2 | Auto-sizing of grid radius. | R_min matches expected value. |
| S3.3 | Fixed clamped boundary conditions. | Wave reflects off clamped boundary. |
| S3.4 | Multi-ply grid generation (Mode B). | Correct counts = N_plies × single_grid. |
| S3.5 | Penalty contact between adjacent plies. | Correct repulsion; load distributes to both layers. |
| S3.6 | Per-layer failure tracking. | Independent failure counts and time arrays. |
| S3.7 | Per-layer energy accounting. | Sum of per-layer SE = total SE. |

**Validation:** Front ply fails first in 3-ply sim; total energy within 20% of Mode A.

### Sprint 4: GUI Foundation (2 weeks)

**Goal:** DearPyGui app with configuration, save/load, simulation controls.

| ID | User Story | Acceptance Criteria |
|----|-----------|---------------------|
| S4.1 | Material dropdown with custom override. | Fields populate; custom enables manual input. |
| S4.2 | Projectile configuration inputs. | Live KE computation displayed. |
| S4.3 | Simulation configuration inputs. | Sensible defaults, tooltips, validation. |
| S4.4 | Toggle Mode A / Mode B. | Shows/hides relevant fields. |
| S4.5 | Toggle Infinite / Fixed boundary. | Auto-computes R_min or manual input. |
| S4.6 | Save and load configurations (JSON). | Round-trip fidelity; schema validation. |
| S4.7 | Start / Pause / Stop / Reset controls. | Background thread; resumable pause. |
| S4.8 | Progress indicator. | Progress bar with ETA. |
| S4.9 | Session auto-save and crash recovery. | Auto-save every 60s; restore on launch. |

### Sprint 5: GUI Visualization (2 weeks)

**Goal:** Live plots and 3D viewport.

| ID | User Story | Acceptance Criteria |
|----|-----------|---------------------|
| S5.1 | Real-time strain vs. time plot. | Updates every N steps with threshold line. |
| S5.2 | Real-time energy balance plot. | KE, SE, damping, total traces. |
| S5.3 | 3D viewport of impact event. | Nodes/springs colored by strain. |
| S5.4 | Orbit, pan, zoom. | Mouse and keyboard controls. |
| S5.5 | Failed springs highlighted. | Dark red / transparent + counter. |
| S5.6 | Post-sim playback controls. | Play/pause/step/scrub/speed. |
| S5.7 | Layer visibility toggle (Mode B). | Per-layer checkboxes. |
| S5.8 | Pass/Fail dashboard. | Color-coded badge + metrics. |

### Sprint 6: Data Export & Reporting (1.5 weeks)

**Goal:** HDF5/CSV export, video, reports.

| ID | User Story | Acceptance Criteria |
|----|-----------|---------------------|
| S6.1 | Export results to HDF5. | Matches schema; loads in Python and MATLAB. |
| S6.2 | Export results to CSV. | Opens in Excel. |
| S6.3 | Export animation as MP4/GIF. | Configurable resolution and frame rate. |
| S6.4 | Generate HTML report. | Self-contained with embedded plots. |
| S6.5 | Generate PDF report. | Same content as HTML. |
| S6.6 | Snapshot interval configuration. | Shows estimated file size. |

### Sprint 7: Performance Optimization (2 weeks)

**Goal:** GPU acceleration on Apple Silicon and NVIDIA.

| ID | User Story | Acceptance Criteria |
|----|-----------|---------------------|
| S7.1 | Auto-detect and use GPU. | Displays backend in status bar. |
| S7.2 | Vectorized spring force computation. | ≥10× speedup over naive loop (500×500). |
| S7.3 | JIT-compiled timestep kernel. | ≥100× faster after compilation. |
| S7.4 | SoA memory layout. | Positions (3, N) not (N, 3). |
| S7.5 | Performance benchmarks. | 50²–1000² grid timings logged. |
| S7.6 | Memory usage profiling. | 1000² fits in 16 GB. |

**Targets:** 100² < 0.5s (GPU), 500² < 5s (GPU), 1000² < 20s (GPU) per 1000 steps.

### Sprint 8: Integration Testing & Polish (2 weeks)

**Goal:** End-to-end validation, regression, final polish.

| ID | User Story | Acceptance Criteria |
|----|-----------|---------------------|
| S8.1 | 1D wave propagation validation. | Wave speed within 2% of analytical. |
| S8.2 | Energy conservation validation. | Drift < 0.1% over full sim. |
| S8.3 | Regression dataset. | 3 golden test cases in CI. |
| S8.4 | Mode A vs. Mode B cross-validation. | Documented acceptable range. |
| S8.5 | GUI interaction tests. | Save/load round-trip, invalid input rejection. |
| S8.6 | Error messages. | User-friendly dialogs. |
| S8.7 | User manual / help section. | Built-in help with quick-start guide. |
| S8.8 | CI performance regression detection. | Alert on >20% runtime increase. |

---

## 7. Testing & Validation Strategy

| Test Category | Scope | Tooling | When |
|---------------|-------|---------|------|
| **Unit tests** | Spring force, CFL, wave speed, energy, grid gen, failure | `pytest` + `numpy.testing` | Every commit (CI) |
| **Integration tests** | 1D wave, point impact, multi-ply contact | `pytest` (longer, separate CI stage) | Every PR merge |
| **Energy conservation** | Total energy drift < 0.1% (undamped) | Custom assertions | Every PR merge |
| **Regression tests** | Golden dataset comparison (3 cases) | `pytest` + HDF5 diff | Every release candidate |
| **Performance benchmarks** | Wall-clock and memory for 50²–1000² | `pytest-benchmark` | Weekly CI + pre-release |
| **GUI tests** | Config save/load, buttons, validation | DearPyGui scripted / `pyautogui` | Every PR merge |

---

## 8. Future Enhancement Roadmap

| Feature | Sprint Slot | Dependency |
|---------|-------------|------------|
| Adaptive CFL timestep | Post-MVP A | Core solver stable |
| Kinematic inter-ply contact | Post-MVP B | Penalty contact validated |
| Progressive damage | Post-MVP C | Binary failure validated |
| Strain-rate dependent material | Post-MVP D | Core material model validated |
| Deformable projectile (semi-rigid) | Post-MVP E | Rigid projectile validated |
| Rayleigh damping | Sprint 1 (toggle) | Viscous damping validated |
| Automatic N_plies bisection sweep | Post-MVP F | Mode A validated |

---

## 9. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Taichi-Metal not stable on Apple Silicon | Low | Medium | Numba CPU fallback. Test both in CI. |
| DearPyGui 3D insufficient for large grids | Medium | Medium | Fall back to PyVista embedded window. |
| Performance targets not met for 1000² | Medium | High | SoA layout, JIT, sparse storage. Profile early. |
| Penalty contact instability (Mode B) | Low | Medium | k_penalty as user param. Future: kinematic. |
| HDF5 files too large | Low | Low | Configurable snapshot interval. gzip compression. |

---

## 10. Timeline

**Total estimated duration:** ~16.5 weeks (~4 months) for Sprints 0–8 with a single developer. With 2 developers, Sprints 5–6 can run parallel with Sprint 7, reducing to ~12 weeks.
