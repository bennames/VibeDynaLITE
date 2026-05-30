# Sprint 1: Core Physics Engine (Single Grid) — Task List

**Duration:** 1.5 weeks
**Goal:** Implement the core vectorized physics solver kernels (grid, forces, integration, damping, rupture, energy tracking) for a single-ply or multi-ply woven panel, with full static checking and physical verification.

---

## Tasks

### S1.1 — Rectangular Grid Generation (`solver/grid.py`)

- [x] Generate planar `(nx * ny, 3)` coordinates centered around `(0, 0)`.
- [x] Compute lumped node masses based on corner ($\frac{1}{4}$), edge ($\frac{1}{2}$), and interior ($1.0$) tributary areas:
  $$m_i = N_{\text{plies}} \cdot \rho_{\text{areal}} \cdot d_x^2$$
- [x] Configure 8-neighbor connections containing:
  - **Orthogonal springs** (warp/weft direction) with rest length $L_0 = d_x$ and stiffness $k_{\text{ortho}} = N_{\text{plies}} \cdot E \cdot t$.
  - **Diagonal springs** (shear direction) with rest length $L_0 = \sqrt{2} d_x$ and stiffness $k_{\text{shear}} = k_{\text{ortho}} \cdot \left(\frac{G}{E}\right)$.
- [x] Precompute a boolean mask identifying orthogonal springs for tension-only constraints.

**Acceptance Criteria:**
- Correct number of nodes ($N_x \times N_y$) and springs ($4 N_x N_y - 3 (N_x + N_y) + 2$).
- Corner nodes have exactly 25% of interior lumped mass, and edge nodes have exactly 50%.
- Rest lengths and spring stiffnesses match analytical values based on library material constants.

---

### S1.2 — Spring Force & Strain Solver (`solver/forces.py`)

- [x] Implement vectorized distance and engineering strain computation per spring:
  $$\varepsilon = \frac{L - L_0}{L_0}$$
- [x] Implement vectorized spring force magnitude:
  $$F_j = k_j \cdot \varepsilon \cdot L_0$$
- [x] Enforce tension-only limits: orthogonal springs yield zero force when under compression ($\varepsilon < 0$).
- [x] Zero out spring forces for ruptured springs ($D_j = 1$).
- [x] Accumulate force vectors on nodes using efficient array scattering operations.

**Acceptance Criteria:**
- Strains computed match hand calculations for simple geometries.
- Under compression, orthogonal springs report zero force (tension-only limit).
- Under stretch, force magnitude scales linearly with engineering strain and stiffness.

---

### S1.3 — Synchronous Velocity Verlet Integration (`solver/integrator.py`)

- [x] Implement state-synchronous central-difference integration:
  1. Position update:
     $$\mathbf{x}_i(t + dt) = \mathbf{x}_i(t) + \mathbf{v}_i(t) \cdot dt + \frac{1}{2} \mathbf{a}_i(t) \cdot dt^2$$
  2. Half-step velocity update:
     $$\mathbf{v}_i\left(t + \frac{dt}{2}\right) = \mathbf{v}_i(t) + \frac{1}{2} \mathbf{a}_i(t) \cdot dt$$
  3. Accelerations update based on newly evaluated forces:
     $$\mathbf{a}_i(t + dt) = \frac{\mathbf{F}_i(t + dt)}{m_i}$$
  4. Full-step velocity update:
     $$\mathbf{v}_i(t + dt) = \mathbf{v}_i\left(t + \frac{dt}{2}\right) + \frac{1}{2} \mathbf{a}_i(t + dt) \cdot dt$$
- [x] Correctly handle boundary node constraints (clamped boundaries or free edges).

**Acceptance Criteria:**
- Free particle with initial velocity moves linearly with zero acceleration.
- Constant force on a node produces perfect quadratic displacement and linear velocity curves.
- Positions and velocities remain synchronous at integer step intervals.

---

### S1.4 — CFL Stability Timestep Limit (`solver/timestep.py`)

- [x] Compute the analytical Courant-Friedrichs-Lewy (CFL) limit for the grid structure:
  $$\Delta t_{\text{crit}} = \text{CFL} \cdot \sqrt{\frac{m_{\text{min}}}{k_{\text{max}}}}$$
- [x] Factor in a safety parameter ($\text{CFL} \approx 0.9$) to ensure explicit stability.

**Acceptance Criteria:**
- Critically calculated timestep scales inversely with the square root of stiffness and linearly with the square root of mass.
- Solver remains stable over long periods under high grid resolution when utilizing the calculated timestep.

---

### S1.5 — Damping Solvers (`solver/damping.py`)

- [x] Implement velocity-proportional viscous damping representing air/drag resistance:
  $$\mathbf{F}_i^{\text{visc}} = -c \cdot \mathbf{v}_i$$
- [x] Implement structural relative-velocity Rayleigh damping:
  - Projects relative velocities of nodes connected by spring $j$ onto the spring's instantaneous unit direction:
    $$f^{\text{damp}}_j = \beta \cdot k_j \cdot \left((\mathbf{v}_b - \mathbf{v}_a) \cdot \frac{\mathbf{r}_{ab}}{L}\right)$$
  - Converts scalar spring damping to equal and opposite nodal forces.

**Acceptance Criteria:**
- Damping forces oppose motion (negative power dissipation).
- Viscous damping acts globally relative to the fixed ground.
- Rayleigh damping resists only relative axial motion along spring lines, preserving rigid body translations.

---

### S1.6 — Irreversible Rupture Criteria (`solver/failure.py`)

- [x] Track binary, strain-based spring failure states.
- [x] If a spring's engineering strain exceeds the material's failure strain limit ($\varepsilon > \varepsilon_{\text{fail}}$), irreversibly mark the spring failed:
  $$D_j = 1$$
- [x] Ensure that once failed, a spring never recovers and cannot transmit tension or compression forces.

**Acceptance Criteria:**
- Springs rupture exactly at the threshold strain limit.
- Failed springs remain ruptured forever, transferring zero force in subsequent steps.

---

### S1.7 — Kinetic, Elastic, and Damping Energy Tracking (`solver/energy.py`)

- [x] Compute total kinetic energy of the grid:
  $$E_k = \frac{1}{2} \sum_{i} m_i \|\mathbf{v}_i\|_2^2$$
- [x] Compute total elastic strain energy of all springs:
  $$E_s = \frac{1}{2} \sum_{j} k_j \cdot \varepsilon_j^2 \cdot L_{0, j} \quad (\text{if active and not failed})$$
- [x] Calculate energy balance drift to verify physical correctness:
  $$\Delta E = |E_{\text{total}}(t) - E_{\text{total}}(0)| \le 0.1\% \cdot E_{\text{total}}(0) \quad (\text{undamped})$$

**Acceptance Criteria:**
- Energy balance closes with minimal numerical noise.
- Under undamped conditions, kinetic and elastic strain energy swap smoothly while maintaining near-perfect total energy conservation.
- Damping forces monotonically dissipate mechanical energy.

---

### S1.8 — Ply Count Scaling Verification (`tests/integration/test_multiply.py`)

- [x] Formulate scalar multiplier rules where ply count ($N_{\text{plies}}$) linearly scales node masses and spring stiffnesses.
- [x] Verify scaling matches both Mode A (single lumped fabric equivalence) and Mode B (distinct laminated layers).

**Acceptance Criteria:**
- Mass and stiffness scale exactly linearly with $N_{\text{plies}}$.
- Waves travel at equivalent speeds in multi-ply grids since the ratio $\frac{k}{m}$ remains invariant.

---

## Definition of Done (Sprint 1)

- [x] All solver mathematical modules (`grid`, `forces`, `integrator`, `damping`, `failure`, `timestep`, `energy`) fully implemented.
- [x] Static type checker `mypy` compiles with zero warnings or strict typing errors.
- [x] Formatter and linter `ruff` report zero formatting issues or syntax warnings.
- [x] Verification unit tests (36 cases) and physical integration tests (6 cases) pass successfully.
- [x] Wave speed propagation limit matches theoretical sound speed within 2%.
- [x] Energy drift remains below 0.1% for the explicit Velocity Verlet integrator.
- [x] Sprint 1 codebase staged, committed, and pushed successfully to GitHub repository.
