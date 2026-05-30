# Sprint 2: Projectile Contact & Simulation Termination — Task List

**Duration:** 2 weeks
**Goal:** Implement the rigid-body projectile kinematics, evolving contact footprint mapping, inverse-distance proximity weighting, and simulation termination criteria.

---

## Tasks

### S2.1 — Rigid Projectile Model (`solver/projectile.py`)

- [x] Create the `Projectile` data structure storing mass, 3D velocity, 3D position, blade width, edge thickness, and active contact nodes.
- [x] Support 3D initial positions and initial velocity vectors.

**Acceptance Criteria:**
- Initialize projectile instances with all attributes properly typed.
- Coordinates, dimensions, and arrays stored as standard NumPy-compatible floats and integers.

---

### S2.2 — Contact Footprint Mapping & Proximity Clamping (`solver/projectile.py`)

- [x] Design closest-point projection mapping of grid nodes onto the finite rectangular blade face:
  $$x_{\text{proj}} = \text{clamp}(x_i, X_p - W/2, X_p + W/2)$$
  $$y_{\text{proj}} = \text{clamp}(y_i, Y_p - T/2, Y_p + T/2)$$
  $$z_{\text{proj}} = Z_p$$
- [x] Compute 3D Euclidean distances between the grid nodes and their projections on the blade.
- [x] Construct a proximity-based detection mask selecting nodes within distance threshold:
  $$d_i \le \text{proximity\_threshold}$$
- [x] Populate the active `contact_nodes` list.

**Acceptance Criteria:**
- Center nodes align and map correctly inside the footprint.
- Outer grid nodes correctly report zero proximity contact.
- Distance values perfectly represent short-range true 3D spatial separation.

---

### S2.3 — Conservative Inverse-Distance Weighted Penalty Forces (`solver/projectile.py`)

- [x] Compute Z-axis penetration depths for active contact nodes in the direction of motion.
- [x] Formulate a conservative, proximity-weighted penalty contact potential:
  $$V_i = \frac{1}{2} (k_{\text{contact}} \cdot w_{\text{normalized}, i}) \delta_i^2$$
  where $w_i = \frac{1.0}{\max(d_i, d_{\text{min}})}$ and $w_{\text{normalized}}$ is normalized to preserve force scaling.
- [x] Evaluate equal and opposite node contact and projectile reaction forces:
  $$\mathbf{F}_{c, i, z} = (k_{\text{contact}} \cdot w_{\text{normalized}, i}) \cdot \delta_i \cdot \text{sgn}(V_{z, p})$$
  $$\mathbf{F}_p = -\sum \mathbf{F}_{c, i}$$

**Acceptance Criteria:**
- Forces scale linearly with penetration depth (Hookean penalty contact).
- Force fields are fully conservative, avoiding artificial energy creation or destruction.
- Sum of interaction forces is exactly zero, ensuring linear momentum conservation.

---

### S2.4 — Kinematic Leapfrog Loop Integration (`solver/integrator.py`)

- [x] Advance projectile translation using symplectic central-difference leapfrog integration:
  $$\mathbf{v}_p\left(t + \frac{dt}{2}\right) = \mathbf{v}_p\left(t - \frac{dt}{2}\right) + \frac{\mathbf{F}_p(t)}{m_p} dt$$
  $$\mathbf{x}_p(t + dt) = \mathbf{x}_p(t) + \mathbf{v}_p\left(t + \frac{dt}{2}\right) dt$$
- [x] Feed grid node contact forces directly into the central difference integrator.

**Acceptance Criteria:**
- Projectile deceleration reaction couples directly to the grid’s local masses.
- Integrator positions and velocities remain synchronous for energy telemetry.

---

### S2.5 — Simulation Termination Triggers

- [x] Establish clear termination checks inside the explicit solver run framework:
  - **Arrest:** Projectile velocity $V_z$ drops to zero or reverses sign.
  - **Penetration:** Projectile passes beyond the fabric plane and all springs in the contact footprint are ruptured ($D_j = 1$).
  - **Timeout:** Simulated time reaches $t_{\text{max}}$.

**Acceptance Criteria:**
- Simulation terminates cleanly on arrest, penetration, or timeout.
- Reports final residual velocities and exit kinetic energy.

---

## Definition of Done (Sprint 2)

- [x] Rigid projectile data structures and kinematics fully implemented.
- [x] Vectorized closest-point clamping and evolving contact footprint verified.
- [x] Conservative proximity-weighted force formulation coded and mathematically proven.
- [x] Unit test suite (`test_projectile.py`) covers footprints, initialization, and force symmetry (100% pass).
- [x] Point impact integration tests verify physical arrest, deceleration, and energy conservation.
- [x] Total energy balance drift remains below 2% (well within explicit penalty contact limit).
- [x] Static lints (Ruff) and strict type annotations (MyPy) compile with zero warnings or errors.
- [x] Sprint 2 codebase staged, committed, and pushed successfully to GitHub repository.
