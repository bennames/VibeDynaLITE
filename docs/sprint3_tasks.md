# Sprint 3: Infinite Boundary & Mode B Multi-Ply — Task List

**Duration:** 2 weeks
**Goal:** Implement boundary conditions, dynamic boundary sizing, stacked multi-ply grid generation (Mode B), inter-ply penalty contact forces, and per-layer failure/energy telemetry tracking.

---

## Tasks

### S3.1 — Clamped Boundaries & Dynamic Sizing (`solver/boundary.py`)

- [x] Implement `apply_clamped_boundary` to zero out velocities of boundary nodes.
- [x] Implement `compute_min_radius` to compute minimum grid radius:
  $$R_{\text{min}} = c_{\text{transverse}} \cdot t_{\text{sim}} \cdot \text{safety\_factor}$$

**Acceptance Criteria:**
- Clamped nodes have zero velocity and cannot translate.
- Minimum half-width scales linearly with wave speed, simulation duration, and safety factors.

---

### S3.2 — Multi-Ply Stacked Grid Generation (`solver/grid.py`)

- [x] Update `generate_rectangular_grid` to support stacking layers along the Z-axis in Checkout Mode (Mode B).
- [x] Program offset indices for multi-ply spring connectivity:
  $$\text{spring\_indices} = \text{base\_spring\_indices} + n \cdot n_{\text{nodes\_per\_layer}}$$
- [x] Set unscaled single-ply mass and stiffness coefficients for Mode B, preserving Mode A scalar scaling.

**Acceptance Criteria:**
- Grid generation outputs correct total node count ($N_{\text{plies}} \cdot n_{\text{nodes\_per\_layer}}$) and spring count ($N_{\text{plies}} \cdot n_{\text{springs\_per\_layer}}$).
- Node Z-coordinates are spaced perfectly by $n \cdot t_{\text{ply}}$.

---

### S3.3 — Inter-Ply Penalty Contact Forces (`solver/forces.py`)

- [x] Formulate node-to-node penalty contact forces resisting layer interpenetration:
  $$\delta_i = z_n(i) - z_{n+1}(i) + t_{\text{ply}}$$
  $$\mathbf{F}_{c, n, z}(i) = -k_{\text{penalty}} \cdot \delta_i \cdot \hat{\mathbf{z}}$$
  $$\mathbf{F}_{c, n+1, z}(i) = +k_{\text{penalty}} \cdot \delta_i \cdot \hat{\mathbf{z}}$$
- [x] Calculate inter-ply contact potential energy for full telemetry closure:
  $$E_{\text{contact, interply}} = \frac{1}{2} \sum k_{\text{penalty}} \cdot \delta_i^2$$

**Acceptance Criteria:**
- Forces apply equal and opposite loads on corresponding nodes of adjacent layers (action-reaction).
- Generates zero force when layers are separated.
- Potential energy closes the energy balance equation perfectly.

---

### S3.4 — Per-Layer Rupture Tracking (`solver/failure.py`)

- [x] Implement `get_layer_failure_stats` to count ruptured springs per ply:
  $$\text{spring\_layer} = \text{springs}[:, 0] // n_{\text{nodes\_per\_layer}}$$

**Acceptance Criteria:**
- Vectorized indexing accurately filters failed springs layer-by-layer.
- Summary outputs represent exact mathematical counts.

---

## Definition of Done (Sprint 3)

- [x] Boundary clamping, min radius sizing, stacked grid generation, and inter-ply contact fully implemented.
- [x] Mypy strict checks compile with zero warnings or strict typing errors.
- [x] Ruff formatting and style rules pass with zero warnings.
- [x] Boundary unit tests (`test_boundary.py`) verify clamping and radius calculations (100% pass).
- [x] Stacked grid and inter-ply force unit tests (`test_grid.py`, `test_forces.py`) pass cleanly.
- [x] Integration test verifying sequential load transfer and ply contact passes cleanly.
- [x] Combined energy drift remains below 2% during multi-ply penalty impacts.
- [x] Sprint 3 codebase staged, committed, and pushed successfully to GitHub repository.
