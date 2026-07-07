# Walkthrough: Optional 2D Explicit Finite Element Metallic Sheet Solver

We have successfully implemented the optional **2D explicit Finite Element (FE) metallic sheet solver** alongside the existing fabric solver, incorporating Q4 Reissner-Mindlin shell elements, J2 radial return plasticity, vertical corrugation mesh generation, and element erosion/deletion rupture dynamics.

## Changes Made

### 1. Configuration Validation & Parser (`src/kevlargrid/io/config.py`)
- Added validation and unit normalization for the new grid corrugation parameters:
  - `corrugation_amplitude` (metres)
  - `corrugation_period` (metres)
  - `corrugation_axis` (defaulting to `"x"`, supporting `"x"` or `"y"`)
- Added support for simulation structure type switching:
  - `structure_type` = `"fabric"` or `"metallic_sheet"`
- Added support for J2 plasticity material model parameters:
  - `material_model` = `"linear"` or `"j2_plasticity"`
  - `yield_strength_gpa` (GPa)
  - `hardening_modulus_gpa` (GPa)
  - `ultimate_strain` (element erosion equivalent plastic strain limit)
  - `poisson_ratio` (defaulting to `0.3`)

### 2. Rotational Kinetic Energy (`src/kevlargrid/solver/worker.py` and `projectile.py`)
- Updated the calculation of projectile kinetic energy (`initial_ke`, `final_ke` in the worker, and `residual_ke` in the impact report) to account for initial/final rotational kinetic energy:
  \[ KE_{rot} = 0.5 \sum (I_{diag} \omega^2) \]

### 3. Grid & Corrugation Mesh Generation (`src/kevlargrid/solver/grid.py`)
- Modified `generate_rectangular_grid` to support optional sinusoidal corrugations:
  \[ z_i \mathrel{+}= A \sin\left(\frac{2\pi \cdot \text{axis}_i}{P}\right) \]
- Computed spring rest lengths as the actual initial Euclidean distances between connected nodes so that the corrugated mesh begins completely stress-free.
- Generated Q4 quadrilateral element connectivity list `elements` of shape `(n_elements, 4)` for finite element force calculation.

### 4. 2D Explicit Reissner-Mindlin Shell Element Solver (`src/kevlargrid/solver/fused.py`)
- Implemented Flanagan-Belytschko hourglass-controlled bilinear quadrilateral Q4 shell element formulation in Numba space:
  - Handles 6 DOFs per node (3 translations, 3 rotations).
  - Evaluates membrane strains, bending curvatures, and transverse shear strains.
  - Simpson's rule integration through-thickness using 3 points.
- Implemented J2 radial return plasticity stress update at each thickness integration point.
- Implemented strain-based element erosion/deletion: if the equivalent plastic strain exceeds the ultimate strain limit at all 3 thickness integration points, the element is deleted, and its contribution to forces, moments, and contact mechanics drops to zero.
- Added automatic fallback to Numba backend when the user specifies `structure_type = "metallic_sheet"` and `backend = "taichi"`.

### 5. Strain Energy (`src/kevlargrid/solver/energy.py`)
- Updated `compute_strain_energy` to calculate strain energy from element elastic stresses when `elements` and `element_stress` are provided.

### 6. Built-in Corten Steel Preset & GUI Configuration Panel (`src/kevlargrid/gui/config_panel.py`)
- Added a researched `"Corten Steel (14 Gauge)"` preset for weathering steel shipping container sheets:
  - Elastic Modulus: $200.0\text{ GPa}$, Yield Strength: $0.345\text{ GPa}$, Tensile Strength: $0.485\text{ GPa}$, Density: $7.85\text{ g/cc}$, Areal Density: $15.7\text{ kg/m}^2$, Poisson's Ratio: $0.30$, Hardening Modulus: $1.0\text{ GPa}$, Ultimate Strain: $0.20$.
- Created a dynamic GUI Material Properties section:
  - Toggling **Structure Type** to `"Metallic Sheet"` hides fabric-only fields (e.g. Failure Strain, crimp, shear ratio, yarns) and shows metal-only fields (Material Model, Yield/Hardening Modulus, Ultimate Strain, Poisson's Ratio, and Thickness).
  - Toggling **Structure Type** to `"Metallic Sheet"` also hides the **Number of Plies** and **Analysis Mode** rows (since steel is a single-thickness sheet and not stacked fabric plies).
  - Automatically filters presets list to only show fabrics when `"Fabric"` is selected, and metallic presets when `"Metallic Sheet"` is selected.
- Updated `current_grid_key` tracking in `src/kevlargrid/gui/app.py` to watch corrugations, structure type, and thickness changes, allowing the interactive 3D viewport preview to update instantly and dynamically as sliders/inputs are adjusted.
- Added **Initial Orientation** (Roll, Pitch, and Yaw in degrees) and **Initial Rotation** (RPM) input fields to the GUI Projectile section. The configuration panel dynamically converts these inputs to quaternions and rad/s angular velocity for the backend solver.
- Guarded all propeller shape calculations in both the solver (`projectile.py`) and GUI (`viewport3d.py`) against division by zero when span is $0.0$.
- Fixed post-processing `ValueError` broadcasting crash by mapping element failures to spring failures in `worker.py` during strain evaluation.
- Fixed quaternion orientation integration bug in the shell solver JIT loop `_fused_shell_loop_jit` in `fused.py` by utilizing `numba_q_mul`.
- Added a labeled global Coordinate System (CSYS) tripod widget in both PyVista (using `add_axes()`) and fallback DearPyGui paths to easily identify spatial orientations.
- Shaded the projectile/propeller blade as a solid 3D surface with visible mesh edges (`style="surface"` with `show_edges=True`) rather than a hollow wireframe for a premium and detailed CAD-like appearance.
- Vectorized the mapping from element failures to spring failures in `Viewport3D.redraw` using pure NumPy array index slicing, restoring the GUI frame rate back to full speed (~1ms update).
- Fixed the post-simulation perforation and strain reporting `IndexError` in `worker.py` for metallic sheets.

---

## Verification Results

### 1. Automated Unit & Integration Tests
- Created `tests/unit/test_metallic_sheet.py` containing:
  - `test_grid_corrugation_and_elements`: Verifies Z-perturbation and element connectivity generation.
  - `test_config_validation_metallic_sheet`: Verifies config parsing and validation rules.
  - `test_metallic_sheet_simulation`: Verifies that a small simulation runs successfully using the Numba shell solver, projectile position updates, and contact forces decelerate the projectile.
  - `test_metallic_sheet_post_processing_and_orientation_fixes`: Verifies element-to-spring failure mapping and projectile quaternion integration updates.
- Ran the entire fast test suite locally:
  ```bash
  .venv/bin/pytest -m "not slow"
  ```
  **Result**: `100 passed, 1 skipped, 10 deselected in 50.83s`

### 2. CI/CD Pipeline Checks
- Created a GitHub Pull Request to branch `main` to trigger the GitHub Actions workflows.
- Verified that all CI checks passed successfully:
  - `lint` (ruff formatting, code quality, and mypy static analysis): **PASSED**
  - `test` (pytest unit tests): **PASSED**
