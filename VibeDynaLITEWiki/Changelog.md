# Changelog

All notable changes to this project are documented in this file.

---

## [Unreleased] — Sprint 9: 6-DOF Rotational Physics Correction, Labeled Global CSYS & Viewport redrawing stability

### Added
- **Labeled Global CSYS Tripod**: Added a labeled coordinate system tripod (X in Red, Y in Green, Z in Blue) to both PyVista and DearPyGui fallback viewfinders to visually align spatial directions.
- **Initial Orientation in Degrees**: Implemented Roll, Pitch, and Yaw in degrees in the GUI config panel (instead of quaternions) for intuitive user input, with automatic conversion to quaternions for solver input.
- **Initial Rotation in RPM**: Changed initial angular velocity units in the GUI to RPM, converting it to rad/s for integration.
- **Exact Rotational KE Feedback**: Calculated the exact initial rotational kinetic energy in the GUI config panel feedback based on the selected projectile shape and its principal moments of inertia.
- **Corten Steel in Fabric Mode**: Allowed the baseline "Corten Steel (14 Gauge)" preset to be selected and simulated in standard Fabric mode for immediate structural performance comparison.

### Fixed
- **Physically Rigorous 6-DOF Rotational Dynamics**: Fixed the global torque/inertia multiplication bug by rotating global torque and angular velocity into the body-fixed frame, integrating Euler's equations of motion with diagonal inertia components, and rotating the updated angular acceleration back to the global frame.
- **In-place Rotational Velocity (proj_omega) Updates**: Fixed JIT shell solver loop reassigning `proj_omega` instead of slice mutating it in-place (`proj_omega[:] = ...`), which previously blocked telemetry updates from capturing correct angular velocities.
- **UnboundLocalError in Subprocess Termination**: Resolved `UnboundLocalError: cannot access local variable 'reason'` by pre-initializing `reason = None` before the time integration loop starts.
- **GUI Projectile Panning De-synchronization**: Added camera panning offsets (`self.pan_x` and `self.pan_y`) to the projectile projection screen coordinates in the fallback DPG renderer, ensuring the projectile is panned/zoomed in sync with the grid nodes.
- **Fabric Mode Red Mesh Error**: Only activated the vectorized quadrilateral element-to-spring failure mapping when structure type is `"metallic_sheet"`, preventing index/size mismatches from coloring the entire fabric mesh red as failed.
- **Dynamic 3D Viewport Failure Threshold**: Replaced the hardcoded yield/failure strain threshold of `0.036` (suitable only for Kevlar 29) with a dynamic material-based threshold (querying `ultimate_strain` for metals or `failure_strain` for fabrics), avoiding premature red/yield color indications.
- **Displacement-Based Shell Strain Formulation**: Corrected the finite element membrane and transverse shear strain calculations to evaluate nodal displacements ($x - X_{\text{ref}}$) rather than raw global coordinates, resolving the critical bug where steel elements instantly failed on the first step.
- **Chunk-Persistent Reference Configuration**: Persisted the initial undeformed reference nodes `X_ref` across solver execution chunks, preventing reference state resets and unphysical strain jumps that previously caused the metallic sheet to explode upon contact.

---

## [Unreleased] — Sprint 8: 2D Explicit Finite Element Metallic Sheet Solver & Container Impacts

### Added
- **2D Explicit FE Shell Element Solver**: Implemented a Q4 Reissner-Mindlin shell element formulation with Flanagan-Belytschko hourglass control and 3-point Simpson's rule through-thickness integration.
- **J2 Radial Return Plasticity**: Integrated von Mises yield criterion and radial return stress mapping with isotropic hardening for metallic sheets.
- **Element Erosion / Deletion**: Added strain-based element erosion/deletion (rupture dynamics) when equivalent plastic strain exceeds the ultimate strain limit at all 3 thickness integration points.
- **Vertical Corrugated Mesh Generation**: Extended grid generator to support sinusoidal corrugations along the "x" or "y" axis to represent shipping container walls.
- **Built-in Corten Steel Preset**: Added researched weathering steel `"Corten Steel (14 Gauge)"` parameters.
- **Dynamic GUI Material & Grid Controls**: Swapping the structure type dynamically adjusts shown input fields, hides plies/stacking modes for steel, filters presets, and displays sheet thickness.
- **Rotational Kinetic Energy**: Updated projectile kinetic energy tracking to include $KE_{\text{rot}} = 0.5 \sum (I_{\text{diag}} \omega^2)$.

---

## [Unreleased] — Sprint 5: 3D Analytical SDF Contact & 6-DOF Kinematics

### Added
- **3D Analytical SDF Contact Solver**: Added support for sphere, cylinder (with edge rounding), bullet (tangent ogive with cylindrical body), and propeller blade (twisted, tapered span with rounded tip) shapes via analytical Signed Distance Fields (SDFs).
- **6-DOF Rigid Body Kinematics**: Formulated full translational and rotational dynamics using quaternions for orientation tracking, assuming uniform density to calculate volume, mass, and principal moments of inertia.
- **Node-to-Surface Penalty contact**: Integrated contact force calculation using the JIT compiler loops under Numba and Taichi backends.
- **Bazant Strain Regularization**: Resolved mesh dependency issues at small grid spacing ($dx < 1.0\text{ mm}$) using regularized failure strain $\epsilon_{\text{fail}} = \epsilon_0 \sqrt{h_0 / dx}$.
- **6-DOF Projectile Telemetry Dashboard**: Created a dedicated telescoping summary table detailing calculated volume, diagonal inertia tensors, velocities, angular velocities, and orientation quaternions.
- **Dynamic shape config panel**: Implemented combo box shape dropdown and shape-specific input fields (Sphere radius, Cylinder edge-radius, Bullet ogive, Propeller twist/span/tip-radius).
- **Physics Benchmarks 9, 10, 11**: Added free flight energy conservation validation, oblique impact tumbling dynamics, and mesh refinement V50 convergence benchmarks, with associated unit/integration tests and wiki documentation.

---

## [Unreleased] — Sprint 7.14: Dynamic CFL Timestep & Thermodynamically Consistent Energy Accounting

### Added
- **JAX JIT CFL Compilation Unit Tests**: Added [test_jax_jit_cfl.py](file:///Users/bennames/Developer/VibeDynaLITE/tests/solver/test_jax_jit_cfl.py) to verify compile-time JIT validation under the JAX backend for both dynamic and static CFL configurations.
- **Rupture Energy Plotting**: Exposed `"Rupture Energy"` (progressive damage dissipated energy) as a crimson-red line series in the GUI's `System Energy Telemetry History` plot.
- **Detailed Energy Serialization**: Added `"failure_dissipated"` (fiber breakage/rupture) and `"clamp_dissipated"` (velocity clamping) to HDF5 binary trajectories and CSV spreadsheets.
- **Active-Only Peak Strain History**: Added frame-by-frame tracking of peak fabric yarn strains computed exclusively across active (non-ruptured) springs.

### Fixed
- **JAX Solver Tracer Crash**: Resolved `TracerBoolConversionError` when evaluating the dynamic CFL branch `if cfl_factor > 0.0:` by adding `"cfl_factor"` to JAX `static_argnames` in the `@backend.jit` decorator in [fused.py](file:///Users/bennames/Developer/VibeDynaLITE/src/kevlargrid/solver/fused.py).
- **Unphysical Peak Strain Growth**: Stopped unruptured peak strain values from exploding post-failure by applying the `~grid.failed` mask before computing the maximum strain in [worker.py](file:///Users/bennames/Developer/VibeDynaLITE/src/kevlargrid/solver/worker.py).
- **Total Energy Telemetry Balance**: Fixed the phantom total energy rise on telemetry plots. Recalculated total energy by summing all conservative and non-conservative components (Fabric KE + Projectile KE + Strain Energy + Damped Energy + Rupture Energy + Clamping Energy).
- **Stepwise Telemetry History Jumps**: Linearly interpolated cumulative damping, rupture, and clamping energies across history frames in [app.py](file:///Users/bennames/Developer/VibeDynaLITE/src/kevlargrid/gui/app.py) to ensure smooth lines during post-run playback and scrubbing.
- **Damage-Aware Telemetry Strain Energy**: Updated the live progress telemetry in [worker.py](file:///Users/bennames/Developer/VibeDynaLITE/src/kevlargrid/solver/worker.py) to pass the progressive damage fraction array to `compute_strain_energy`, matching the JIT solver loop logic.
- **CSV Exporter Backward Compatibility**: Shifted new CSV columns (Rupture/Clamping Energy) to the end of the CSV headers to prevent index offsets from breaking existing test suites.

### Changed
- **Dynamic CFL Timestep**: Timestep calculation ($dt$) is now computed dynamically at each step based on the actual maximum connected nodal stiffness ($K_{\text{total}, i}$), which dynamically shrinks when contact is active and expands when inactive, keeping simulations stable at the default `cfl_factor = 0.8`.
- **Degraded Strain Energy Formulation**: Updated strain energy calculations in [energy.py](file:///Users/bennames/Developer/VibeDynaLITE/src/kevlargrid/solver/energy.py) to use degraded stiffness: $SE = \frac{1}{2} k (1 - D) x^2$.
- **Continuous Dissipated Energy Tracking**: Implemented continuous thermodynamic work tracking during the progressive damage phase in both Numba (CPU) and Taichi (GPU) integration loops.

---

## [0448266] — 2026-06-11
### Fixed
- Corrected rest lengths grid-order mismatch (half mesh blue/red color bug).
- Synchronized camera orientation/pan/zoom in 3D viewport with exported video frames.

## [8636389] — 2026-06-11
### Optimized
- Refactored 3D video export to use PyVista off-screen GPU rendering with fallback to vectorized Line3DCollection in Matplotlib.

## [dd3cccc] — 2026-06-11
### Fixed
- Resolved VideoExporter ply count mismatch in Mode A causing empty MP4 files.

## [ca41843] — 2026-06-10
### Fixed
- Forced H.264 codec and YUV420p pixel format for QuickTime video compatibility.

## [94ed7a7] — 2026-06-10
### Fixed
- Removed double-specified FPS parameters in movie writer calls.
