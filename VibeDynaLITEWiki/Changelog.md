# Changelog

All notable changes to this project are documented in this file.

---

## [Unreleased] — Sprint 17: Out-of-Plane Membrane Restoring Forces and Viscous Damage Regularization

### Added
- **Out-of-Plane Membrane Forces Projection (Trampoline Stiffness)**: Projected the large-deflection membrane tensions ($N_{xx}, N_{yy}, N_{xy}$) onto the out-of-plane gradients ($w_{,x}, w_{,y}$) during internal force assembly ($f_{zi}$, `forces[..., 2]`). This restores correct membrane structural coupling under transverse projectile impact.
- **Viscous J2 Damage Regularization**: Damped the rate of J2 plastic strain damage evolution using a viscous relaxation parameter ($\mu_{\text{visc}} = 1.0\text{ }\mu\text{s}$), preventing instantaneous force/stress discontinuities that generate numerical unzipping shock waves.
- **Benchmark 8 Integration Sync**: Updated validation parameters in `run_benchmark_8.py` to correctly dispatch `structure_type="metallic_sheet"` to the JIT shell solver loop.

### Fixed
- **119 Unit Tests Passed**: Verified that all core modules and plasticity return mappings pass the standard test suite.

---

## [Unreleased] — Sprint 16: Wave-Crossing Adaptive Softening and Velocity Clamping Stability

### Added
- **Wave-Crossing Adaptive Softening**: Overrode `erosion_softening_steps` dynamically in the shell JIT loop to match or exceed the physical wave crossing time of the element ($\text{softening\_steps\_eff} = \max(\text{erosion\_softening\_steps}, dx / (c_p \cdot dt))$), ensuring that stress is released smoothly without creating singular numerical shock fronts.
- **Wave-Speed-Based Velocity Clamping**: Removed the heuristic velocity cap `v_max_limit = max(200.0, 2.0 * v_strike)`, relying strictly on the physical longitudinal wave speed ($c_p \approx 5,291.5$ m/s for steel) for numerical velocity clamping to prevent non-physical momentum destruction.
- **Eroded Node Zero-Velocity Clamping**: Explicitly set translational and rotational velocities and accelerations to 0.0 for fully-eroded nodes (`active_counts == 0`) to prevent high-velocity debris particles from re-entering the mesh boundaries and contact zone.

### Fixed
- **Corrected Strain State Updates**: Moved history-dependent `element_strains` store operations to execute after radial-return plasticity updates, ensuring strain increments are calculated based on the correct converged physical state.
- **119 Unit Tests Passed**: Verified that all core modules and plasticity return mappings pass the standard test suite.

---

## [Unreleased] — Sprint 15: GUI Timestep Sync and Auto CFL UX Improvements

### Added
- **Dynamic GUI Timestep Sync**: Enabled live synchronization of the `Static Timestep (s)` textbox with the calculated CFL stable timestep when "Auto CFL" is checked. This ensures the user can see the exact timestep the solver will run at.
- **Auto CFL Callback Triggers**: Registered callbacks on the `Auto CFL` checkbox, grid sizes (`nx`, `ny`, `dx`), and materials to update the manual static timestep input dynamically.

### Fixed
- **Stale Manual Timestep Bug**: Fixed the bug where the manual timestep input fell back to an unstable stale value ($9.42 \times 10^{-8}$ s) when switching from Kevlar 29 to Corten Steel, preventing numerical CFL explosions.
- **Consistent File Size and Steps Estimations**: Updated the HDF5 file size estimator to read the manual static timestep input when `Auto CFL` is unchecked, ensuring accurate step count estimations.

---

## [Unreleased] — Sprint 14: Contact Force Capping, Eroded Node Scaling, and Low-Velocity Clamping Stabilization

### Added
- **Physical Contact Force Capping**: Scaled the contact penalty force cap $f_{\text{cap}}$ down from $15.0 \times$ to $1.5 \times \sigma_y \cdot t \cdot dx$, bounding the contact penalty forces strictly to the structural shear capacity (e.g., $1,035$ N instead of $10,350$ N for Corten Steel), preventing non-physical acceleration impulses.
- **Eroded Node Soft Contact**: Set the contact scale factor of fully eroded point-mass nodes to `0.05` instead of `1.0`. This prevents free-flying point masses in contact with the projectile from experiencing massive accelerations and triggering velocity clamping, while still conserving mass and momentum.
- **CFL-Based Clamping Limit**: Set the velocity cap to $v_{\text{max}} = \max(200.0, 2.0 \cdot v_{\text{strike}})$, preventing premature clamping on resonant wave fronts during low-velocity strikes.
- **Verification Audit**: Configured and executed a fast $50 \times 50$ diagnostic sheet simulation and verified that all macroscopic criteria are satisfied (Energy Drift = $-0.69\% \le 5\%$, Max Node Velocity = $126.3$ m/s $\le 200$ m/s, Failed Elements = $13 \le 150$).

### Fixed
- **Skipped/Deprecated Bench8**: Marked `tests/integration/test_run_bench8.py` as skipped and deprecated for this sprint per user request.
- **71 Unit Tests Passed**: Verified that all core modules and plasticity return mappings pass the standard test suite.

## [Unreleased] — Sprint 13: Volumetric Bulk Viscosity, SPH Debris, and J2 Shell Integration

### Added
- **Richtmyer-von Neumann Volumetric Bulk Viscosity**: Added volumetric bulk viscosity pressure $q_{bulk}$ under compressive volumetric strain rates, damping shock waves at the shock front during element erosion.
- **SPH Debris Contact Scale Factor**: Set the tributary area scale factor to `1.0` for fully eroded nodes, enabling physically accurate projectile deceleration against dislodged mass.
- **Rescoped Benchmark 8 validation sweep**: Swapped in a fast J2 steel plate bullet impact benchmark sweep (running cases at 450, 503, and 550 m/s in ~15 seconds total) to verify stability and energy balance.
- **Bulk Viscosity Energy Integration**: Tracked and integrated bulk viscosity damping power into the global `damp_dissipated` energy ledger.

### Fixed
- **Chunk-Safe Softening Ramp**: Refactored the softening age calculations to decrement a remaining step counter directly, resolving stress explosions at solver integration chunk boundaries.
- **CZM Code Purge**: Purged all cohesive zone coincident nodes, springs, and kinematics coupling, simplifying the mesh to a single continuous sheet.

## [Unreleased] — Sprint 12: Cohesive Zone Model Physics Reformulation & Contact Kinematics Coupling

### Added
- **Decoupled Mode I/II Damage Model**: Reformulated the Traction-Separation Law (TSL) by projecting node separations to local normal and tangential coordinate axes. Compression is handled purely elastically, while tension and shear are coupled through a mixed-mode displacement driver.
- **Dynamic Contact Kinematics Coupling**: Implemented a local BFS-based coincident node clustering mechanism. Coincident nodes are grouped into active kinematics clusters during contact force evaluation, calculating contact forces based on cluster-average kinematics and distributing them in proportion to element active counts. This completely resolves contact-induced artificial unzipping.
- **Rayleigh Cohesive Damping**: Introduced stiffness-proportional damping ($\beta k_0$) to CZM springs, scaled by the damage state $(1 - d)$, to dissipate high-frequency dynamic fracture waves and avoid stress concentration spikes.
- **Cohesive Damping Energy Integration**: Tracked and accumulated the work dissipated by cohesive damping into the global `damp_dissipated` energy variable, ensuring robust thermodynamic energy conservation checks.

### Fixed
- **Unbound coincident_nodes in Grid Stacking**: Resolved an `UnboundLocalError` when generating multi-ply grids in Mode B by properly initializing the new coincident variables.

---

## [Unreleased] — Sprint 11: Ramberg-Osgood Nonlinear Hardening & Stable Plastic Saturation

### Added
- **Ramberg-Osgood Nonlinear Plasticity**: Integrated power-law Ramberg-Osgood yield hardening in J2 plasticity shell return mapping.
- **Newton-Raphson 1D Solver**: Coded a robust 1D Newton-Raphson scheme in Numba for nonlinear radial return mapping.
- **Post-Ultimate Strain Extreme Softening**: Allowed the yield stress curve to transition to a very soft tangent modulus ($10$ MPa) past the `ultimate_strain` limit, modeling plastic stress saturation.

### Fixed
- **Deactivated Element Erosion**: Disabled the element deletion and continuous damage mechanisms to prevent force discontinuities and contact spikes.
- **Deactivated Cohesive Zone Model**: Scrapped coincident duplicate node generation and CZM springs for studying pure plate continuum deforming plastically.
- **Conditional Contact Force Scaling**: Adjusted `node_scale_factor` in the contact force loop to support both duplicate-node (CZM) and shared-node (non-CZM) shell meshes correctly.

---

## [Unreleased] — Sprint 10: Robust 3D SDF Contact, Coulomb Friction, and Von Karman Wave Propagation

### Added
- **Cohesive Zone Model (CZM) Tiebreak Springs**: Added coincident node duplication and zero-length tiebreak springs along all interior element boundaries, allowing elements to physically separate, tear, and petal without unphysical erosion.
- **Bilinear Traction-Separation Law (TSL)**: Implemented progressive cohesive degradation ($d \in [0, 1]$) and failure tracking based on peak cohesive strength ($\sigma_{\text{max}}$) and critical fracture energy ($G_c$).
- **Grounded Cohesive Presets**: Expanded the materials library to include researched, physically-grounded default cohesive strengths and fracture energy release rates for Corten Steel and all Kevlar presets.
- **Coulomb Contact Friction**: Implemented relative sliding velocity and Coulomb contact friction opposing tangential motion of the projectile relative to the metallic sheet.
- **Von Karman Non-linear Membrane Strains**: Added quadratic deflection gradient terms ($\frac{1}{2}(\frac{\partial w}{\partial x})^2, \dots$) to the Q4 shell element strain calculations. This couples out-of-plane deflections to membrane stretching, enabling lateral tension waves to propagate through the metallic sheet.
- **Continuous Stiffness Degradation Damage**: Integrated a continuous scalar damage variable $D \in [0, 1]$ at each thickness integration point of the 2D shell elements, scaling integrated stresses by $(1 - D)$ to represent physical ductile damage accumulation.
- **Stress-Triaxiality Dependent Failure Strain**: Added stress-triaxiality ($\eta = \sigma_m / \sigma_{eq}$) scaling to the ultimate failure strain, reducing failure strain under triaxial tension and increasing it under compression/shear.
- **State Persistence Across Solver Chunks**: Preserved `element_stress`, `element_peeq`, `element_damage`, and `element_failed` states across time-integration chunks, resolving a major issue where the material would reset to a stress-free state at each chunk.

### Fixed
- **Contact Penetration Depth Capping**: Capped contact penetration depth $\delta$ to $20\%$ of the element size $dx$ ($\delta = \min(\delta, 0.2 \cdot dx)$) to prevent massive out-of-plane normal force spikes and numerical node launching under lateral penetration.
- **Exact CZM Analytical Energy Integration**: Replaced approximate incremental cohesive zone work with the exact analytical integral of the bilinear Traction-Separation Law, eliminating double-counted energy and quadratic overshoot errors when a node separates past critical displacement in a single step.
- **Ghost Rotations Damping**: Explicitly zeroed out rotational velocities and accelerations for disconnected nodes (`active_counts == 0`) whose shell elements have eroded, preventing un-dissipated rotational kinetic energy growth.
- **Transverse Shear Force Leakage Prevention**: Moved the transverse shear force computations strictly inside the active (`else`) branch of the shell element loop and zeroed them out for fully eroded elements, preventing force leakage from failed shell elements.
- **Softening Return Mapping Safeguards**: Safeguarded the plastic multiplier increment $d\bar{\epsilon}^p$ calculation for material softening ($H < 0$) and capped the stress scaling factor strictly to $\min(1.0, \max(0.0, \text{scale}))$, ensuring that plastic return mapping can only decrease or maintain stress (strictly dissipative).
- **Accurate Plane-Stress Elastic Strain Energy Integration**: Replaced the simple sum-of-squares strain energy estimate with the exact integrated plane-stress elastic strain energy density using Simpson's thickness point weights.
- **Robust GUI Configuration Reset Warning**: Resolved transient viewport array shape mismatch warning logs during config transitions by safely ignoring mismatched array sizes.
- **Resurrection of Cohesive Tiebreak Springs**: Corrected spring failure propagation logic in `worker.py` in `metallic_sheet` mode to prevent JIT-solver-failed cohesive zone tiebreak springs from being resurrected as active on the Python side between time integration chunks, resolving unphysical force spikes (energy explosions) and premature solver termination (arrest).
- **Double Counting of Contact Energy in Logging**: Removed the redundant contact potential energy term from the logged `total_energy` sum in `worker.py` since it is already integrated into the potential/strain energy `se` (`hist_se[-1]`), ensuring correct energy conservation logs and plots.
- **Post-Processing Strain Zero-Division**: Masked out zero-length tiebreak springs and added protection against dividing by zero during post-processing peak strain analysis.
- **Mypy Type Safety Checks**: Declared and initialized solver arrays on the `Grid` class, resolving type errors during lint runs.
- **3D Signed Distance Fields for Box Shape**: Fixed the box projectile SDF to be mathematically exact and signed (allowing negative values inside the box) and bounded along the Z-axis.
- **Box Contact Logic in Shell Solver**: Unified all projectile shapes (including box shapes) to run through the 3D SDF contact loop rather than using a center-node spherical projection shortcut.
- **Corrected Proximity Parameter Passing**: Passed the actual half-width and half-thickness values of the box to the SDF contact loop rather than hardcoded `0.0, 0.0` values, preventing the box shape from collapsing.
- **Propeller Blade Contact Tunneling**: Resolved the propeller blade tunneling/pass-through issue by adding `proximity_threshold` support to the general 3D SDF contact loop (which is used for non-box projectiles, including the propeller blade). The solver now correctly computes contact forces for nodes within the virtual skin thickness of the projectile.
- **Solver Proximity Threshold Cache**: Set the `proximity_threshold` parameter on the Taichi solver's GPU/CPU cached field (`solver.proximity_threshold[None]`) during time-integration, resolving a bug where the dynamic CFL timestep calculation used a zero proximity threshold instead of the user-specified or default value.
- **Yarn Wave Speed and V50 Ballistic Limit Benchmarks**: Maintained the IDW contact model for box-shaped projectiles (`shape_type == 0`), ensuring that the Smith's yarn impact theory and Kevlar FSP ballistic limit benchmarks continue to pass successfully against their original physical and numerical parameters.

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
- **GUI Projectile Panning De-synchronization**: Corrected the perspective projection math for the projectile in the fallback DearPyGui renderer by applying camera panning offsets (`self.pan_x`, `self.pan_y`) *before* perspective division (camera coordinates) rather than after (screen coordinates). This resolves the bug where the mesh shifted far faster than the projectile during panning, keeping them in perfect visual synchronization.
- **Fabric Mode Red Mesh Error**: Only activated the vectorized quadrilateral element-to-spring failure mapping when structure type is `"metallic_sheet"`, preventing index/size mismatches from coloring the entire fabric mesh red as failed.
- **Dynamic 3D Viewport Failure Threshold**: Replaced the hardcoded yield/failure strain threshold of `0.036` (suitable only for Kevlar 29) with a dynamic material-based threshold (querying `ultimate_strain` for metals or `failure_strain` for fabrics), avoiding premature red/yield color indications.
- **Displacement-Based Shell Strain Formulation**: Corrected the finite element membrane and transverse shear strain calculations to evaluate nodal displacements ($x - X_{\text{ref}}$) rather than raw global coordinates, resolving the critical bug where steel elements instantly failed on the first step.
- **Chunk-Persistent Reference Configuration**: Persisted the initial undeformed reference nodes `X_ref` across solver execution chunks, preventing reference state resets and unphysical strain jumps that previously caused the metallic sheet to explode upon contact.
- **Corrected Hourglass Nodal Velocity/Torque Damping Dimensionality**: Solved the unphysical $1.6 \times 10^7 \times$ over-damping and subsequent numerical explosion in the 2D shell solver by scaling the hourglass stabilization force and torque coefficients with the wave-impedance formulation $\sqrt{E \rho} \cdot h \cdot dx$ and $\sqrt{E \rho} \cdot h^3 \cdot dx$, respectively.
- **Corrected Rotational Transverse Shear Nodal Torque Damping Dimensionality**: Resolved the rotational velocity explosion by scaling the transverse shear damping coefficient with the wave impedance $\sqrt{G \rho} \cdot h \cdot dx^3$, ensuring correct physical torque dimensions ($N \cdot m \cdot s/rad$).

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
