# Element Erosion Instability

When elements or springs erode (fail) in an explicit dynamic simulation, the sudden loss of stiffness and mass can trigger severe numerical instabilities. In **VibeDynaLITE**, this manifests as nodes being launched at extreme velocities (e.g., kinetic energy jumping from a few joules to millions of joules in a single chunk).

This document serves as a detailed learning guide and reference for understanding, preventing, and auditing element/spring erosion instabilities in VibeDynaLITE.

---

## 1. Physical & Numerical Mechanisms of Instability

### 1.1 The "Element Erosion Shotgun" (Force Discontinuity)
When a spring (in fabric mode) or a shell element (in metallic sheet mode) reaches its failure threshold, its internal stress is immediately zeroed to represent material rupture. 

* **Before failure**: The element/spring is carrying a large tensile load $F = k \cdot \varepsilon \cdot L_0$, which is balanced by neighboring elements/springs to maintain nodal equilibrium.
* **At failure**: The element's forces drop to zero instantly. However, the neighboring elements continue to exert their full forces on the shared nodes. This creates a massive **unbalanced force impulse**.
* **Result**: The shared nodes undergo extreme acceleration:
  \[
  a = \frac{F_{\text{unbalanced}}}{m_{\text{node}}}
  \]
  Because explicit time integration step size $dt$ is on the order of nanoseconds, these huge accelerations launch nodes at velocities far exceeding physical limits (creating the "shotgun" effect).

### 1.2 Uncapped Contact Force Spikes
When springs/elements fail at the impact zone, the projectile punches through the resulting gap.
* Nodes adjacent to the failure zone suddenly experience deep projectile penetration ($\delta$).
* In the penalty contact formulation:
  \[
  F_{\text{contact}} = k_{\text{penalty}} \cdot \delta + c \cdot \dot{\delta}
  \]
  Because there is no upper bound on $\delta$ or $F_{\text{contact}}$, deep penetrations under high-speed impact generate massive contact forces.
* If a node is completely freed from all active elements (meaning it has zero structural stiffness), but contact forces are still applied to it, the node gains massive energy without any restoring structural forces to arrest it.

### 1.3 State Persistence Reset (Numba/Taichi JIT Boundaries)
A critical implementation issue occurs when state arrays are not correctly passed between Python and the JIT solver across chunks:
* If arrays like `element_strains` or rotational DOFs (`ang_positions`, `ang_velocities`) are initialized to zero at the start of each integration chunk rather than being persisted, the solver interprets the sudden step change as a massive deformation impulse.
* On step 0 of a new chunk, the strain increment $\Delta \varepsilon = \varepsilon_{\text{current}} - \varepsilon_{\text{old}}$ equals the entire current strain (since $\varepsilon_{\text{old}} = 0$). This spikes stresses to several GPa, launching nodes and injecting huge amounts of spurious energy.

### 1.4 Phantom Energy Accounting
If the dissipated fracture/failure energy is computed analytically or globally rather than incrementally:
* Recomputing the total dissipated work from scratch each step can introduce enormous numerical artifacts.
* Furthermore, if the solver zero-outs failed forces but does not explicitly subtract/absorb the stored elastic strain energy from the nodes, the system "creates" energy because the nodes retain their deformation-driven kinetic energy while the accounting ledger records it as fully dissipated.

---

## 2. Best-Practice Mitigation Strategies (LS-DYNA Comparison)

To stabilize erosion and prevent non-physical energy growth, explicit dynamic codes employ several key numerical treatments:

| Instability Cause | LS-DYNA Implementation | VibeDynaLITE Target Formulation |
| :--- | :--- | :--- |
| **Force Discontinuity** | `SOFT` option in `*MAT_ADD_EROSION` (gradual force ramp-down over 5–10 steps) | **Softened Spring/Element Deletion**: Linearly decay forces from the failure point to zero over $N_{\text{ramp}}$ steps. |
| **Freed Node Contact** | Eroded nodes are removed from contact slave/master surfaces | **Dead-Node Contact Exclusion**: Skip penalty contact calculations for nodes where active element/spring count is 0. |
| **Unlimited Contact Force** | Segment-based contact with depth limits and $F_{\text{max}}$ caps | **Penalty Force Capping**: Limit $F_{\text{contact}}$ to the structural shear capacity of the local element. |
| **Spurious Velocity** | Node velocity capped by material wave speed ($c_p$) | **Longitudinal Wave Speed Clamping**: Clamp velocities at $2 \times c_p$ instead of the generous numerical limit $dx/dt$. |
| **Chunk Wave Reset** | Continuous history tracking throughout the time loop | **Persistent Solver State**: Pass and return all history-dependent arrays across the chunk boundary. |

---

## 3. Formulas for Softened Deletion

To implement softened deletion, each failed spring/element must track its "failure age" (number of timesteps since failure). The force scaling factor $\mathcal{S}$ is defined as:
\[
\mathcal{S} = \max\left(0.0, 1.0 - \frac{t - t_{\text{fail}}}{N_{\text{ramp}} \cdot dt}\right)
\]
During the softening phase:
1. The force applied to the nodes is scaled by $\mathcal{S}$:
   \[
   F_{\text{applied}} = \mathcal{S} \cdot F_{\text{nominal}}
   \]
2. The remaining strain energy is incrementally dissipated as work of failure rather than being dumped into nodal kinetic energy.
3. Once $t - t_{\text{fail}} \ge N_{\text{ramp}} \cdot dt$, the element is fully eroded, and $\mathcal{S} = 0.0$.

---

## 4. Solver Hardening Updates (July 2026)

To resolve cascading failures and non-physical energy generation, the explicit solver was hardened with the following exact formulations:

### 4.1 Contact Penetration Capping
To prevent unphysical out-of-plane forces when projectile nodes tunnel deeply into the mesh boundaries, the penetration depth $\delta$ is capped at $20\%$ of the element size $dx$:
\[
\delta = \min(\delta, 0.2 \cdot dx)
\]
This prevents numerical force spikes while preserving the structural capacity limit $f_{\text{cap}} = \sigma_y \cdot dx \cdot t$.

### 4.2 Exact Cohesive Zone Model (CZM) Analytical Energy
Instead of adding approximate incremental work $F \cdot dv \cdot dt$ (which double-counts energy and is prone to quadratic overshoot errors if a node separates past $\delta_c$ in a single timestep), we integrate the analytical envelope of the bilinear Traction-Separation Law. The dissipated energy $E_{\text{diss}}$ as a function of damage $D \in [0, 1]$ is:
\[
E_{\text{diss}}(D) = \frac{1}{2} k_0 \delta_0^2 \frac{\delta_c D}{\delta_c - D(\delta_c - \delta_0)} \quad (D < 1.0)
\]
\[
E_{\text{diss}}(1.0) = \frac{1}{2} k_0 \delta_0 \delta_c = G_c
\]
At each timestep, the change in energy $\Delta E_{\text{diss}} = E_{\text{diss}}(D_{\text{new}}) - E_{\text{diss}}(D_{\text{old}})$ is accumulated, preventing all energy ledger drift.

### 4.3 Ghost Rotations & Transverse Shear Leakage
* **Ghost Rotations**: When all elements sharing a node are deleted, the node's translational contact is bypassed, but its rotational degrees of freedom can continue spinning endlessly due to zero torque damping. The solver now explicitly zeros out angular velocities and accelerations for any nodes where `active_counts == 0`.
* **Shear Leakage**: Previously, the transverse shear forces $Q_x, Q_y$ were computed unconditionally and bypassed the erosion `ramp`. These are now computed strictly inside the active (`else`) block of the element loop, preventing force leakage from failed shell elements.

### 4.4 Softening Return Mapping Safeguards
For J2 radial return plasticity with material softening ($H < 0$), the plastic multiplier increment $d\bar{\epsilon}^p$ is safeguarded against negative denominators:
\[
d\bar{\epsilon}^p = \begin{cases} 
\frac{f_{\text{yield}}}{3G + H} & \text{if } 3G + H > 0 \\
0 & \text{otherwise}
\end{cases}
\]
Additionally, the stress scaling factor $\text{scale} = 1 - \frac{3G d\bar{\epsilon}^p}{\sigma_{\text{vm}}^{\text{trial}}}$ is strictly bounded:
\[
\text{scale} = \min(1.0, \max(0.0, \text{scale}))
\]
This guarantees that plastic return mapping can only decrease or keep stress constant, preventing spurious energy injection.

---

## 5. Reactivation and Upgrade of Shell Element Erosion (Current Stage)

To accurately capture high-velocity perforation, plugging, and fragmentation, shell element erosion has been reactivated and significantly upgraded:

### 5.1 Damage Evolution Onset Threshold
To prevent premature cascading damage across the plastic zone, ductile J2 damage now strictly evolves only when the local equivalent plastic strain $PEEQ$ exceeds the damage onset strain ($ultimate\_strain$):
* **Hardening Phase ($PEEQ \le ultimate\_strain$)**: No damage accumulates, and the material retains its full carrying capacity.
* **Softening Phase ($PEEQ > ultimate\_strain$)**: Damage evolves incrementally based on the plastic strain increment $d\_peeq$ and fracture energy $G_f$.

This prevents the initial plastic deformation wave from weakening the entire sheet and resolves the unzipping/cascading failure wave issue.

### 5.2 Multi-Point Shell Failure Criterion
In shell bending, the neutral axis (the center integration point, $k=2$) remains near zero strain, which previously caused the element to never fail despite full cracking of the outer layers (neutral-axis lock-up). We now employ a multi-point shell failure criterion:
An element initiates softening and eventual erosion when:
1. **Average Damage**: The mean damage across all 5 thickness points is $\ge 70\%$.
2. **Neutral Axis Damage**: The damage at the center integration point $k=2$ is $\ge 90\%$.
3. **Outer Surface Damage**: Both outer integration points (0 and 4) are $\ge 95\%$ damaged.

This physically and numerically represents through-thickness rupture and necking, allowing the element to enter the softened erosion phase smoothly.

### 5.3 SPH Point-Mass Debris Contact Conversion
Once all elements connected to a node have fully eroded, the node is not deleted. Instead:
1. **SPH Particle Conversion**: The node becomes a free point-mass SPH particle.
2. **Contact Retention**: The SPH particle continues to undergo contact with the projectile to conserve mass and momentum.
3. **Contact Scale Factor Correction**: Rather than scaling the force down by $1 / N_{initial\_elements}$, the contact scale factor is set to `1.0` for fully eroded nodes (`active_counts == 0.0`), representing a full dislodged particle mass that decelerates the projectile physically.
4. **Ghost Rotations Mitigation**: Rotational degrees of freedom (angular velocity and acceleration) are set to exactly 0 to prevent numerical rotation spikes.

### 5.4 Richtmyer-von Neumann Volumetric Bulk Viscosity
To suppress numerical shock waves generated by element deletion, we implemented Richtmyer-von Neumann volumetric bulk viscosity:
* Calculates a viscous pressure $q_{bulk}$ when the volumetric strain rate $\dot{\epsilon}_{vol} = \dot{\epsilon}_{xx} + \dot{\epsilon}_{yy} < 0.0$ (compressive volumetric strain rate):
  $$q_{bulk} = \rho \cdot dx \cdot \left( 0.06 \cdot c_s \cdot |\dot{\epsilon}_{vol}| + 1.5 \cdot dx \cdot \dot{\epsilon}_{vol}^2 \right)$$
* Subtracted from the normal stresses during internal force calculations, successfully damping shock fronts and preventing spurious cascading failures.
* Viscous work is fully tracked in the damping energy diagnostics ledger (`step_stiff_damp_power`).

### 5.5 Chunk-Safe Softening Ramp
To prevent force and stress explosions when element softening spans across integration chunk boundaries:
* Refactored the softening age calculations to decrement a remaining step counter directly: `element_failed_step[e] -= 1` each step, starting from `erosion_softening_steps`.
* The stress softening ramp is computed as:
  $$\text{ramp} = \frac{\text{remaining\_steps}}{\text{erosion\_softening\_steps}}$$
* This guarantees that the softening behavior is perfectly persistent and independent of boundary step counter resets.

### 5.6 Penalty Contact Force Capping & Eroded Node Scaling (July 2026 Sprint 14)
To resolve the spurious energy pump in low-velocity impacts (e.g. 100 m/s) where velocity clamping artificially traps nodes within the projectile, three critical updates were implemented:
1. **Physical Contact Cap**: The contact penalty force cap $f_{\text{cap}}$ is scaled down from $15.0 \times$ to $1.5 \times \sigma_y \cdot t \cdot dx$. This bounds the penalty contact forces strictly to the structural shear capacity (e.g. $1,035$ N instead of $10,350$ N), preventing node velocity launch impulses.
2. **Eroded Node Contact Softening**: When all elements attached to a node erode (`active_counts == 0.0`), the node is retained as an SPH debris point-mass, but its contact scale factor is set to `0.05`. This soft contact prevents the free point-mass from experiencing massive penalty forces that would trigger numerical launch instabilities, while still conserving mass and momentum.
3. **CFL-Based Velocity Clamping Limit**: The velocity cap is set to $v_{\text{max}} = \max(200.0, 2.0 \cdot v_{\text{strike}})$, ensuring that low-velocity impacts do not trigger premature clamping on resonant wave fronts.

### 5.7 Physical Wave-Speed Clamping & Adaptive Softening (July 2026 Sprint 16)
To resolve cascading failures caused by unphysical wave clipping and shock front propagation:
1. **Wave-Speed-Based Velocity Clamping**: Removed the heuristic $v_{\text{max\_limit}} = \max(200.0, 2.0 \cdot v_{\text{strike}})$ velocity cap. The solver now relies strictly on the physical longitudinal wave speed of the material ($c_p \approx 5,291.5$ m/s for steel) for numerical velocity clamping. This prevents non-conservative momentum destruction of physical snap-back waves and Poisson reflections.
2. **Wave-Crossing Adaptive Softening**: Instead of dropping internal stresses over a hardcoded 10 steps, the solver dynamically overrides `erosion_softening_steps` to match or exceed the physical wave crossing time of the element:
   $$\text{softening\_steps\_eff} = \max\left(\text{erosion\_softening\_steps}, \frac{dx}{c_p \cdot dt}\right)$$
   This ensures that stresses are released smoothly over at least one full wave-crossing duration, preventing the formation of unphysical singular shock fronts that drive adjacent elements to fail in a chain reaction.
3. **Corrected Strain State Updates**: Moved history-dependent `element_strains` store operations to execute after radial-return plasticity updates, ensuring strain increments are calculated based on the correct converged physical state.
4. **Eroded Node Zero-Velocity Clamping**: Set translational and rotational velocities and accelerations to exactly 0.0 for fully-eroded nodes (`active_counts == 0`) to prevent high-velocity debris particles from re-entering the contact zone.

### 5.8 Out-of-Plane Membrane Restoring Forces & Viscous Damage Regularization (July 2026 Sprint 17)
To resolve unphysical cascading failures under high-energy impacts and correct shell solver kinematics:
1. **Out-of-Plane Membrane Forces Projection (The Trampoline Effect)**:
   Under large out-of-plane deflections ($w$), membrane tensions ($N_{xx}, N_{yy}, N_{xy}$) must resist out-of-plane motion. We project these tensions onto the mid-surface slope gradients ($w_{,x}, w_{,y}$):
   \[
   Q_x^{\text{eff}} = Q_x + N_{xx} w_{,x} + N_{xy} w_{,y}
   \]
   \[
   Q_y^{\text{eff}} = Q_y + N_{yy} w_{,y} + N_{xy} w_{,x}
   \]
   These effective shear resultants are used for out-of-plane nodal force assembly ($f_{zi}$, `forces[..., 2]`), restoring the membrane stiffness response to dynamic transverse impact.
2. **Viscous Damage Regularization**:
   To prevent unphysical high-frequency shock waves generated during element deletion, we damp the damage rate using a viscous relaxation parameter ($\mu_{\text{visc}} = 1.0\text{ }\mu\text{s}$):
   \[
   d^{n+1} = d^n + \left(\frac{dt}{dt + \mu_{\text{visc}}}\right) \Delta d
   \]
   This regularizes damage growth over time, smoothing the local stress drop and stopping domino-like unzipping of adjacent elements.

### 5.9 Objective Green-Lagrange Strain Kinematics & Work-Conjugate Forces (July 2026 Sprint 18)
To resolve the remaining numerical energy leakage/pump under large rotations, the flat Q4 shell element solver's kinematics was upgraded:
1. **Objective Green-Lagrange Strain Tensor**:
   The trigonometric-clamped Von Karman strain was replaced by the exact, frame-invariant Green-Lagrange strain tensor to capture large rotations correctly:
   $$\epsilon_{xx} = u_{,x} + \frac{1}{2}(u_{,x}^2 + v_{,x}^2 + w_{,x}^2)$$
   $$\epsilon_{yy} = v_{,y} + \frac{1}{2}(u_{,y}^2 + v_{,y}^2 + w_{,y}^2)$$
   $$\gamma_{xy} = u_{,y} + v_{,x} + u_{,x} u_{,y} + v_{,x} v_{,y} + w_{,x} w_{,y}$$
2. **Work-Conjugate Nodal Force Projections**:
   To satisfy energy conservation, internal forces are assembled using the work-conjugate projections of the membrane stresses ($N_{xx}, N_{yy}, N_{xy}$):
   $$T_{xx} = N_{xx} (1 + u_{,x}) + N_{xy} u_{,y}, \quad T_{xy} = N_{yy} u_{,y} + N_{xy} (1 + u_{,x})$$
   $$T_{yx} = N_{xx} v_{,x} + N_{xy} (1 + v_{,y}), \quad T_{yy} = N_{yy} (1 + v_{,y}) + N_{xy} v_{,x}$$
   $$Q_{x,\text{eff}} = Q_x + N_{xx} w_{,x} + N_{xy} w_{,y}, \quad Q_{y,\text{eff}} = Q_y + N_{yy} w_{,y} + N_{xy} w_{,x}$$
   This conservative force projection ensures that no numerical energy is pumped or leaked under coordinate noise or wave fronts, preventing artificial cascading failures.
3. **`element_strains` Persistence Fix**:
   Passed and persisted the `element_strains` history array across integration chunks in the validation benchmarks, preventing spurious strain-increment spikes at chunk boundaries.

