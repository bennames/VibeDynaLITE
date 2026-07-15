# 2D Shell Finite Elements

To support modeling metallic sheets, container walls, and other continuous thin-walled structures, VibeDynaLITE includes a **2D Explicit Finite Element (FE) Shell Solver** alongside the traditional mass-spring lattice formulation. 

---

## Element Formulation

The solver utilizes a **Q4 Reissner-Mindlin Bilinear Quadrilateral Shell Element** to capture both membrane and bending behavior:

1. **Degrees of Freedom**: Each node has 6 degrees of freedom:
   - 3 translational: $u_x, u_y, u_z$
   - 3 rotational: $\theta_x, \theta_y, \theta_z$
2. **Kinematics**: Transverse shear deformation is included based on Reissner-Mindlin theory, allowing the element to remain valid for both thin and moderately thick plates.
3. **Geometric Non-Linearity (Green-Lagrange Strain Tensor & Work-Conjugate Forces)**: To model membrane-bending coupling under large out-of-plane deflections objectively and conservatively under arbitrary rotations, the solver utilizes the Green-Lagrange strain tensor:
   $$\epsilon_{xx} = \frac{\partial u}{\partial x} + \frac{1}{2}\left[\left(\frac{\partial u}{\partial x}\right)^2 + \left(\frac{\partial v}{\partial x}\right)^2 + \left(\frac{\partial w}{\partial x}\right)^2\right]$$
   $$\epsilon_{yy} = \frac{\partial v}{\partial y} + \frac{1}{2}\left[\left(\frac{\partial u}{\partial y}\right)^2 + \left(\frac{\partial v}{\partial y}\right)^2 + \left(\frac{\partial w}{\partial y}\right)^2\right]$$
   $$\gamma_{xy} = \frac{\partial u}{\partial y} + \frac{\partial v}{\partial x} + \frac{\partial u}{\partial x}\frac{\partial u}{\partial y} + \frac{\partial v}{\partial x}\frac{\partial v}{\partial y} + \frac{\partial w}{\partial x}\frac{\partial w}{\partial y}$$
   To satisfy energy conservation and frame invariance under large rotations, the internal nodal forces are evaluated using the work-conjugate projections of the membrane stresses:
   $$T_{xx} = N_{xx} \left(1 + \frac{\partial u}{\partial x}\right) + N_{xy} \frac{\partial u}{\partial y}, \quad T_{xy} = N_{yy} \frac{\partial u}{\partial y} + N_{xy} \left(1 + \frac{\partial u}{\partial x}\right)$$
   $$T_{yx} = N_{xx} \frac{\partial v}{\partial x} + N_{xy} \left(1 + \frac{\partial v}{\partial y}\right), \quad T_{yy} = N_{yy} \left(1 + \frac{\partial v}{\partial y}\right) + N_{xy} \frac{\partial v}{\partial x}$$
   $$Q_{x,\text{eff}} = Q_x + N_{xx} \frac{\partial w}{\partial x} + N_{xy} \frac{\partial w}{\partial y}, \quad Q_{y,\text{eff}} = Q_y + N_{yy} \frac{\partial w}{\partial y} + N_{xy} \frac{\partial w}{\partial x}$$
   This mathematically rigorous formulation ensures that the internal forces remain conservative (zero energy leakage/generation) during dynamic wave propagation, eliminating self-excitation and artificial cascading failures.
4. **Bending Curvatures**: Curvatures are evaluated from nodal rotation gradients:
   $$\kappa_{xx} = \frac{\partial \theta_y}{\partial x}, \quad \kappa_{yy} = -\frac{\partial \theta_x}{\partial y}, \quad \kappa_{xy} = \frac{\partial \theta_y}{\partial y} - \frac{\partial \theta_x}{\partial x}$$
5. **Transverse Shear Strains**: Transverse shear strains allow for cross-sectional rotation relative to the mid-surface normal:
   $$\gamma_{xz} = \frac{\partial w}{\partial x} + \theta_y, \quad \gamma_{yz} = \frac{\partial w}{\partial y} - \theta_x$$
6. **Total In-Plane Strains**: The total in-plane strain field at a distance $z$ from the shell mid-surface is:
   $$\epsilon_{xx}(z) = \epsilon_{xx} + z \kappa_{xx}, \quad \epsilon_{yy}(z) = \epsilon_{yy} + z \kappa_{yy}, \quad \gamma_{xy}(z) = \gamma_{xy} + z \kappa_{xy}$$
7. **Numerical Integration**: In-plane integration is performed using a single Gauss point at the center of the element, combined with **Flanagan-Belytschko hourglass control** to suppress spurious zero-energy modes. The stabilization forces and torques are scaled using physical wave-impedance properties to prevent numerical explosions:
   - Translational Hourglass Damping: $C_{\text{damp}} = 0.015 \sqrt{E \rho} \cdot h \cdot dx$
   - Rotational Hourglass Damping: $C_{\text{rot\_damp}} = 0.015 \sqrt{E \rho} \cdot h^3 \cdot dx$
   - Transverse Shear Hourglass Damping: $C_{\text{shear\_damp}} = 0.015 \sqrt{G \rho} \cdot h \cdot dx^3$
8. **Through-Thickness Integration**: Integration through the thickness is performed using **Simpson's rule** with 3 integration points ($z_k \in \{-0.5h, 0, 0.5h\}$ with weights $w_k \in \{h/6, 4h/6, h/6\}$) to capture bending and nonlinear material response accurately.

---

## J2 Radial Return Plasticity with Ramberg-Osgood Hardening

For metallic materials like Corten Steel, a **J2 radial return plasticity model with Ramberg-Osgood nonlinear hardening** is implemented at each through-thickness integration point:

1. **Stress-Strain Update**: Strains and curvatures are evaluated from nodal displacement rates. An elastic trial stress is computed under plane-stress assumptions:
   $$\sigma_{xx}^{\text{trial}} = \sigma_{xx}^n + \frac{E}{1-\nu^2} (\Delta \epsilon_{xx} + \nu \Delta \epsilon_{yy})$$
   $$\sigma_{yy}^{\text{trial}} = \sigma_{yy}^n + \frac{E}{1-\nu^2} (\Delta \epsilon_{yy} + \nu \Delta \epsilon_{xx})$$
   $$\tau_{xy}^{\text{trial}} = \tau_{xy}^n + G \Delta \gamma_{xy}$$
2. **Yield Criterion**: The von Mises equivalent trial stress $\sigma_{\text{vm}}^{\text{trial}}$ is compared to the yield strength $\sigma_y(e_{\text{peeq}})$.
3. **Ramberg-Osgood Hardening Curve**: The yield stress varies nonlinearly with equivalent plastic strain $p$:
   - **For $p \le \epsilon_u$** (under ultimate strain):
     $$\sigma_y(p) = \sigma_{y,0} + K_{\text{RO}} \left[ (p + \epsilon_{\text{reg}})^{0.2} - \epsilon_{\text{reg}}^{0.2} \right]$$
     where $K_{\text{RO}}$ is fit to match the ultimate tensile strength $\sigma_u$ at $p = \epsilon_u$, and $\epsilon_{\text{reg}} = 10^{-5}$.
   - **For $p > \epsilon_u$** (post-ultimate extreme softening):
     $$\sigma_y(p) = \sigma_u + H_{\text{soft}} (p - \epsilon_u)$$
     where $H_{\text{soft}} = 10\text{ MPa}$ is an extremely small tangent modulus to model stress saturation.
4. **Newton-Raphson Return Mapping**: Stresses are radially scaled back to the yield surface. The plastic multiplier increment $\Delta e_{\text{peeq}}$ is solved iteratively using a 1D Newton-Raphson scheme:
   $$g(\Delta e_{\text{peeq}}) = \sigma_{\text{vm}}^{\text{trial}} - 3G \Delta e_{\text{peeq}} - \sigma_y(e_{\text{peeq}}^n + \Delta e_{\text{peeq}}) = 0$$
   $$\sigma^{n+1} = \left(1 - \frac{3G \Delta e_{\text{peeq}}}{\sigma_{\text{vm}}^{\text{trial}}}\right) \sigma^{\text{trial}}$$
5. **Plastic Dissipation**: The energy dissipated by plastic work is accumulated:
   $$\Delta W_{\text{plastic}} = \sigma_y(e_{\text{peeq}}^{n+1}) \Delta e_{\text{peeq}} \cdot w_k \cdot dx^2$$

---

## Continuous Damage Mechanics (CDM) & Stress Triaxiality

> [!WARNING]
> **Status: Deactivated / Scrapped**
> Element damage accumulation, stiffness degradation, and element erosion are currently deactivated in the active solver to study nonlinear plastic saturation under the Ramberg-Osgood model.

To model progressive failure and ductile tearing when activated, VibeDynaLITE integrates a scalar continuous damage variable $D \in [0, 1]$ at each through-thickness integration point:
1. **Damage Accumulation**: $\Delta D = \frac{\Delta e_{\text{peeq}}}{\epsilon_f}$
2. **Triaxiality-Dependent Failure Strain ($\epsilon_f$)**: The ultimate failure strain scales according to the local stress triaxiality state $\eta$:
   - **Tension ($\eta > 0$)**: $\epsilon_f = \epsilon_{\text{ult}} \exp\left(-1.5\left(\eta - \frac{1}{3}\right)\right)$
   - **Compression & Shear ($\eta \le 0$)**: $\epsilon_f = \epsilon_{\text{ult}} \exp(-0.5\eta)$
3. **Stiffness Degradation**: Stresses are degraded: $\sigma_{\text{total}} = \sigma_{\text{nominal}} (1 - D)$.
4. **Element Erosion**: Elements are deleted when $D \ge 1.0$ at all 3 thickness integration points.

---

## Cohesive Zone Model (CZM) & Tiebreak Springs

VibeDynaLITE includes a reactivated and physically robust **Cohesive Zone Model (CZM)** for simulating progressive crack propagation, tearing, and petaling in thin shell structures. Coincident duplicate nodes are generated along element boundaries and bonded by zero-length cohesive springs governing a decoupled mixed-mode bilinear Traction-Separation Law (TSL):

1. **Initial Cohesive Stiffness**: $k_0 = \frac{\sigma_{\text{cohesive}}^2 dx \cdot h}{0.04 G_c}$
2. **Local Coordinate Projection**: The node separation vector $\mathbf{s} = \mathbf{x}_{n1} - \mathbf{x}_{n0}$ is projected onto the normal vector $\mathbf{n}_c$ connecting adjacent element centers:
   $$\delta_n = \mathbf{s} \cdot \mathbf{n}_c$$
   $$\mathbf{s}_t = \mathbf{s} - \delta_n \mathbf{n}_c, \quad \delta_t = \|\mathbf{s}_t\|$$
3. **Decoupled Mixed-Mode Damage Driver**: The damage variable $d \in [0, 1]$ accumulates based on a positive equivalent displacement driver:
   $$\delta_{mix} = \sqrt{\langle\delta_n\rangle^2 + \delta_t^2}$$
   where $\langle\delta_n\rangle = \max(0, \delta_n)$. When $\delta_{mix} > \delta_0$, damage softens the interface:
   $$d = \frac{\delta_c (\delta_{mix} - \delta_0)}{\delta_{mix} (\delta_c - \delta_0)}$$
4. **Thermodynamic Contact Constraints**: Under compression ($\delta_n < 0$), the normal contact stiffness remains fully undamaged ($k_{normal} = k_0$), while shear stiffness softens with damage:
   $$f_n = k_0 \delta_n \quad (\text{if } \delta_n < 0)$$
   $$f_n = (1 - d) k_0 \delta_n \quad (\text{if } \delta_n \ge 0)$$
   $$f_t = (1 - d) k_0 \delta_t$$
5. **Rayleigh Cohesive Damping**: To suppress dynamic stress wave oscillations, stiffness-proportional damping is added to the cohesive force, scaled by the remaining damage factor:
   $$\mathbf{f}_{\text{damp}} = \beta k_0 (1 - d) \mathbf{v}_{\text{rel}}$$
6. **Dynamic Contact Kinematics Coupling**: When a projectile strikes duplicate nodes, a local Breadth-First Search (BFS) is run along active cohesive springs to construct connected tied node clusters. The penalty contact force is evaluated on the cluster-average kinematics and distributed back to the active nodes based on active-element weight metrics. This eliminates artificial unzipping and spurious shear tearing.

---

## 3D SDF Contact & Friction

To model collisions between the projectile and the metallic sheet robustly:
1. **3D Signed Distance Fields (SDF)**: The projectile shapes (sphere, cylinder, bullet, propeller, and box) are represented by mathematically exact 3D SDFs. Contact occurs when the signed distance is negative (penetration), and penalty contact forces are applied.
2. **Coulomb Contact Friction**: When contact is detected, tangential relative velocity $\mathbf{v}_{\text{tang}}$ is computed relative to the contact normal $\mathbf{n}$:
   $$\mathbf{v}_{\text{tang}} = \mathbf{v}_{\text{rel}} - (\mathbf{v}_{\text{rel}} \cdot \mathbf{n}) \mathbf{n}$$
   The friction force is applied in the direction opposing slip:
   $$\mathbf{F}_{\text{friction}} = -\mu_s F_{\text{normal}} \frac{\mathbf{v}_{\text{tang}}}{\|\mathbf{v}_{\text{tang}}\|} \quad (\text{for } \|\mathbf{v}_{\text{tang}}\| > 0)$$
   This prevents the projectile from sliding tangentially through the sheet without resistance, capturing sliding contact physics correctly.

---

## Dynamic GUI Options

In the GUI, selecting `"Metallic Sheet"` under **Structure Type** adapts the inputs:
- Hides fabric-specific controls (crimp, yarn counts, etc.) and grid settings (n_plies, stack modes).
- Exposes metal properties (**Poisson's Ratio**, **Yield Strength**, **Hardening Modulus**, **Ultimate Strain Limit**, and sheet **Thickness**).
- Exposes CZM interface properties: **Use CZM (enable/disable)**, **Cohesive Strength (GPa)**, and **Fracture Energy ($J/m^2$)**.
- Includes the `"Corten Steel (14 Gauge)"` preset containing researched properties for container panels.
