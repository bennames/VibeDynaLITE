# 2D Shell Finite Elements

To support modeling metallic sheets, container walls, and other continuous thin-walled structures, VibeDynaLITE includes a **2D Explicit Finite Element (FE) Shell Solver** alongside the traditional mass-spring lattice formulation. 

---

## Element Formulation

The solver utilizes a **Q4 Reissner-Mindlin Bilinear Quadrilateral Shell Element** to capture both membrane and bending behavior:

1. **Degrees of Freedom**: Each node has 6 degrees of freedom:
   - 3 translational: $u_x, u_y, u_z$
   - 3 rotational: $\theta_x, \theta_y, \theta_z$
2. **Kinematics**: Transverse shear deformation is included based on Reissner-Mindlin theory, allowing the element to remain valid for both thin and moderately thick plates.
3. **Geometric Non-Linearity (Von Karman strains)**: To model transverse-membrane coupling under large deflections (necessary for lateral tension wave propagation), the strain formulation incorporates non-linear out-of-plane displacement gradients:
   $$\epsilon_{xx} = \frac{\partial u}{\partial x} + \frac{1}{2}\left(\frac{\partial w}{\partial x}\right)^2, \quad \epsilon_{yy} = \frac{\partial v}{\partial y} + \frac{1}{2}\left(\frac{\partial w}{\partial y}\right)^2, \quad \gamma_{xy} = \frac{\partial u}{\partial y} + \frac{\partial v}{\partial x} + \frac{\partial w}{\partial x}\frac{\partial w}{\partial y}$$
   This ensures that vertical deflection $w$ induces membrane tension, distributing transverse loads laterally and initiating dynamic wave propagation.
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

## J2 Radial Return Plasticity

For metallic materials like Corten Steel, a **J2 radial return plasticity model** is implemented at each through-thickness integration point:

1. **Stress-Strain Update**: Strains and curvatures are evaluated from nodal displacement rates. An elastic trial stress is computed under plane-stress assumptions:
   $$\sigma_{xx}^{\text{trial}} = \sigma_{xx}^n + \frac{E}{1-\nu^2} (\Delta \epsilon_{xx} + \nu \Delta \epsilon_{yy})$$
   $$\sigma_{yy}^{\text{trial}} = \sigma_{yy}^n + \frac{E}{1-\nu^2} (\Delta \epsilon_{yy} + \nu \Delta \epsilon_{xx})$$
   $$\tau_{xy}^{\text{trial}} = \tau_{xy}^n + G \Delta \gamma_{xy}$$
2. **Yield Criterion**: The von Mises equivalent trial stress $\sigma_{\text{vm}}^{\text{trial}}$ is evaluated:
   $$\sigma_{\text{vm}}^{\text{trial}} = \sqrt{(\sigma_{xx}^{\text{trial}})^2 + (\sigma_{yy}^{\text{trial}})^2 - \sigma_{xx}^{\text{trial}} \sigma_{yy}^{\text{trial}} + 3 (\tau_{xy}^{\text{trial}})^2}$$
   and compared to the isotropic-hardened yield strength:
   $$f = \sigma_{\text{vm}}^{\text{trial}} - (\sigma_{y,0} + H e_{\text{peeq}}^n)$$
   where $H$ is the isotropic hardening modulus, and $e_{\text{peeq}}$ is the equivalent plastic strain.
3. **Radial Return Mapping**: If $f > 0$, plastic flow occurs. Stresses are radially scaled back to the yield surface:
   $$\Delta e_{\text{peeq}} = \frac{f}{3G + H}$$
   $$\sigma^{n+1} = \left(1 - \frac{3G \Delta e_{\text{peeq}}}{\sigma_{\text{vm}}^{\text{trial}}}\right) \sigma^{\text{trial}}$$
   $$e_{\text{peeq}}^{n+1} = e_{\text{peeq}}^n + \Delta e_{\text{peeq}}$$
   where $G = \frac{E}{2(1+\nu)}$ is the shear modulus.
4. **Plastic Dissipation**: The energy dissipated by plastic work during the step is accumulated:
   $$\Delta W_{\text{plastic}} = \left(\sigma_{y,0} + H e_{\text{peeq}}^n\right) \Delta e_{\text{peeq}} \cdot w_k \cdot dx^2$$

---

## Continuous Damage Mechanics (CDM) & Stress Triaxiality

To model progressive failure and ductile tearing, VibeDynaLITE integrates a scalar continuous damage variable $D \in [0, 1]$ at each through-thickness integration point:

1. **Damage Accumulation**: Damage evolves based on the increment of equivalent plastic strain scaled by a triaxiality-dependent failure strain $\epsilon_f$:
   $$\Delta D = \frac{\Delta e_{\text{peeq}}}{\epsilon_f}$$
   $$D_{n+1} = \min(1.0, D_n + \Delta D)$$
2. **Stress Triaxiality ($\eta$)**: Computed as the ratio of hydrostatic (mean) stress to von Mises equivalent stress:
   $$\eta = \frac{\sigma_m}{\sigma_{\text{vm}}} = \frac{\sigma_{xx} + \sigma_{yy}}{3 \sigma_{\text{vm}}}$$
3. **Triaxiality-Dependent Failure Strain ($\epsilon_f$)**: The ultimate failure strain scales according to the local stress state to capture ductile vs. shear damage mechanisms:
   - **Tension ($\eta > 0$)**: Failure strain decays exponentially under multi-axial tension (brittle tearing behavior):
     $$\epsilon_f = \epsilon_{\text{ult}} \exp\left(-1.5\left(\eta - \frac{1}{3}\right)\right)$$
   - **Compression & Shear ($\eta \le 0$)**: Failure strain increases to capture higher ductility:
     $$\epsilon_f = \epsilon_{\text{ult}} \exp(-0.5\eta)$$
   - **Regularization Limit**: A lower bound is enforced to prevent premature or instant failure: $\epsilon_f \ge 0.005$.
4. **Stiffness Degradation**: The nominal stress (both elastic-plastic and viscous Rayleigh damping stresses) is degraded by the active damage parameter:
   $$\sigma_{\text{total}} = (\sigma_{\text{nominal}} + \sigma_{\text{damp}}) (1 - D)$$
5. **Element Erosion / Deletion**: An element is eroded and deleted from the simulation ONLY when the damage parameter reaches $D \ge 1.0$ at all 3 thickness integration points. Once deleted, its contribution to internal forces drops to zero, and it is excluded from contact calculations.

---

## Cohesive Zone Model (CZM) & Tiebreak Springs

To simulate structural tearing, petaling, and separation along element boundaries without relying solely on element erosion (which deletes mass and can create artificial holes), VibeDynaLITE implements an interface **Cohesive Zone Model (CZM)**:

```
Normal Shared Mesh:
Node 0 ------------ Node 1
  |   e1 (shared)     |
  |                   |

CZM Duplicated Mesh:
Node e1_0 --------- Node e1_1   <-- Element 1
=============================   <-- Inter-element Boundary (Tiebreak Springs)
Node e2_3 --------- Node e2_2   <-- Element 2
```

1. **Mesh Duplication**: In CZM mode, the finite element mesh is duplicated such that elements do not share nodes. Each bilinear quadrilateral element $e$ has 4 unique node IDs, resulting in $4 \times N_{el}$ total nodes. Corner coordinates are initially coincident.
2. **Tiebreak Springs**: Adjacent element edges are bonded together by zero-length tiebreak springs at the corners. These springs govern the interface separation behavior under a bilinear **Traction-Separation Law (TSL)**:
   - **Tributary Area**: $A_{\text{trib}} = 0.5 dx \cdot h$ (where $h$ is the sheet thickness)
   - **Peak Force Capacity**: $F_{\text{max}} = \sigma_{\text{cohesive}} A_{\text{trib}}$
   - **Critical Separation (Complete Failure)**: $\delta_c = \frac{2 G_c}{\sigma_{\text{cohesive}}}$
   - **Elastic Limit Separation (Damage Initiation)**: $\delta_0 = 0.01 \delta_c$
   - **Initial Cohesive Stiffness**: $k_0 = \frac{F_{\text{max}}}{\delta_0} = \frac{\sigma_{\text{cohesive}}^2 dx \cdot h}{0.04 G_c}$
3. **Damage & Softening**: 
   - The separation distance is measured between coincident nodes: $\delta = \|\mathbf{x}_{n1} - \mathbf{x}_{n0}\|$.
   - For separations exceeding the elastic limit $\delta > \delta_0$, a scalar damage variable $d \in [0, 1]$ accumulates irreversibly:
     $$d = \min\left(1.0, \max\left(d_n, \frac{\delta_c (\delta - \delta_0)}{\delta (\delta_c - \delta_0)}\right)\right)$$
   - The spring force is softened continuously:
     $$F_{\text{cohesive}} = (1 - d) k_0 \delta$$
   - Once $d \ge 1.0$, the tiebreak spring is broken permanently (`spring_failed = 1`), allowing elements to physically separate, tear, and petal.
4. **Energy Tracking**: Cohesive fracture work is accumulated dynamically:
   $$\Delta W_{\text{cohesive}} = \mathbf{F}_{\text{cohesive}} \cdot \Delta \mathbf{v}_{\text{half}} \cdot dt$$

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
