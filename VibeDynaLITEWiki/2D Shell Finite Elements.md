# 2D Shell Finite Elements

To support modeling metallic sheets, container walls, and other continuous thin-walled structures, VibeDynaLITE includes a **2D Explicit Finite Element (FE) Shell Solver** alongside the traditional mass-spring lattice formulation. 

---

## Element Formulation

The solver utilizes a **Co-Rotational Belytschko-Tsay Shell Element** to capture large-deformation membrane, bending, and transverse shear behavior objectively and frame-invariantly:

1. **Degrees of Freedom**: Each node has 6 degrees of freedom:
   - 3 translational: $u_x, u_y, u_z$
   - 3 rotational: $\theta_x, \theta_y, \theta_z$
2. **Local Coordinate Triad Construction**: At each timestep, an orthonormal local frame $(\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3)$ is constructed at the element center:
   - Diagonals are defined as $\mathbf{s}_1 = \mathbf{x}_2 - \mathbf{x}_0$ and $\mathbf{s}_2 = \mathbf{x}_3 - \mathbf{x}_1$.
   - The normal vector is $\mathbf{e}_3 = \text{normalize}(\mathbf{s}_1 \times \mathbf{s}_2)$.
   - The local x-axis is $\mathbf{e}_1 = \text{normalize}(\mathbf{s}_1)$.
   - The local y-axis is $\mathbf{e}_2 = \mathbf{e}_3 \times \mathbf{e}_1$.
3. **Local Kinematics Projection**: Global velocities $\mathbf{v}_i$ and angular velocities $\boldsymbol{\omega}_i$ are projected onto the local frame to yield local velocities $(v'_{x,i}, v'_{y,i}, v'_{z,i})$ and spin rates $(\omega'_{x,i}, \omega'_{y,i}, \omega'_{z,i})$.
4. **Local Strain and Curvature Rates**: Local velocity strain rates and bending curvature rates are evaluated at the element center:
   - Velocity strain rates:
     $$\dot{\epsilon}'_{xx} = \sum v'_{x,i} N_{i,x'}, \quad \dot{\epsilon}'_{yy} = \sum v'_{y,i} N_{i,y'}, \quad \dot{\gamma}'_{xy} = \sum (v'_{x,i} N_{i,y'} + v'_{y,i} N_{i,x'})$$
   - Bending curvature rates:
     $$\dot{\kappa}'_{xx} = \sum \omega'_{y,i} N_{i,x'}, \quad \dot{\kappa}'_{yy} = -\sum \omega'_{x,i} N_{i,y'}, \quad \dot{\kappa}'_{xy} = \sum (\omega'_{y,i} N_{i,y'} - \omega'_{x,i} N_{i,x'})$$
   - Transverse shear strain rates (Reissner-Mindlin):
     $$\dot{\gamma}'_{xz} = \sum v'_{z,i} N_{i,x'} + \bar{\omega}'_y, \quad \dot{\gamma}'_{yz} = \sum v'_{z,i} N_{i,y'} - \bar{\omega}'_x$$
     where $\bar{\omega}'_x, \bar{\omega}'_y$ are the average nodal rotational rates, and $N_{i,x'}, N_{i,y'}$ are the local bilinear shape function derivatives.
5. **Rate-Integration**: Strain increments are integrated incrementally:
   $$\Delta \epsilon'_{ij} = \dot{\epsilon}'_{ij} \Delta t, \quad \Delta \kappa'_{ij} = \dot{\kappa}'_{ij} \Delta t, \quad \Delta \gamma'_{iz} = \dot{\gamma}'_{iz} \Delta t$$
6. **Stress Resultants**: J2 plastic stress integration is performed through the thickness, and local membrane forces $\mathbf{N}'$, bending moments $\mathbf{M}'$, and shear forces $\mathbf{Q}'$ are computed by integration.
7. **Local Force Assembly**: Local nodal forces and moments are assembled from stress resultants (resisting forces are negated to act against deformation):
   $$f'_{x,i} = -A (N'_{xx} N_{i,x'} + N'_{xy} N_{i,y'})$$
   $$f'_{y,i} = -A (N'_{yy} N_{i,y'} + N'_{xy} N_{i,x'})$$
   $$f'_{z,i} = -A (Q'_x N_{i,x'} + Q'_y N_{i,y'})$$
   $$m'_{x,i} = -A (-M'_{yy} N_{i,y'} - M'_{xy} N_{i,x'}) + 0.25 A Q'_y$$
   $$m'_{y,i} = -A (M'_{xx} N_{i,x'} + M'_{xy} N_{i,y'}) - 0.25 A Q'_x$$
8. **Flanagan-Belytschko Hourglass Control**: Visco-plastic resisting forces and moments are added in the local frame to suppress spurious zero-energy modes using coefficient $\epsilon_{hg} = 0.05$:
   - Translational: $C_{\text{damp}} = 0.05 \sqrt{E \rho} \cdot h \cdot dx$
   - Rotational: $C_{\text{rot\_damp}} = 0.05 \sqrt{E \rho} \cdot h^3 \cdot dx$
   - Transverse Shear: $C_{\text{shear\_damp}} = 0.05 \sqrt{G \rho} \cdot h \cdot dx^3$
9. **Global Rotation**: Local nodal forces $\mathbf{f}'_i$ and moments $\mathbf{m}'_i$ (including hourglass contributions) are rotated back to the global frame:
   $$\mathbf{f}_i = \mathbf{R}^T \cdot \mathbf{f}'_i, \quad \mathbf{m}_i = \mathbf{R}^T \cdot \mathbf{m}'_i$$
   where $\mathbf{R} = [\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3]^T$ is the local-to-global rotation matrix.
10. **Through-Thickness Integration**: Integration through the thickness is performed using **Simpson's rule** with 5 integration points ($z_k \in \{-0.5h, -0.25h, 0, 0.25h, 0.5h\}$ with weights $w_k \in \{h/12, 4h/12, 2h/12, 4h/12, h/12\}$) to capture bending and nonlinear material response accurately.

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
