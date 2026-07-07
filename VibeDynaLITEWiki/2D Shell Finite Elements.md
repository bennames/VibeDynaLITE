# 2D Shell Finite Elements

To support modeling metallic sheets, container walls, and other continuous thin-walled structures, VibeDynaLITE includes a **2D Explicit Finite Element (FE) Shell Solver** alongside the traditional mass-spring lattice formulation. 

---

## Element Formulation

The solver utilizes a **Q4 Reissner-Mindlin Bilinear Quadrilateral Shell Element** to capture both membrane and bending behavior:

1. **Degrees of Freedom**: Each node has 6 degrees of freedom:
   - 3 translational: $u_x, u_y, u_z$
   - 3 rotational: $\theta_x, \theta_y, \theta_z$
2. **Kinematics**: Transverse shear deformation is included based on the Reissner-Mindlin theory, allowing the element to remain valid for both thin and moderately thick plates.
3. **Integration**: Numerical integration is performed in-plane using a single Gauss point at the center of the element, combined with **Flanagan-Belytschko hourglass control** (dimensionally scaled using physical wave-impedance properties $\sqrt{E \rho} h dx$ and $\sqrt{G \rho} h dx^3$ to prevent numerical explosions) to suppress spurious zero-energy modes.
4. **Through-Thickness Integration**: Integration through the thickness is performed using **Simpson's rule** with 3 integration points (top, middle, and bottom) to capture bending and nonlinear material response accurately.

---

## J2 Radial Return Plasticity

For metallic materials like Corten Steel, a **J2 radial return plasticity model** is implemented at each through-thickness integration point:

1. **Stress-Strain Update**:
   - Strains and curvatures are evaluated from nodal displacement rates.
   - An elastic trial stress is computed at each integration point:
     \[ \boldsymbol{\sigma}_{\text{trial}} = \boldsymbol{\sigma}_{n} + \mathbf{C} : \Delta\boldsymbol{\epsilon} \]
2. **Yield Criterion**:
   - The von Mises equivalent stress $\sigma_{\text{eq}}$ is compared to the yield strength $\sigma_y$:
     \[ f = \sigma_{\text{eq}} - (\sigma_y + H e_{\text{pl}}) \]
     where $H$ is the isotropic hardening modulus, and $e_{\text{pl}}$ is the equivalent plastic strain.
3. **Radial Return Mapping**:
   - If $f > 0$, plastic flow occurs. Stresses are radially scaled back to the yield surface:
     \[ \Delta e_{\text{pl}} = \frac{f}{3G + H} \]
     \[ \boldsymbol{\sigma}_{n+1} = \left(1 - \frac{3G \Delta e_{\text{pl}}}{\sigma_{\text{eq}}}\right) \boldsymbol{\sigma}_{\text{trial}} \]
     where $G$ is the shear modulus.

---

## Element Erosion & Deletion

Rupture/breaking of the metallic sheet is modeled via **equivalent plastic strain-based element erosion**:
* At each timestep, the equivalent plastic strain $e_{\text{pl}}$ is monitored at all 3 thickness integration points of each shell element.
* When $e_{\text{pl}}$ exceeds the **Ultimate Strain Limit** ($\epsilon_{\text{ult}}$) at all 3 thickness integration points, the element is **eroded/deleted**.
* Once deleted:
  - The element's contribution to internal forces and moments drops to zero.
  - The element is excluded from contact mechanics (no projectile collision forces will act on nodes via this element).
  - Visual rendering updates to show the physical puncture hole.

---

## Dynamic GUI Options

In the GUI, selecting `"Metallic Sheet"` under **Structure Type** adapts the inputs:
- Hides fabric-specific controls (crimp, yarn counts, etc.) and grid settings (n_plies, stack modes).
- Exposes metal properties (**Poisson's Ratio**, **Yield Strength**, **Hardening Modulus**, **Ultimate Strain Limit**, and sheet **Thickness**).
- Includes the `"Corten Steel (14 Gauge)"` preset containing researched properties for container panels.
