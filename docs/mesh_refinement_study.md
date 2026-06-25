# Technical Note: Mesh Sensitivity and Ballistic Limit Convergence in Mass-Spring Fabric Models

This note analyzes the grid-sensitivity and convergence of the V50 ballistic limit in the Kevlar 29 mass-spring explicit dynamics solver. It documents the results of a mesh refinement study down to $dx = 0.00125 \text{ m}$ ($241 \times 241$ grid) and discusses the underlying mathematical and physical causes of the grid-dependence.

---

## 1. Quantitative Benchmark Results

The simulation was run for a single-ply Kevlar 29 square sheet ($0.30 \text{ m} \times 0.30 \text{ m}$ physical domain) impacted by a 17-grain FSP projectile ($1.1\text{ g}$). We compared two contact configurations:
1. **Global Center-Node Release**: Contact forces on all nodes are zeroed as soon as the center node's connected springs fail.
2. **Local Contact Release (Non-Reflecting)**: Contact forces are scaled locally at each node based on its active spring count, with stress-wave-absorbing boundary conditions.

### Fitted Jonas-Laval (Lambert-Jonas) Parameters:
* Jonas-Laval Fit Equation: $V_r = \alpha \sqrt{V_s^2 - V_{50}^2}$ for $V_s > V_{50}$ (else $0.0$).

| Mesh Size ($dx$) | Grid Resolution | Global Release V50 | Local Release V50 | Convergence Trend |
| :--- | :--- | :--- | :--- | :--- |
| **10.00 mm** | $31 \times 31$ | **304.0 m/s** | **304.0 m/s** | Coarse baseline (stable) |
| **5.00 mm** | $61 \times 61$ | **177.0 m/s** | **172.0 m/s** | Severe drop |
| **2.50 mm** | $121 \times 121$ | **143.0 m/s** | **127.0 m/s** | Continued degradation |
| **1.25 mm** | $241 \times 241$ | *N/A* | **136.0 m/s** | Asymptotic bottoming |

### Observed Low-Velocity Behavior:
At $dx \le 5\text{ mm}$, the projectile penetrates the fabric even at $100\text{ m/s}$ (producing residual velocities of $\sim 50\text{--}60\text{ m/s}$). In a physical system, Kevlar 29 easily arrests a $1.1\text{ g}$ projectile at $100\text{ m/s}$ without damage.

---

## 2. Mathematical Analysis of the Strain Singularity

The lack of mesh convergence is caused by a **strain singularity** under the localized contact penalty force as the grid spacing $dx \to 0$.

### 1. Deflection vs. Strain Kinematics
Consider a node at the center of the impact zone. Let the grid spacing be $dx$. Under transverse impact, the node is deflected vertically by a distance $dz$. The stretched length of a connected spring is:
\[
L = \sqrt{dx^2 + dz^2}
\]
The engineering strain $\epsilon$ in the spring is:
\[
\epsilon = \frac{L - dx}{dx} = \sqrt{1 + \left(\frac{dz}{dx}\right)^2} - 1
\]
For small deflections ($dz \ll dx$), this simplifies via Taylor expansion to:
\[
\epsilon \approx \frac{1}{2} \left(\frac{dz}{dx}\right)^2
\]
Solving for the critical vertical deflection $dz_{\text{fail}}$ that triggers spring rupture ($\epsilon = \epsilon_{\text{fail}}$):
\[
dz_{\text{fail}} \approx dx \sqrt{2 \epsilon_{\text{fail}}}
\]

### 2. Physical Implications of the Singularity
Because $dz_{\text{fail}}$ is directly proportional to $dx$:
* At $dx = 10\text{ mm}$ and $\epsilon_{\text{fail}} = 3.6\%$, the node must deflect **$2.68\text{ mm}$** before failing.
* At $dx = 1.25\text{ mm}$, the node fails at just **$0.34\text{ mm}$** of deflection.
* At $100\text{ m/s}$ strike velocity, the projectile travels $0.34\text{ mm}$ in **$3.4\text{ microseconds}$**.
* Consequently, on refined grids, the center node's springs fail almost instantly. The contact force then shifts to the adjacent nodes. Because they are also very close to the center, they fail under the same deflection mechanism. This leads to a rapid, localized "zipper" tear through the fabric.

---

## 3. Options for Achieving Grid Convergence

To obtain a mesh-independent V50 limit, the local failure criteria must be regularized. We propose two potential strategies:

### Option A: Bazant-Style Crack Band Regularization
In finite element analysis of concrete and other cracking materials, local strain limits are scaled based on the cell size to maintain a constant energy release rate ($G_c$) during element failure. 

In a mass-spring system, the failure strain can be scaled dynamically with the grid spacing:
\[
\epsilon_{\text{fail}}(dx) = \epsilon_0 \sqrt{\frac{h_0}{dx}}
\]
where $h_0$ is a reference grid spacing (e.g. $10\text{ mm}$) and $\epsilon_0$ is the baseline failure strain ($3.6\%$). This ensures that the energy required to rupture a unit width of the fabric remains constant regardless of the grid refinement.

### Option B: Non-local Strain Averaging (Yarn Interaction Area)
Instead of evaluating the rupture criterion on individual local springs, compute a non-local weighted average strain over a physical interaction zone (e.g., a circle of radius $R \approx 10\text{ mm}$, matching the typical yarn/blade contact area):
\[
\bar{\epsilon}_i = \frac{\sum_j w_{ij} \epsilon_j}{\sum_j w_{ij}}
\]
where $w_{ij}$ is a distance-based weight. Rupture is only triggered when the averaged non-local strain exceeds the physical failure limit, preventing localized singularities from driving premature failure.
