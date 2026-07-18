# Benchmark 8: Ballistic Limit (V50) Validation against Physical Data (Dry Kevlar 29)

## 1. Objective and Rationale
This benchmark provides a strict, physically grounded standard to validate the explicit solver's ability to accurately predict the $V_{50}$ ballistic limit of **dry, uncoated Kevlar 29 fabric**.

Because explicit dynamics solvers are highly sensitive to contact penalty stiffness, friction coefficients, and mesh resolution, this document establishes absolute real-world experimental data as the ground truth. An AI agent or CI/CD pipeline should use this file to autonomously run bounding test cases, measure residual velocities, and tune solver parameters to match physical reality without introducing "artificial" stiffness or phantom energy.

---

## 2. Experimental Ground Truth (Reference Standards)

The physical parameters below are derived directly from declassified U.S. Army Research Laboratory (ARL) impact data and MIL-STD-662F testing for the PASGT (Personnel Armor System for Ground Troops) soft armor vest. The PASGT vest was constructed entirely from dry, matrix-free Kevlar 29 plies.

### 2.1. Material Specifications: Kevlar® 29 (Style 713)
*   **Material**: Poly-paraphenylene terephthalamide (Kevlar 29)
*   **Condition**: Dry fabric (friction-only inter-ply and inter-yarn interactions, no resin)
*   **Yarn Denier**: 1000 Denier
*   **Fabric Areal Density (per ply)**: $475 \text{ g/m}^2$ ($14.0 \text{ oz/yd}^2$)
*   **Material Volumetric Density ($\rho$)**: $1,440 \text{ kg/m}^3$ ($1.44 \text{ g/cm}^3$)
*   **Longitudinal Elastic Modulus ($E$)**: $70.5 \text{ GPa}$
*   **Tensile Strength ($\sigma_{fail}$)**: $2.9 \text{ GPa}$
*   **Strain-to-Failure ($\epsilon_{fail}$)**: $3.6\% - 4.0\%$
*   **Wave Speed ($c$)**: $\approx 7,000 \text{ m/s}$
*   **Dry Static Friction Coefficient ($\mu_s$)**: $0.18 - 0.22$ (Aramid-on-Aramid)

### 2.2. Projectile Threat (MIL-P-46593A)
To isolate the fabric's failure mechanics from the complex plastic deformation of a soft lead bullet, the standard physical test uses a rigid steel penetrator.
*   **Projectile**: 17-Grain Fragment Simulating Projectile (FSP)
*   **Mass**: $1.10 \text{ grams}$ (17 grains)
*   **Geometry**: Right Circular Cylinder (RCC) with a flat/chiseled striking face.
*   **Diameter**: $5.46 \text{ mm}$ (0.22 caliber)
*   **Material**: 4340 Hardened Steel (Treat as a non-deformable rigid body in the simulation).

### 2.3. Physical Benchmark Target
*   **Target Configuration**: 13 physical plies of Style 713 Kevlar 29.
*   **Total Target Areal Density**: $\approx 6.17 \text{ kg/m}^2$
*   **Experimental $V_{50}$ Limit**: **$503 \text{ m/s}$** ($1650 \text{ ft/s}$). 
    *(At this velocity, the projectile has exactly a 50% chance of completely passing through the 13 plies).*

---

## 3. Simulation Boundary Setup

To guarantee that variations in behavior are due to solver mechanics rather than differing boundary setups, the digital twin must enforce the following geometry:
1.  **Target Dimensions**: A square fabric patch measuring **$250 \text{ mm} \times 250 \text{ mm}$** (approx. $10 \times 10$ inches).
2.  **Boundary Conditions**: All 4 edges must be fully **clamped** (fixed translations and rotations: $U_x = U_y = U_z = 0$) to represent a standard rigid test frame.
3.  **Mesh Resolution**: The grid/mesh size directly under the projectile impact zone must be fine enough to capture localized shear and strain. A minimum of **3 to 4 nodes** must span the $5.46 \text{ mm}$ diameter of the projectile face. (Max element size $\approx 1.3 \text{ mm}$).
4.  **Ply Interaction**: Plies must NOT share nodes. They must be modeled as separate sliding layers separated by a minor physical gap (e.g., $0.1 \text{ mm}$) to allow for transverse wave transmission and inter-ply contact penalties.
5.  **Impact Location**: Exact geometric center of the fabric. Normal impact ($0^\circ$ obliquity).

---

## 4. Execution Protocol: Pass/Fail Bounds

Explicit dynamic solvers are prone to chaotic variance. Instead of attempting to hit the exact $503 \text{ m/s}$ $V_{50}$ on a single run, an AI agent or CI/CD script should run the following three bounding scenarios to verify the physics engine is bounded and correct.

### Test Case A: Sub-Limit Proofing (Arrest)
*   **Impact Velocity ($V_i$)**: $450 \text{ m/s}$
*   **Verification Metric**: The projectile MUST be fully arrested ($V_{residual} = 0 \text{ m/s}$).
*   **Phenomenology Check**: The front 4-6 plies should fail via localized shear (punching) due to the FSP's flat edge. The rear plies should exhibit severe backface deformation and strain out, but do not rupture.

### Test Case B: The $V_{50}$ Critical Threshold
*   **Impact Velocity ($V_i$)**: $503 \text{ m/s}$
*   **Verification Metric**: The solver should output a mixed/threshold result—either barely caught in the final ply, or perforating with a negligible residual velocity ($V_{residual} < 25 \text{ m/s}$).

### Test Case C: Super-Limit Proofing (Perforation)
*   **Impact Velocity ($V_i$)**: $550 \text{ m/s}$
*   **Verification Metric**: The projectile MUST cleanly perforate all 13 plies. 
*   **Phenomenology Check**: Based on ballistic energy conservation ($V_r = \sqrt{V_i^2 - V_{50}^2}$), the residual velocity after exiting the target should be roughly **$220 \text{ m/s}$** ($\pm 20 \text{ m/s}$).

---

## 5. Agent Troubleshooting Guide (Heuristics)

If the solver outputs wildly different behaviors in random trials, or misses the $V_{50}$ targets above by $>10\%$, the AI agent should use this heuristic guide to tune the solver:

1. **Projectile Penetrates at 450 m/s (Fabric is artificially weak):**
   * **Check Inter-ply Friction**: Dry Kevlar dissipates up to 20% of impact energy via friction. If $\mu = 0$, penetration happens too early as the plies cannot transfer transverse loads. Ensure $\mu \ge 0.18$.
   * **Check CFL Stability**: Kevlar has an exceptionally high wave speed ($c \approx 7,000 \text{ m/s}$). If the timestep ($\Delta t$) is too large, the stress wave jumps elements, causing nodes to falsely exceed the $3.6\%$ strain limit in a single tick. Force the CFL safety factor to `0.1` and re-run.

2. **Projectile is Arrested at 550 m/s (Fabric is artificially strong):**
   * **Check Failure Strain Criteria**: Ensure elements are actively deleted or their stress tensor zeroed out when longitudinal strain exceeds $3.6\%$. If they continue to carry stress past failure, the fabric will act like an invincible net.
   * **Check Mass Scaling**: If artificial mass was added to speed up the explicit solver, it increases the inertial resistance of the fabric. The fabric will act like a solid steel plate, prematurely decelerating the projectile.

3. **Wildly Inconsistent Residual Velocities (Chaos in the Contact Algorithm):**
   * **Check Contact Penalty Stiffness**: If the solver uses a penalty-force method to keep plies from passing through one another, a stiffness that is too high acts like an explosive spring. When plies compress, the penalty force shoots them apart, injecting artificial phantom energy into the system. 
   * **Action**: Check the Energy Conservation logs. Total system energy (kinetic + internal strain + friction) must not drift by more than **$2.0\%$**. If phantom energy is generated upon projectile strike, significantly lower the penalty stiffness constant.