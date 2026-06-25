# Benchmark 11: Mesh Refinement V50 Convergence

## 1. Physics Objective & Theory

This benchmark validates that Bazant-style strain limit regularization ($\epsilon_{\text{fail}} = \epsilon_0 \sqrt{h_0 / dx}$) resolves mesh dependency under grid refinement, ensuring converged physical results and preventing unphysical localized tearing on fine meshes.

In a discrete mass-spring grid representing a woven barrier:
* Point contact forces create localized strain concentrations. As the grid is refined ($dx \to 0$), the local strain at the point of contact increases without bound.
* Without regularization, this local strain singularity triggers premature failure of spring elements, causing the fabric to tear at unphysically low velocities ("zipper" tearing), leading to mesh dependency.
* To restore grid independence, Bazant's crack band theory scales the failure strain threshold:
  \[
  \epsilon_{\text{fail}}(dx) = \epsilon_0 \sqrt{\frac{h_0}{dx}}
  \]
  where $h_0 = 10\text{ mm}$ is the baseline yarn width, and $\epsilon_0 = 3.6\%$ is the reference physical failure strain.
* Under this regularization, the energy dissipated by spring failure remains constant across grid resolutions, yielding converged ballistic limit results.

---

## 2. Code Implementation & Test Design

The benchmark is implemented in the `test_mesh_refinement_v50_convergence` function in [test_physics_6dof.py](file:///Users/bennames/Developer/VibeDynaLITE/tests/integration/test_physics_6dof.py#L143).

### Test Setup
1. A projectile strikes a representative panel of constant physical size ($10\text{ cm} \times 10\text{ cm}$) at three different grid resolutions:
   * **Resolution 1:** $dx_1 = 10\text{ mm}$ ($11 \times 11$ nodes)
   * **Resolution 2:** $dx_2 = 5\text{ mm}$ ($21 \times 21$ nodes)
   * **Resolution 3:** $dx_3 = 2.5\text{ mm}$ ($41 \times 41$ nodes)
2. The projectile has a mass of $m = 0.01\text{ kg}$ and strikes the panel at a velocity of $200\text{ m/s}$.
3. Timestep $dt$ is scaled proportionally to $dx$ to respect the CFL safety limit.
4. The test runs each simulation for a constant physical duration of 20 microseconds.
5. The test asserts that the final residual velocities at all three resolutions remain stable within a tight $15\%$ window of the mean.

---

## 3. Verification & Validation Results

* **Residual Velocity Convergence:**
  * **Expected:** Residual velocities at $dx = 10\text{ mm}$, $5\text{ mm}$, and $2.5\text{ mm}$ remain within $15\%$ of the mean.
  * **Observed:** Under Bazant regularization, the residual velocities converged closely (PASSED).

---

## 4. Current Status

* **Status:** **PASSED & VERIFIED**
* **Active Suite Integration:** Integrated as `test_mesh_refinement_v50_convergence` in the standard test runner.
