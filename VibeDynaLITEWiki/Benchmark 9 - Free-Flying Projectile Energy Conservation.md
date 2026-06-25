# Benchmark 9: Free-Flying Projectile Energy Conservation

## 1. Physics Objective & Theory

This benchmark verifies the conservation of total mechanical energy (both linear kinetic energy and rotational kinetic energy) for a 6-DOF rigid-body projectile flying in a vacuum (no contact or external forces/torques acting on it).

According to classical rigid body mechanics:
* In the absence of external forces, the center-of-mass linear velocity $\mathbf{v}$ remains constant, conserving linear kinetic energy:
  \[
  E_{k,\text{lin}} = \frac{1}{2} m \|\mathbf{v}\|^2
  \]
* In the absence of external torques, the angular momentum is conserved. For a sphere with a diagonal isotropic inertia tensor, the angular velocity $\boldsymbol{\omega}$ also remains constant, conserving rotational kinetic energy:
  \[
  E_{k,\text{rot}} = \frac{1}{2} \boldsymbol{\omega}^T \mathbf{I} \boldsymbol{\omega} = \frac{1}{2} I_{\text{sphere}} \|\boldsymbol{\omega}\|^2
  \]
* The total kinetic energy $E_{\text{tot}} = E_{k,\text{lin}} + E_{k,\text{rot}}$ must remain constant over time.

---

## 2. Code Implementation & Test Design

The benchmark is implemented in the `test_projectile_free_flying_energy_conservation` function in [test_physics_6dof.py](file:///Users/bennames/Developer/VibeDynaLITE/tests/integration/test_physics_6dof.py#L9).

### Test Setup
1. A projectile with a mass of $m = 0.5\text{ kg}$ and a radius of $R = 0.02\text{ m}$ is configured as a sphere, yielding a diagonal inertia tensor $I = \frac{2}{5} m R^2 = 8 \times 10^{-5}\text{ kg m}^2$.
2. The initial linear velocity is set to $\mathbf{v} = [10, 20, 30]\text{ m/s}$, and initial angular velocity is $\boldsymbol{\omega} = [1, 2, 3]\text{ rad/s}$.
3. The projectile is placed far above a dummy fabric grid to ensure zero contact force.
4. The solver runs for $100$ steps with a timestep of $dt = 10^{-4}\text{ s}$ using the CPU JIT loop `fused_leapfrog_loop`.
5. The test asserts that the final linear velocity, angular velocity, and total kinetic energy match the initial conditions within a tight numerical tolerance.

---

## 3. Verification & Validation Results

* **Linear Velocity Conservation:**
  * **Expected:** $\mathbf{v}_{\text{final}} = [10.0, 20.0, 30.0]\text{ m/s}$
  * **Observed:** Fully conserved (PASSED).
* **Angular Velocity Conservation:**
  * **Expected:** $\boldsymbol{\omega}_{\text{final}} = [1.0, 2.0, 3.0]\text{ rad/s}$
  * **Observed:** Fully conserved (PASSED).
* **Total Energy Drift:**
  * **Expected:** Drift $<0.01\%$
  * **Observed:** $0.00\%$ drift (PASSED).

---

## 4. Current Status

* **Status:** **PASSED & VERIFIED**
* **Active Suite Integration:** Integrated as `test_projectile_free_flying_energy_conservation` in the standard test runner.
