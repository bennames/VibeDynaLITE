# Benchmark 10: Oblique Impact Tumbling Dynamics

## 1. Physics Objective & Theory

This benchmark verifies the coupling of contact forces to rotational dynamics for a 6-DOF rigid-body projectile. An oblique strike on a fabric barrier must generate a non-zero contact torque and initiate tumbling (rotational acceleration).

According to classical multi-body impact physics:
* When a projectile strikes a fabric mesh at an oblique angle, the contact forces acting on the fabric nodes produce equal-and-opposite reaction forces on the projectile surface.
* Since these contact points do not align with the projectile's center of mass (CoM), they generate an eccentric torque:
  \[
  \boldsymbol{\tau}_{\text{proj}} = \sum_i \mathbf{r}_i \times (-\mathbf{F}_i)
  \]
  where $\mathbf{r}_i = \mathbf{P}_i - \mathbf{pos}$ is the relative vector from the CoM to node $i$.
* This torque drives angular acceleration according to Euler's equations of motion:
  \[
  \dot{\boldsymbol{\omega}} = \mathbf{I}^{-1} \boldsymbol{\tau}_{\text{proj}}
  \]
* The benchmark verifies that contact forces successfully couple to 3D rotation, initiating tumbling dynamics.

---

## 2. Code Implementation & Test Design

The benchmark is implemented in the `test_oblique_impact_tumbling_dynamics` function in [test_physics_6dof.py](file:///Users/bennames/Developer/VibeDynaLITE/tests/integration/test_physics_6dof.py#L71).

### Test Setup
1. A $5 \times 5$ fabric grid with clamped boundary edges is initialized.
2. A cylinder projectile of mass $m = 0.05\text{ kg}$, radius $R = 1.0\text{ cm}$, and length $L = 3.0\text{ cm}$ is placed just below the grid.
3. The projectile is given an oblique velocity vector $\mathbf{v} = [20, 0, 100]\text{ m/s}$ (upward strike with horizontal velocity) and zero initial angular velocity.
4. The solver runs for $100$ steps with a timestep of $dt = 10^{-6}\text{ s}$ using the CPU JIT loop `fused_leapfrog_loop`.
5. The test asserts that the oblique contact forces produce a torque that initiates non-zero angular velocity ($\|\boldsymbol{\omega}\| > 10^{-3}\text{ rad/s}$).

---

## 3. Verification & Validation Results

* **Tumbling Initiation:**
  * **Expected:** $\|\boldsymbol{\omega}\| > 1e-3\text{ rad/s}$ after impact.
  * **Observed:** Projectile acquired significant angular velocity due to eccentric contact forces (PASSED).

---

## 4. Current Status

* **Status:** **PASSED & VERIFIED**
* **Active Suite Integration:** Integrated as `test_oblique_impact_tumbling_dynamics` in the standard test runner.
