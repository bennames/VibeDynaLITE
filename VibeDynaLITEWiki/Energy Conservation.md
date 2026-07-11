# Energy Conservation

In an explicit dynamics simulation, total energy should be approximately conserved at every timestep. Energy isn't created or destroyed — it moves between kinetic, potential, and dissipated forms. If total energy drifts upward, something is non-physical and the simulation results are unreliable. Monitoring energy balance is the single most important diagnostic for validating a VibeDynaLITE run.

---

## The Energy Balance Equation

$$KE_{structure} + SE_{elastic} + E_{contact} + KE_{projectile} + E_{damped} + E_{fracture} + E_{clamped} + E_{friction} \approx KE_{initial}$$

The left side should remain approximately equal to the initial projectile kinetic energy throughout the simulation. A small downward drift (energy loss) is acceptable — it means damping is doing its job. Any **upward** drift signals a bug.

---

## Energy Terms

### $KE_{fabric}$ — Fabric Kinetic Energy

The kinetic energy of all fabric nodes:

$$KE_{fabric} = \sum_i \frac{1}{2} m_i \|v_i\|^2$$

Computed by `compute_kinetic_energy()` in `energy.py`. This is typically near zero at time $t = 0$ (fabric starts at rest) and grows rapidly as the transverse wave propagates, then decreases as the projectile decelerates.

### $SE_{elastic}$ — Elastic Strain Energy

Computed by `compute_strain_energy()` in `energy.py`:
- **Fabric Mode**: The potential energy stored in all active (non-failed) springs:
  $$SE = \sum_j \frac{1}{2} k_j (\varepsilon_j \cdot L_{0,j})^2$$
  Only tensile strains are counted (compressive strain energy is zeroed via `maximum(0.0, strains)`). Failed springs contribute zero strain energy.
- **Shell Element Mode**: The strain energy of the shell elements computed from active element stresses integrated over the element volume using Simpson's 5-Point Rule:
  $$SE = \sum_e \frac{1}{2} dx^2 \sum_{k=1}^5 w_k \frac{\sigma_{xx, k}^2 + \sigma_{yy, k}^2 - 2\nu\sigma_{xx, k}\sigma_{yy, k} + 2(1+\nu)\tau_{xy, k}^2}{E} (1 - d_k)$$
  where $dx$ is element size, $k$ represents the 5 Simpson thickness integration points, $w_k \in \{h/12, 4h/12, 2h/12, 4h/12, h/12\}$ are the thickness integration weights, $\nu$ is Poisson's ratio, and $d_k$ is the local damage at integration point $k$. Eroded elements contribute zero strain energy.

### $E_{contact}$ — Contact Potential Energy

The elastic potential energy stored in penalty contact springs. When contact forces are capped to structural shear capacity $f_{\text{cap}}$, the elastic potential energy stored in contact is:

$$E_{contact} = \sum_i 0.5 \frac{F_{elastic, capped}^2}{k_{penalty}} \cdot \text{scale\_factor}_i$$

where $F_{elastic, capped} = \min(k_{penalty} \cdot \delta_i, f_{\text{cap}})$, and $\delta_i$ is the penetration depth. This capped formulation ensures contact energy calculations remain perfectly consistent with the actual forces applied to the nodes, eliminating artificial energy growth.

### $KE_{projectile}$ — Projectile Kinetic Energy

The rigid-body kinetic energy of the projectile, including both translational and rotational components:

$$KE_{projectile} = \frac{1}{2} m_{proj} \|v_{proj}\|^2 + \frac{1}{2} \boldsymbol{\omega}_{body}^T \boldsymbol{I}_{body} \boldsymbol{\omega}_{body}$$

Computed inline in the fused loop. This starts at $KE_{initial}$ and decreases as the projectile decelerates and transfers energy via impact. For a full perforation event, it levels off at a non-zero residual velocity. For a catch, it goes to zero.

### $E_{damped}$ — Damping Dissipation

The cumulative energy removed by [[Damping Models|damping]] forces:

$$E_{damped} = \int_0^t -F_{damp} \cdot v \, dt' \approx \sum_{n} (-P_{damp}^n) \cdot \Delta t$$

Tracked as a running sum (`damp_dissipated`) in the fused loop. This should grow monotonically — if it ever decreases, the damping model has a sign error.

### $E_{fracture}$ — Fracture & Plastic Dissipation

Energy dissipated during material failure and inelastic flow:
- **Fabric Mode**: When a spring fails, the elastic strain energy it was storing is booked as dissipated fracture energy (multiplied by the configured `fracture_energy_multiplier` to account for sub-grid friction, fibrillation, and yarn pull-out). See [[Spring Failure Mechanics]] for details.
- **J2 Plasticity Mode**: For shells, plastic work is accumulated across all thickness points of active elements:
  $$\Delta W_{\text{plastic}} = \sigma_y \Delta e_{\text{peeq}} \cdot w_k \cdot dx^2$$
  where $w_k$ are the Simpson rule weights.
- **Cohesive Zone Model (CZM) Mode**: The cohesive work dissipated by tearing of tiebreak springs:
  $$\Delta W_{\text{cohesive}} = \mathbf{F}_{\text{cohesive}} \cdot \Delta \mathbf{v}_{\text{half}} \cdot dt$$
All three contributions are summed dynamically into the `failure_dissipated` term in the solver loop.

### $E_{clamped}$ — Velocity Clamp Dissipation

Energy removed by the CFL velocity clamp (see [[CFL Stability Condition]]). When a node exceeds the maximum physical wave speed velocity $v_{max} = dx / dt$, its velocity is scaled down, and the excess kinetic energy is stored in a tracking buffer to preserve the energy balance.

### $E_{friction}$ — Friction Dissipation

The energy dissipated by Coulomb contact sliding between the projectile and the sheet/fabric:
$$E_{friction} = \int_0^t -\mathbf{F}_{friction} \cdot \mathbf{v}_{tang} \, dt'$$
This is tracked as `friction_dissipated` in the solver.

---

The `compute_energy_balance()` function in `energy.py` returns a dictionary including all terms:

```python
{
    "kinetic": ke,
    "strain": se,
    "damped": damped,
    "failure_dissipated": failure_dissipated,
    "clamp_dissipated": clamp_dissipated,
    "friction_dissipated": friction_dissipated,
    "projectile_kinetic": proj_ke,
    "total": ke + se + damped + failure_dissipated + clamp_dissipated + friction_dissipated + proj_ke
}
```

The fused loop aggregates and updates these values in real-time.

---

## Lagrangian Inertial Drift for Ejected Nodes (Sprint 7.8)

When elements delete and nodes become completely detached from the mesh ($N_{active} = 0$), they can cause non-physical energy spikes if they continue to experience contact forces (such as inter-ply contact or projectile contact) without any structural resistance. 

To resolve this while maintaining momentum conservation, VibeDynaLITE implements **Lagrangian Inertial Drift**:
1. **Identify Inactive Nodes**: In each integration step, the solver dynamically gathers the number of active springs connected to each node.
2. **Exclude from Contact**: For any node where $N_{active} == 0$, all projectile and inter-ply contact forces are scaled to exactly $0.0$.
3. **Pure Inertial Motion**: The node moves passively with constant velocity (constant kinetic energy and momentum) and experiences zero external forces, eliminating artificial energy generation and satisfying exact physical conservation post-breakthrough.

---

## What Causes Non-Physical Energy Growth

If you see total energy increasing over time, the cause is almost certainly one of these three mechanisms:

### 1. Ghost Forces

**The problem:** A spring at peak strain applies maximum force for one full timestep *after* failure detection, before its force is zeroed on the next step. During that ghost timestep, the large force accelerates nodes, injecting energy.

**Why it happens:** In the current fused loop, spring forces are computed first (line 215), and failure detection happens later (line 255). A spring that should have failed gets one extra timestep of full-stiffness force.

**The fix:** Detect failures BEFORE computing forces in each timestep. Move the strain check and `grid_failed` update above the `compute_spring_forces()` call.

### 2. Binary Failure Discontinuity

**The problem:** A spring's stiffness drops instantaneously from full $k$ to zero when it fails. This creates a force discontinuity — nodes that were being pulled back by the spring are suddenly released, and the momentum they had resisting the spring force now launches them.

**The fix:** A [[Spring Failure Mechanics|progressive damage model]] that gradually reduces stiffness between a damage onset strain and the final failure strain. By the time the spring fully breaks, its force is near zero.

### 3. Missing Fracture Energy Accounting

**The problem:** When a spring breaks, the elastic strain energy $\frac{1}{2} k (\varepsilon \cdot L_0)^2$ it was storing simply vanishes from the strain energy sum. It doesn't appear in any dissipation term, so the total energy appears to drop. While this is energy *loss* not *growth*, it can mask real energy growth from ghost forces — the two artifacts partially cancel, hiding bugs.

**The fix:** Track fracture energy explicitly. When a spring fails, compute its stored elastic energy at the moment of failure, multiply by the Kevlar fracture energy multiplier (see [[Spring Failure Mechanics]]), and add it to a cumulative `E_fracture` term.

> [!TIP]
> A quick diagnostic: plot the total energy ratio $E_{total}(t) / KE_{initial}$ over time. It should start at 1.0 and stay within ~1–5% for a well-behaved simulation. Values above 1.0 indicate energy injection. Values significantly below 1.0 indicate untracked dissipation (fracture, clamping).

---

## See Also

- [[Damping Models]] — How damping dissipates energy and the different damping approaches
- [[Spring Failure Mechanics]] — Fracture energy, progressive damage, and the ghost force problem
- [[CFL Stability Condition]] — Velocity clamping and its energy implications
