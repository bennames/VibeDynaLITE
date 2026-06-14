# Energy Conservation

In an explicit dynamics simulation, total energy should be approximately conserved at every timestep. Energy isn't created or destroyed — it moves between kinetic, potential, and dissipated forms. If total energy drifts upward, something is non-physical and the simulation results are unreliable. Monitoring energy balance is the single most important diagnostic for validating a VibeDynaLITE run.

---

## The Energy Balance Equation

$$KE_{fabric} + SE + KE_{projectile} + E_{damped} + E_{fracture} + E_{clamped} \approx KE_{initial}$$

The left side should remain approximately equal to the initial projectile kinetic energy throughout the simulation. A small downward drift (energy loss) is acceptable — it means damping is doing its job. Any **upward** drift signals a bug.

---

## Energy Terms

### $KE_{fabric}$ — Fabric Kinetic Energy

The kinetic energy of all fabric nodes:

$$KE_{fabric} = \sum_i \frac{1}{2} m_i \|v_i\|^2$$

Computed by `compute_kinetic_energy()` in `energy.py`. This is typically near zero at time $t = 0$ (fabric starts at rest) and grows rapidly as the transverse wave propagates, then decreases as the projectile decelerates.

### $SE$ — Elastic Strain Energy

The potential energy stored in all active (non-failed) springs:

$$SE = \sum_j \frac{1}{2} k_j (\varepsilon_j \cdot L_{0,j})^2$$

Computed by `compute_strain_energy()` in `energy.py`. Only tensile strains are counted (compressive strain energy is zeroed via `maximum(0.0, strains)`). Failed springs contribute zero strain energy.

### $KE_{projectile}$ — Projectile Kinetic Energy

The rigid-body kinetic energy of the projectile:

$$KE_{projectile} = \frac{1}{2} m_{proj} \|v_{proj}\|^2$$

Computed inline in the fused loop. This starts at $KE_{initial}$ and decreases as the projectile decelerates. For a full perforation event, it levels off at a non-zero residual velocity. For a catch, it goes to zero.

### $E_{damped}$ — Damping Dissipation

The cumulative energy removed by [[Damping Models|damping]] forces:

$$E_{damped} = \int_0^t -F_{damp} \cdot v \, dt' \approx \sum_{n} (-P_{damp}^n) \cdot \Delta t$$

Tracked as a running sum (`damp_dissipated`) in the fused loop. This should grow monotonically — if it ever decreases, the damping model has a sign error.

### $E_{fracture}$ — Fracture Dissipation

Energy dissipated when springs break. See [[Spring Failure Mechanics]] for details. When a spring fails, the elastic strain energy it was storing is booked as dissipated fracture energy (multiplied by the configured `fracture_energy_multiplier` to account for sub-grid friction, fibrillation, and yarn pull-out).

### $E_{clamped}$ — Velocity Clamp Dissipation

Energy removed by the CFL velocity clamp (see [[CFL Stability Condition]]). When a node exceeds the maximum physical wave speed velocity $v_{max} = dx / dt$, its velocity is scaled down, and the excess kinetic energy is stored in a tracking buffer to preserve the energy balance.

---

## Current Implementation

The `compute_energy_balance()` function in `energy.py` returns a dictionary including all terms:

```python
{
    "kinetic": ke,
    "strain": se,
    "damped": damped,
    "failure_dissipated": failure_dissipated,
    "clamp_dissipated": clamp_dissipated,
    "projectile_kinetic": proj_ke,
    "total": ke + se + damped + failure_dissipated + clamp_dissipated + proj_ke
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
