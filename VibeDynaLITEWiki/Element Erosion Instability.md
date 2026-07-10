# Element Erosion Instability

When elements or springs erode (fail) in an explicit dynamic simulation, the sudden loss of stiffness and mass can trigger severe numerical instabilities. In **VibeDynaLITE**, this manifests as nodes being launched at extreme velocities (e.g., kinetic energy jumping from a few joules to millions of joules in a single chunk).

This document serves as a detailed learning guide and reference for understanding, preventing, and auditing element/spring erosion instabilities in VibeDynaLITE.

---

## 1. Physical & Numerical Mechanisms of Instability

### 1.1 The "Element Erosion Shotgun" (Force Discontinuity)
When a spring (in fabric mode) or a shell element (in metallic sheet mode) reaches its failure threshold, its internal stress is immediately zeroed to represent material rupture. 

* **Before failure**: The element/spring is carrying a large tensile load $F = k \cdot \varepsilon \cdot L_0$, which is balanced by neighboring elements/springs to maintain nodal equilibrium.
* **At failure**: The element's forces drop to zero instantly. However, the neighboring elements continue to exert their full forces on the shared nodes. This creates a massive **unbalanced force impulse**.
* **Result**: The shared nodes undergo extreme acceleration:
  \[
  a = \frac{F_{\text{unbalanced}}}{m_{\text{node}}}
  \]
  Because explicit time integration step size $dt$ is on the order of nanoseconds, these huge accelerations launch nodes at velocities far exceeding physical limits (creating the "shotgun" effect).

### 1.2 Uncapped Contact Force Spikes
When springs/elements fail at the impact zone, the projectile punches through the resulting gap.
* Nodes adjacent to the failure zone suddenly experience deep projectile penetration ($\delta$).
* In the penalty contact formulation:
  \[
  F_{\text{contact}} = k_{\text{penalty}} \cdot \delta + c \cdot \dot{\delta}
  \]
  Because there is no upper bound on $\delta$ or $F_{\text{contact}}$, deep penetrations under high-speed impact generate massive contact forces.
* If a node is completely freed from all active elements (meaning it has zero structural stiffness), but contact forces are still applied to it, the node gains massive energy without any restoring structural forces to arrest it.

### 1.3 State Persistence Reset (Numba/Taichi JIT Boundaries)
A critical implementation issue occurs when state arrays are not correctly passed between Python and the JIT solver across chunks:
* If arrays like `element_strains` or rotational DOFs (`ang_positions`, `ang_velocities`) are initialized to zero at the start of each integration chunk rather than being persisted, the solver interprets the sudden step change as a massive deformation impulse.
* On step 0 of a new chunk, the strain increment $\Delta \varepsilon = \varepsilon_{\text{current}} - \varepsilon_{\text{old}}$ equals the entire current strain (since $\varepsilon_{\text{old}} = 0$). This spikes stresses to several GPa, launching nodes and injecting huge amounts of spurious energy.

### 1.4 Phantom Energy Accounting
If the dissipated fracture/failure energy is computed analytically or globally rather than incrementally:
* Recomputing the total dissipated work from scratch each step can introduce enormous numerical artifacts.
* Furthermore, if the solver zero-outs failed forces but does not explicitly subtract/absorb the stored elastic strain energy from the nodes, the system "creates" energy because the nodes retain their deformation-driven kinetic energy while the accounting ledger records it as fully dissipated.

---

## 2. Best-Practice Mitigation Strategies (LS-DYNA Comparison)

To stabilize erosion and prevent non-physical energy growth, explicit dynamic codes employ several key numerical treatments:

| Instability Cause | LS-DYNA Implementation | VibeDynaLITE Target Formulation |
| :--- | :--- | :--- |
| **Force Discontinuity** | `SOFT` option in `*MAT_ADD_EROSION` (gradual force ramp-down over 5–10 steps) | **Softened Spring/Element Deletion**: Linearly decay forces from the failure point to zero over $N_{\text{ramp}}$ steps. |
| **Freed Node Contact** | Eroded nodes are removed from contact slave/master surfaces | **Dead-Node Contact Exclusion**: Skip penalty contact calculations for nodes where active element/spring count is 0. |
| **Unlimited Contact Force** | Segment-based contact with depth limits and $F_{\text{max}}$ caps | **Penalty Force Capping**: Limit $F_{\text{contact}}$ to the structural shear capacity of the local element. |
| **Spurious Velocity** | Node velocity capped by material wave speed ($c_p$) | **Longitudinal Wave Speed Clamping**: Clamp velocities at $2 \times c_p$ instead of the generous numerical limit $dx/dt$. |
| **Chunk Wave Reset** | Continuous history tracking throughout the time loop | **Persistent Solver State**: Pass and return all history-dependent arrays across the chunk boundary. |

---

## 3. Formulas for Softened Deletion

To implement softened deletion, each failed spring/element must track its "failure age" (number of timesteps since failure). The force scaling factor $\mathcal{S}$ is defined as:
\[
\mathcal{S} = \max\left(0.0, 1.0 - \frac{t - t_{\text{fail}}}{N_{\text{ramp}} \cdot dt}\right)
\]
During the softening phase:
1. The force applied to the nodes is scaled by $\mathcal{S}$:
   \[
   F_{\text{applied}} = \mathcal{S} \cdot F_{\text{nominal}}
   \]
2. The remaining strain energy is incrementally dissipated as work of failure rather than being dumped into nodal kinetic energy.
3. Once $t - t_{\text{fail}} \ge N_{\text{ramp}} \cdot dt$, the element is fully eroded, and $\mathcal{S} = 0.0$.
