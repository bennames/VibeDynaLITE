# Damping Models

Damping forces remove kinetic energy from the simulation, serving two purposes: suppressing non-physical numerical oscillations and representing real energy dissipation (material internal friction, aerodynamic drag, etc.). VibeDynaLITE uses Mass- and Stiffness-Proportional Rayleigh Damping as the primary model, with a legacy Viscous Damping model available as a baseline comparison.

---

## Simple Viscous Damping (Legacy)

Viscous damping applies a constant damping coefficient to every node's velocity:

$$F_{damp} = -c \cdot v$$

where $c$ is the damping coefficient and $v$ is the node velocity vector.

In this model, damping is global and frequency-independent — it removes kinetic energy proportionally to velocity regardless of whether the motion is physical (the expanding transverse cone) or numerical noise (high-frequency ringing at the wavefront).

**Pros:**
- Simple to implement and tune — one scalar parameter.
- Stable and predictable energy dissipation.

**Cons:**
- Damps all frequencies equally, including the low-frequency cone deformation that you actually want to preserve.
- No physical basis — the same coefficient is applied to every node regardless of local stiffness or mass.
- Can artificially slow the transverse wave if the coefficient is set too high.

---

## Rayleigh Damping (Current Standard)

Rayleigh damping constructs the damping matrix as a linear combination of the mass and stiffness matrices:

$$C = \alpha M + \beta K$$

This gives two independent knobs that target different parts of the frequency spectrum.

### Mass-Proportional Damping ($\alpha M$)

The mass-proportional term produces a per-node force:

$$F_\alpha = -\alpha \cdot m_i \cdot v_i$$

This preferentially damps **low-frequency** rigid-body modes (whole-fabric translation, bulk swaying). For ballistic impact, you typically want very little of this — the large-scale cone motion *is* the physics you're trying to capture.

**Guideline:** Keep $\alpha$ small. A reasonable starting point is:

$$\alpha \approx 0.01 \times \omega_{min}$$

where $\omega_{min}$ is the lowest natural frequency of the system. Over-damping $\alpha$ will artificially slow the overall fabric response.

### Stiffness-Proportional Damping ($\beta K$)

The stiffness-proportional term is more involved. Rather than a simple per-node calculation, it operates **per-spring** using the relative velocity projected onto the spring axis, then accumulates the result back to nodes:

1. Compute the relative velocity between each spring's two end-nodes
2. Project that velocity onto the spring's current axis direction
3. Scale by $\beta \times k_{spring}$ to get a damping force magnitude
4. Distribute the damping force equally and oppositely to both end-nodes

This preferentially damps **high-frequency** oscillations — exactly the numerical ringing and wavefront noise that plagues ballistic impact simulations.

**Key advantage:** The $\beta K$ term naturally adapts to local damage state. When a spring softens or fails, its stiffness drops, and so does the damping force through that spring. Intact, stiff springs at the wavefront get more damping. This selectively targets the numerical noise without over-damping the physical deformation.

**Guideline:** A typical starting point for Kevlar ballistic impact is:

$$\beta = 0.01 \times dt_{crit}$$

i.e., 1% of the critical timestep. This provides meaningful high-frequency damping without significantly affecting the stable timestep.

### Implementation Status

Rayleigh damping is fully implemented and compiled inside the optimized JIT leapfrog loops. 

1. **Parameter Exposure**: `Rayleigh Alpha` (mass-proportional) and `Rayleigh Beta` (stiffness-proportional) are exposed as slider widgets in the GUI Config Panel.
2. **JIT Compilation**: The stiffness-proportional damping is vectorized and compiled via JIT in `fused.py` and `taichi_solver.py`. 
3. **Dissipated Energy Integration**: The power dissipated by both mass-proportional ($P_{\alpha} = - \alpha \sum m_i \|v_i\|^2$) and stiffness-proportional ($P_{\beta} = \beta \sum k_j (v_{\text{rel}, j} \cdot u_j)^2$) damping is integrated at every timestep and stored in `damp_dissipated`.
4. **GUI Damping Model Selector**: A "Damping Model" dropdown combo box exposes the option to toggle between `"Rayleigh Damping"` (default, with stable settings of $\alpha=0.0$, $\beta=0.00001$) and `"Viscous Damping"` (with a default coefficient of $0.05$). Selecting a model dynamically shows the relevant input widgets in the sidebar and hides the others.


---

## CFL Impact of Stiffness-Proportional Damping

Adding $\beta K$ damping modifies the critical timestep. The undamped critical timestep is:

$$dt_{crit} = \frac{2}{\omega_{max}}$$

With damping ratio $\xi$, this becomes:

$$dt_{crit} = \frac{2}{\omega_{max}} \left(\sqrt{1 + \xi^2} - \xi\right)$$

where $\xi$ is the damping ratio contributed by the stiffness-proportional term. In practice, for the small $\beta$ values used in ballistic impact ($\beta \approx 0.01 \times dt_{crit}$), this is a small reduction — typically **5–10%** off the undamped critical timestep. The CFL safety factor already provides a margin that usually absorbs this.

See [[CFL Stability Condition]] for the full timestep calculation, including the penalty contact stiffness contribution.

---

## Energy Accounting

All damping dissipates energy, and that energy must be tracked to maintain the [[Energy Conservation|energy balance]]. The instantaneous power dissipated by damping is:

$$P_{damp} = F_{damp} \cdot v$$

The cumulative dissipated energy is integrated over time:

$$E_{damped} += -P_{damp} \cdot dt$$

The current implementation in the fused loop tracks this correctly for viscous damping (lines 226–227 of `fused.py`):

```python
p_damp = sum(damp_forces * velocities)
damp_dissipated += -p_damp * dt
```

With Rayleigh damping, both the mass-proportional and stiffness-proportional power dissipation must be accumulated separately (or summed), since $P = F_{damp} \cdot v$ applies to the combined force. The sign convention ensures that damping always *removes* energy from the system (the dot product $F_{damp} \cdot v$ is always negative for a dissipative force opposing velocity).

---

## See Also

- [[Energy Conservation]] — Full energy balance equation and what causes non-physical energy growth
- [[CFL Stability Condition]] — How damping affects the stable timestep
- [[Spring Failure Mechanics]] — How spring damage interacts with stiffness-proportional damping
