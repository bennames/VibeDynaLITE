# CFL Stability Condition

The Courant–Friedrichs–Lewy (CFL) condition is the fundamental stability constraint for VibeDynaLITE's explicit time integrator. Violating it causes the simulation to blow up — exponentially growing velocities, positions flying to infinity, and NaN propagation. Respecting it (with margin) keeps the simulation stable.

---

## The Core Idea

The CFL condition says: **no information can travel more than one grid cell per timestep.** In a mass-spring system, "information" travels as elastic waves through the spring network. If the timestep is too large, the wave outruns the integrator's ability to track it, and the solution becomes unstable.

---

## Critical Timestep

The critical (maximum stable) timestep is derived from the highest natural frequency in the system:

$$dt_{crit} = \sqrt{\frac{m_{min}}{k_{max}}}$$

where:
- $m_{min}$ = the smallest lumped mass of any node
- $k_{max}$ = the largest stiffness of any spring element

This is computed by `compute_cfl_timestep()` in `timestep.py`:

```python
m_min = np.min(masses)
k_max = np.max(stiffnesses)
dt_crit = np.sqrt(m_min / k_max)
```

### What Determines $k_{max}$?

In VibeDynaLITE, $k_{max}$ is **not** just the yarn spring stiffness. The penalty contact stiffness used for projectile–fabric and inter-ply contact is typically much larger:

$$k_{penalty} = 10 \times k_{ortho}$$

where $k_{ortho}$ is the orthogonal (yarn-direction) spring stiffness. This means the contact stiffness usually controls the critical timestep, not the fabric stiffness.

> [!IMPORTANT]
> If you change the penalty stiffness multiplier, you will change the critical timestep. Doubling the penalty stiffness reduces $dt_{crit}$ by a factor of $\sqrt{2} \approx 1.41$, which means ~41% more timesteps for the same simulation duration.

### What Determines $m_{min}$?

Corner nodes have the smallest lumped mass (¼ of a cell mass). Edge nodes have ½. Interior nodes have the full cell mass. So the corners typically control $m_{min}$:

$$m_{min} = \frac{1}{4} \cdot \rho_A \cdot dx^2$$

where $\rho_A$ is the areal density and $dx$ is the grid spacing.

---

## CFL Safety Factor

The actual timestep is the critical timestep multiplied by a safety factor:

$$dt = CFL \times dt_{crit}$$

Typical values for the CFL factor are **0.5 to 0.9**. The `compute_cfl_timestep()` function enforces $0 < CFL \leq 1.0$ and raises a `ValueError` otherwise.

**Choosing the CFL factor:**

| CFL | Character | When to Use |
|---|---|---|
| 0.9 | Aggressive | Confident in the physics, need speed, no damage modeling |
| 0.7 | Standard | Default for most ballistic impact runs |
| 0.5 | Conservative | New material parameters, debugging energy issues, progressive damage active |

> [!TIP]
> Start with CFL = 0.7. If the energy balance is well-behaved (see [[Energy Conservation]]), you can push toward 0.9. If you see any sign of instability (oscillating energy, nodes flying off), drop to 0.5 and investigate.

---

## Velocity Clamping

Even with a properly chosen timestep, individual nodes can occasionally exceed the maximum physical velocity — especially near the impact zone or at freshly failed springs. The CFL-implied maximum velocity is:

$$v_{max} = \frac{dx}{dt}$$

Any node moving faster than this is violating the CFL condition locally. The velocity clamping mechanism:

1. Checks each node's speed against $v_{max}$
2. Clamps the velocity magnitude to $v_{max}$ while preserving direction
3. Tracks the excess kinetic energy removed as `clamp_dissipated`

$$E_{clamped} = \sum_i \frac{1}{2} m_i \left(\|v_i\|^2 - v_{max}^2\right) \quad \text{for nodes where } \|v_i\| > v_{max}$$

This is a safety net, not a substitute for a correct timestep. In a well-behaved simulation, velocity clamping should trigger rarely and dissipate negligible energy. If it's firing frequently, something else is wrong (ghost forces, excessive penalty stiffness, CFL factor too high).

> [!WARNING]
> Velocity clamp dissipation is not yet tracked in the current implementation (planned for Sprint 7.7). Until then, clamping events silently remove energy from the system without it appearing in the [[Energy Conservation|energy balance]].

---

## Effect of Rayleigh Damping on CFL

Adding stiffness-proportional [[Damping Models|Rayleigh damping]] ($\beta K$) modifies the critical timestep. The undamped critical timestep is:

$$dt_{crit}^{undamped} = \frac{2}{\omega_{max}}$$

With damping, this becomes:

$$dt_{crit}^{damped} = \frac{2}{\omega_{max}} \left(\sqrt{1 + \xi^2} - \xi\right)$$

where $\xi$ is the modal damping ratio. For the recommended $\beta$ values ($\beta \approx 0.01 \times dt_{crit}$), the damping ratio is small and the timestep reduction is typically **5–10%**. The CFL safety factor usually provides enough margin to absorb this without needing to explicitly recompute $dt_{crit}$.

If you're running at CFL = 0.9 and enabling Rayleigh damping, consider dropping to CFL = 0.8 to maintain margin.

---

## Practical Example

For a Kevlar 29 fabric with 17 × 17 yarn count, single ply:

| Parameter | Value |
|---|---|
| $dx$ | 1.49 mm |
| $\rho_A$ | 0.47 kg/m² |
| $m_{corner}$ | $0.25 \times 0.47 \times (0.00149)^2 = 2.61 \times 10^{-7}$ kg |
| $k_{ortho}$ | $71 \times 10^9 \times (0.47 / 1440) = 2.32 \times 10^7$ N/m |
| $k_{penalty}$ | $10 \times 2.32 \times 10^7 = 2.32 \times 10^8$ N/m |
| $dt_{crit}$ | $\sqrt{2.61 \times 10^{-7} / 2.32 \times 10^8} = 3.35 \times 10^{-8}$ s |
| $dt$ (CFL=0.7) | $2.35 \times 10^{-8}$ s ≈ **23.5 ns** |

At 23.5 ns per step, a 100 μs simulation requires ~4,250 steps — very manageable.

---

## See Also

- [[Damping Models]] — How Rayleigh damping affects the critical timestep
- [[Grid Sizing and Mesh Resolution]] — How grid spacing affects the timestep
- [[Energy Conservation]] — How velocity clamping fits into the energy balance
