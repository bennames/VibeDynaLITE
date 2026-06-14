# Mass Scaling

## What It Is

Mass scaling is a common technique in explicit dynamics where you **artificially increase the material density** (mass per unit area) beyond its physical value. This is done purely for computational convenience — it has no physical basis.

In VibeDynaLITE's mass-spring system, mass scaling would mean multiplying each node's mass by a factor $\alpha > 1$:

$$m_{\text{scaled}} = \alpha \cdot m_{\text{physical}}$$

## Why It's Tempting

Mass scaling delivers two major computational benefits:

### 1. Larger Critical Timestep

The [[CFL Stability Condition]] gives a critical timestep proportional to the square root of the node mass:

$$dt_{\text{crit}} \propto \sqrt{\frac{m}{k}}$$

Scaling mass by $\alpha$ increases the allowable timestep by $\sqrt{\alpha}$. A mass scaling factor of 100× means 10× fewer timesteps to cover the same physical duration.

### 2. Smaller Required Grid

The transverse wave speed is:

$$c \propto \frac{1}{\sqrt{\rho}}$$

Scaling mass by $\alpha$ reduces the wave speed by $\sqrt{\alpha}$, which proportionally reduces $R_{\min}$ (see [[Infinite Boundary Conditions]]). Fewer nodes are needed to prevent wave reflections.

The combined effect is dramatic: a 100× mass scaling factor can reduce total computation time by roughly 100× (10× fewer timesteps × 10× fewer nodes in each direction... minus overhead).

## Why It's Problematic for Ballistic Impact

Mass scaling is routinely used in quasi-static forming simulations, crash dynamics, and other applications where inertia effects are secondary. **Ballistic impact is not one of those applications.** The physics of fabric ballistic response is fundamentally governed by wave propagation and inertia, which mass scaling directly corrupts.

### The Projectile-to-Fabric Mass Ratio Changes

The single most important dimensionless parameter in ballistic impact is the ratio of projectile mass to engaged fabric mass. Mass scaling artificially inflates the fabric mass, making the projectile appear lighter relative to the fabric. This changes the energy partition between projectile deceleration and fabric acceleration — the core physics of the problem.

### Cunniff's Dimensionless Velocity

The Cunniff V50 model — the most widely-used analytical benchmark for textile armor — uses the dimensionless parameter:

$$U^* = \frac{\sigma \varepsilon}{2 \rho}$$

where $\sigma$ is tensile strength, $\varepsilon$ is failure strain, and $\rho$ is fiber density. Density appears **explicitly** in the denominator. Mass scaling changes $\rho$, directly corrupting the comparison to the analytical prediction and to physical test data.

### The Cone-vs-Failure Race

When a projectile strikes woven fabric, two things happen simultaneously:

1. **Primary yarns stretch** under the projectile and accumulate strain toward failure
2. **A transverse deflection cone spreads outward** at the wave speed, progressively engaging more yarns to share the load

The V50 is determined by the race between these two processes. If the cone spreads fast enough to engage enough yarns before the primary yarns fail, the fabric stops the projectile.

Mass scaling **slows the cone** (by reducing wave speed) without changing the strain accumulation rate in the primary yarns. The result: fewer yarns engage before failure, the fabric absorbs less energy, and the predicted V50 is **artificially low**. This is not a minor effect — even modest mass scaling factors (10×) can shift V50 predictions by 10–20%.

## When Mass Scaling IS Acceptable

- **Debugging:** Verifying solver logic, contact algorithms, or boundary conditions where physical accuracy is not the goal
- **Convergence studies:** Checking whether a result is mesh-converged, where relative differences (not absolute values) matter
- **Comparative (A vs B) studies:** If you're comparing two material configurations at the same mass scaling factor, the *relative* ranking may still be valid (but verify this for your specific case)

## When Mass Scaling is NOT Acceptable

- **Final V50 predictions** intended to be compared against physical test data
- **Backface deformation** predictions for behind-armor blunt trauma assessment
- **Any simulation** where the absolute value of the result matters, not just relative ranking

## Our Design Decision

We chose **not to implement mass scaling** in the VibeDynaLITE GUI. The risk of a user unknowingly running a mass-scaled simulation and treating the V50 result as physical is too high. The computational cost savings are real, but we address them through other means:

- **[[Compute Backends]]** — GPU-accelerated JAX and Taichi backends for raw throughput
- **[[Grid Sizing and Mesh Resolution]]** — Guidance on using coarser (but physically meaningful) dx values
- **[[Damping Models|Rayleigh damping]]** — For improved numerical stability without corrupting the wave physics

If you need mass scaling for debugging or research purposes, it can be applied manually by modifying the areal density in the material properties — but the GUI will not automate or encourage it.

---

## See Also

- [[Infinite Boundary Conditions]] — How mass scaling would reduce the required grid size
- [[CFL Stability Condition]] — How mass scaling increases the critical timestep
- [[Kevlar Material Properties]] — Physical density values for Kevlar fibers
- [[Energy Conservation]] — How mass scaling affects the energy balance
