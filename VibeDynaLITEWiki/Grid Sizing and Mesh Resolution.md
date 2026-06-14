# Grid Sizing and Mesh Resolution

## What dx Represents

In VibeDynaLITE, the grid spacing `dx` corresponds to the **physical distance between yarn crossover points** in a woven Kevlar fabric. Each node in the simulation represents one crossover — the point where a warp yarn and a weft yarn interlace.

For a standard **17 picks-per-inch** plain weave (the most common construction for Kevlar 29 ballistic fabric), the physical crossover spacing is:

$$dx_{\text{physical}} = \frac{25.4 \text{ mm}}{17} \approx 1.49 \text{ mm}$$

This is the "natural" mesh resolution for the fabric. Using `dx ≈ 1.5 mm` means every node in your simulation maps one-to-one to a real yarn crossover. Using a coarser dx means each node represents a *homogenized* patch of multiple crossovers.

## Choosing dx for Your Analysis

Not every simulation needs yarn-level resolution. The right choice of dx depends on **what quantity you're trying to predict**:

### V50 Ballistic Limit (coarse is often fine)

**Recommended dx: 5–10 mm**

V50 prediction is fundamentally an energy balance problem — does the fabric absorb enough energy to stop the projectile? The total energy capacity of the system is set by the material properties and total fabric area engaged, which are captured reasonably well even on a coarse grid. A 5–10 mm mesh typically gives V50 predictions within a few percent of a fully-resolved simulation, at a fraction of the cost.

### Backface Deformation / Cone Shape

**Recommended dx: 2–3 mm**

If you need to resolve the shape of the transverse deflection cone (important for behind-armor blunt trauma assessment), you need enough nodes across the cone to capture its profile. At 10 mm spacing the cone appears blocky; at 2–3 mm the shape is smooth and the peak deflection converges.

### Detailed Failure Progression

**Recommended dx: ~1.5 mm (yarn-level)**

If you need to track which yarns fail in what sequence — for example, to study the effect of oblique impact or multi-hit scenarios — you need the mesh to resolve individual yarns. Use `dx ≈ 1.5 mm` for a 17/in weave.

## Interaction with Boundary Sizing

Grid spacing has a direct impact on how many nodes you need for [[Infinite Boundary Conditions]]. The minimum grid radius $R_{\min}$ is set by the material's wave speed and the simulation duration — it's a *physical* distance that doesn't change when you change dx. But the number of nodes required to cover that distance is:

$$N = \frac{2 \, R_{\min}}{dx} + 1$$

So halving dx **quadruples** the total node count (in 2D), which can dramatically increase computation time. This is the core tradeoff: finer resolution gives better spatial detail in the impact zone but requires a much larger grid to satisfy the boundary condition.

## Effect on the Critical Timestep

The grid spacing dx also appears in the [[CFL Stability Condition]]. The critical timestep $dt_{\text{crit}}$ depends on the stiffest spring and the lightest node mass — and since cell mass scales as $m \propto dx^2$ (for a given areal density), reducing dx reduces the critical timestep. This compounds the cost: finer grids have more nodes *and* require more timesteps.

## Practical Recommendation

Start coarse, refine where needed:

1. **Screening runs** at dx = 10 mm to explore the parameter space quickly
2. **Convergence check** at dx = 5 mm to verify your V50 isn't mesh-sensitive
3. **Final high-fidelity** at dx = 1.5–3 mm only for the specific configurations you need detailed results for

---

## See Also

- [[Infinite Boundary Conditions]] — How dx affects the required number of nodes
- [[CFL Stability Condition]] — How dx affects the critical timestep
- [[Kevlar Material Properties]] — Yarn counts and weave construction parameters
