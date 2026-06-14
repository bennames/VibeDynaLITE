# Infinite Boundary Conditions

## Why Infinite Boundaries Matter

When a projectile strikes a fabric, stress waves radiate outward from the impact point at the material's transverse wave speed. In a physical test, the fabric panel is typically much larger than the impact event — the waves propagate outward and never come back during the time of interest.

In a simulation with a finite grid, those waves **reflect off the edges** and return to the impact zone, introducing spurious forces that corrupt the result. The reflected waves can artificially stiffen the fabric response, lower the computed backface deformation, and bias the V50 prediction.

The solution is to make the grid large enough that the outward-traveling wave front never reaches the boundary during the simulation. We call this an **infinite boundary condition** — not because the grid is literally infinite, but because it is large enough to *behave* as if it were.

## Computing the Minimum Radius

The minimum required grid half-width is:

$$R_{\min} = c_{\text{transverse}} \times t_{\text{sim}} \times 1.5$$

where:

- $c_{\text{transverse}}$ is the fastest stress wave speed in the fabric (m/s)
- $t_{\text{sim}}$ is the total simulation duration (s)
- $1.5$ is a safety factor (the wave must not reach the boundary *and* reflect back, plus a margin for numerical dispersion)

### Wave Speed

The transverse wave speed on the discrete grid is:

$$c = dx \sqrt{\frac{k}{m}}$$

where:

- $dx$ is the grid spacing (m)
- $k$ is the orthogonal spring stiffness (N/m), computed as $k = E \cdot t$ with $E$ the fiber modulus and $t$ the effective fabric thickness
- $m$ is the cell mass (kg), computed as $m = \rho_A \cdot dx^2$ with $\rho_A$ the areal density

Note that $c$ is **independent of dx** — substituting the expressions for $k$ and $m$:

$$c = dx \sqrt{\frac{E \cdot t}{\rho_A \cdot dx^2}} = \sqrt{\frac{E \cdot t}{\rho_A}}$$

This confirms that the wave speed is a *material property*, not a mesh property. Changing dx does not change $R_{\min}$.

## Grid Sizing from $R_{\min}$

The full grid length needed along each axis is:

$$L_{\text{grid}} = 2 \times R_{\min}$$

(The factor of 2 accounts for the impact point being at the center, with waves traveling outward in both directions.)

The actual grid length in terms of nodes and spacing is:

$$L_{\text{actual}} = (N_x - 1) \times dx$$

Setting these equal and solving for the required node count:

$$N_x = \left\lceil \frac{2 \, R_{\min}}{dx} \right\rceil + 1$$

This is the key relationship: **changing dx doesn't reduce the physical distance the grid must cover**, but it does change how many nodes you need to cover it. See [[Grid Sizing and Mesh Resolution]] for the implications.

## Current Implementation

The physics formula is implemented in the GUI at `config_panel.py` lines 379–402. When the user selects "Infinite Grid (Auto)" as the boundary condition, the panel:

1. Computes the fabric thickness from areal density and fiber density
2. Derives the orthogonal spring stiffness $k = E \cdot t$
3. Calculates cell mass $m = \rho_A \cdot dx^2$
4. Computes the transverse wave speed $c = dx \sqrt{k/m}$
5. Calls `compute_min_radius(c, t_{\text{sim}}, 1.5)` to get $R_{\min}$
6. Displays $R_{\min}$ in the GUI for the user

The underlying `compute_min_radius` function lives in `solver/boundary.py` and simply returns $c \cdot t \cdot f_s$.

### Node Count Clamp

When the user clicks "Apply Infinite Boundary Dimensions", the computed $N_x$ is currently **clamped to a maximum of 101 nodes** (line 713 of `config_panel.py`). This was a temporary safeguard to prevent the GUI from freezing on very large grids before the solver's parallel backends were fully optimized.

> **Note:** This clamp is being raised to 501 nodes as the Numba and JAX backends can now handle larger grids efficiently.

If the clamped value is smaller than the physics requires, the simulation will have reflections. The GUI displays $R_{\min}$ so the user can verify whether the applied grid size is adequate.

## Interaction with Mass Scaling

[[Mass Scaling]] would reduce the wave speed ($c \propto 1/\sqrt{\rho}$), which in turn reduces $R_{\min}$ and the required grid size. This is one of the practical motivations for mass scaling — but it comes with significant accuracy tradeoffs for ballistic impact. See the [[Mass Scaling]] page for details on why we chose not to implement it.

---

## See Also

- [[Grid Sizing and Mesh Resolution]] — How dx affects node counts and computational cost
- [[Mass Scaling]] — The temptation to reduce wave speed, and why we resist it
- [[CFL Stability Condition]] — The other place where wave speed matters
- [[Kevlar Material Properties]] — Source values for modulus, density, and areal density
