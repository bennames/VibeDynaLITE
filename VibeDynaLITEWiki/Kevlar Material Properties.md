# Kevlar Material Properties

Kevlar is a para-aramid synthetic fiber manufactured by DuPont. It's the workhorse of soft body armor due to its exceptional specific strength and energy absorption capacity. VibeDynaLITE includes a built-in material library (`materials/library.py`) with pre-configured properties for three Kevlar variants: **Kevlar 29**, **Kevlar 49**, and **Kevlar KM2**.

---

## Kevlar Grades

### Kevlar 29 — Ballistic Grade

Kevlar 29 is the standard choice for ballistic applications. It has lower modulus but higher elongation to failure than Kevlar 49, giving it better energy absorption.

| Property | Value | Unit |
|---|---|---|
| Tensile modulus | 71.0 | GPa |
| Tensile strength | 2.92 | GPa |
| Failure strain | 3.6% | — |
| Fiber density | 1.44 | g/cm³ |
| Areal density (Style 745) | 0.47 | kg/m² |
| Denier | 3000 | — |
| Yarn count | 17 × 17 | yarns/inch |
| Shear ratio | 0.0004 | — |
| Crimp factor | 0.10 | — |

### Kevlar 49 — Structural Grade

Kevlar 49 is optimized for stiffness, not energy absorption. Higher modulus (112 GPa) but lower failure strain (2.4%) makes it less suitable for ballistic applications but excellent for composite reinforcement.

| Property | Value | Unit |
|---|---|---|
| Tensile modulus | 112.4 | GPa |
| Tensile strength | 3.00 | GPa |
| Failure strain | 2.4% | — |
| Fiber density | 1.44 | g/cm³ |
| Areal density (Style 328) | 0.23 | kg/m² |
| Denier | 1140 | — |
| Yarn count | 17 × 17 | yarns/inch |

### Kevlar KM2 — Advanced Ballistic Grade

KM2 is the latest generation ballistic fiber with improved properties over Kevlar 29. Higher modulus *and* higher failure strain compared to K29 makes it the premium choice for body armor.

| Property | Value | Unit |
|---|---|---|
| Tensile modulus | 84.62 | GPa |
| Tensile strength | 3.40 | GPa |
| Failure strain | 3.55% | — |
| Fiber density | 1.44 | g/cm³ |
| Areal density (Style 706) | 0.180 | kg/m² |
| Denier | 600 | — |
| Yarn count | 34 × 34 | yarns/inch |

> [!NOTE]
> All three Kevlar variants share the same fiber density (1.44 g/cm³) — it's the same base polymer (poly-paraphenylene terephthalamide). The mechanical differences come from the spinning process and molecular orientation.

---

## Weave Architectures

### Plain Weave

Plain weave is the most common architecture for ballistic Kevlar. Each warp yarn passes alternately over and under each weft yarn, creating a 1/1 interlacing pattern. This produces the maximum number of yarn crossovers, which:

- Maximizes inter-yarn friction (energy dissipation during impact)
- Provides balanced properties in warp and weft directions
- Creates the highest fabric stability (yarns resist sliding)

### Yarn Count and Grid Spacing

Typical ballistic Kevlar uses yarn counts from **17 × 17/inch** (Style 745, coarse) to **34 × 34/inch** (Style 706, fine). The yarn crossover spacing defines the natural physical scale for the mass-spring model:

$$dx = \frac{25.4 \text{ mm}}{n_{yarns/inch}}$$

For a 17 × 17 fabric: $dx \approx 1.49$ mm
For a 34 × 34 fabric: $dx \approx 0.75$ mm

This crossover spacing maps directly to the [[Grid Sizing and Mesh Resolution|grid spacing]] parameter `dx` in VibeDynaLITE. Each node in the grid represents a yarn crossover point, and each spring represents a yarn segment between crossovers.

### Crimp

Real woven yarns are not straight — they undulate over and under the cross-yarns. The **crimp factor** (0.10 in the library, meaning 10%) represents the excess yarn length consumed by this undulation. This affects:

- Effective in-plane stiffness (crimped yarns must first straighten before carrying full load)
- The initial non-linear toe region in the stress-strain curve (not currently modeled)
- Actual yarn length per unit fabric length

---

## Why Kevlar Is Special for Simulation

### Linear-Elastic to Failure

Unlike metals (which yield and plastically deform before breaking) or polymers (which show viscoelastic creep), Kevlar fibers are **linear-elastic all the way to failure**. The stress-strain curve is a straight line from zero to fracture. This means:

- The mass-spring model with constant stiffness $k$ is an excellent approximation up to the damage zone
- There's no need for a plasticity model or yield surface
- The failure strain is a clean, well-defined threshold

### High Specific Strength

Kevlar 29's strength-to-weight ratio is approximately 5× that of steel. In a ballistic context, what matters is the specific energy absorption — how much kinetic energy a given areal mass of fabric can absorb.

### Anisotropic Failure (Fibrillation)

Kevlar fibers don't fracture cleanly like glass or carbon. They fail by **axial splitting** — the fiber breaks apart along its length into smaller fibrils (fibrillation). This failure mode:

- Dissipates more energy than a clean fracture (hence the 1.5× fracture energy multiplier in [[Spring Failure Mechanics]])
- Creates a damage zone rather than a sharp crack tip
- Is difficult to model explicitly in a mass-spring framework — hence the use of empirical multipliers

---

## The Cunniff $U^*$ Parameter

Cunniff (1999) proposed a dimensionless parameter that combines the key material properties into a single ballistic performance metric:

$$U^* = \frac{\sigma_f \varepsilon_f}{2 \rho} \sqrt{\frac{E}{\rho}}$$

where:
- $\sigma_f$ = tensile strength
- $\varepsilon_f$ = failure strain
- $\rho$ = density
- $E$ = tensile modulus

The first part ($\sigma_f \varepsilon_f / 2\rho$) is the specific strain energy to failure. The square root part ($\sqrt{E/\rho}$) is the longitudinal wave speed. Their product captures both how much energy the material can absorb and how quickly it can spread the impact load.

Higher $U^*$ = better ballistic performance. Among common fibers:

| Fiber | $U^*$ (relative) |
|---|---|
| Kevlar 29 | 1.0 (baseline) |
| Kevlar KM2 | ~1.2 |
| Dyneema (UHMWPE) | ~1.5 |
| Steel | ~0.1 |

> [!TIP]
> $U^*$ is useful for quick material screening, but the actual $V_{50}$ (ballistic limit velocity) depends heavily on fabric architecture, boundary conditions, and projectile geometry — all of which VibeDynaLITE captures through the simulation itself.

---

## How Properties Map to the Solver

The material dictionary feeds directly into grid generation (`grid.py`):

| Material Property | Solver Parameter | Calculation |
|---|---|---|
| `areal_density_kgm2` | `masses` (per node) | $m = \rho_A \cdot dx^2$ (with tributary area scaling at boundaries) |
| `tensile_modulus_gpa` + `areal_density_kgm2` + `fiber_density_gcc` | `stiffnesses` (orthogonal) | $k_{ortho} = E \cdot t$ where $t = \rho_A / \rho_f$ |
| `shear_ratio` | `stiffnesses` (diagonal) | $k_{shear} = k_{ortho} \times \text{shear\_ratio}$ |
| `failure_strain` | `epsilon_fail` | Direct mapping |

---

## See Also

- [[Grid Sizing and Mesh Resolution]] — How yarn spacing maps to grid `dx`
- [[Spring Failure Mechanics]] — Failure models and the fracture energy multiplier
- [[Mass Scaling]] — How multi-ply configurations scale mass and stiffness
