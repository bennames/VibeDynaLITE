# Spring Failure Mechanics

Spring failure is how VibeDynaLITE models yarn breakage in the Kevlar fabric. When a spring's strain exceeds the failure threshold, it permanently stops carrying load. Getting this right is critical — poor failure modeling is the #1 source of non-physical energy artifacts in ballistic simulations.

---

## Binary Failure (Current Model)

The current implementation in `failure.py` uses a simple binary check:

```python
failed |= strains > epsilon_fail
```

When a spring's engineering strain exceeds `failure_strain` (typically 0.036 for [[Kevlar Material Properties|Kevlar 29]]), its `failed` flag is set to `True` permanently. On subsequent timesteps, `compute_spring_forces()` in `forces.py` zeroes the force for any failed spring:

```python
f_mag = where(failed, 0.0, f_mag)
```

**This is simple and correct in principle, but it creates two serious numerical problems:**

### Problem 1: Force Discontinuity

At the moment of failure, the spring's stiffness drops from full $k$ to zero in a single timestep. If the spring was at 3.6% strain with a stiffness of ~23 MN (typical for Kevlar 29), the force magnitude was on the order of kilonewtons. That force vanishes instantly.

The nodes on either end of the spring were in dynamic equilibrium — their acceleration included a large restoring force from this spring. When that force disappears, the nodes are suddenly unbalanced and get "launched" by the remaining forces. This is the primary mechanism for non-physical energy injection.

### Problem 2: Lost Strain Energy

At the moment of failure, the spring was storing elastic strain energy:

$$SE_{spring} = \frac{1}{2} k (\varepsilon \cdot L_0)^2$$

For a Kevlar 29 spring at failure strain, this can be significant. When the spring's `failed` flag is set, `compute_strain_energy()` zeroes its contribution:

```python
se_springs = where(failed, 0.0, se_springs)
```

That stored energy simply vanishes from the energy balance — it doesn't appear in any dissipation term. See [[Energy Conservation]] for the full impact on energy tracking.

---

## Progressive Damage (Planned)

The planned upgrade replaces the binary cliff with a gradual linear stiffness degradation between two strain thresholds.

### Damage Variable

$$d = \text{clamp}\left(\frac{\varepsilon - \varepsilon_{onset}}{\varepsilon_{fail} - \varepsilon_{onset}},\; 0,\; 1\right)$$

where:
- $\varepsilon_{onset}$ = damage onset strain (default: 60% of `failure_strain`)
- $\varepsilon_{fail}$ = full failure strain
- $d = 0$ means undamaged, $d = 1$ means fully failed

### Effective Stiffness

$$k_{eff} = k \times (1 - d)$$

As strain increases from $\varepsilon_{onset}$ to $\varepsilon_{fail}$, the spring's stiffness linearly decreases from full $k$ to zero. The force carried by the spring at the moment of full failure is near zero — eliminating the discontinuous "cliff" that launches nodes.

### How It Changes the Force Profile

| Strain Range | Binary Model | Progressive Model |
|---|---|---|
| $\varepsilon < \varepsilon_{onset}$ | Full stiffness $k$ | Full stiffness $k$ |
| $\varepsilon_{onset} \leq \varepsilon < \varepsilon_{fail}$ | Full stiffness $k$ | Degrading: $k(1-d)$ |
| $\varepsilon = \varepsilon_{fail}$ | Instant drop to 0 | Smooth arrival at 0 |
| $\varepsilon > \varepsilon_{fail}$ | Zero force | Zero force |

> [!NOTE]
> The default `damage_onset_strain` at 60% of `failure_strain` gives a damage zone of ~1.4% strain for Kevlar 29 (from ~2.2% to 3.6%). This provides enough strain range for a smooth degradation without significantly affecting the pre-damage mechanical response.

### Interaction with Rayleigh Damping

Progressive damage interacts well with the stiffness-proportional ($\beta K$) term in [[Damping Models|Rayleigh damping]]. As a spring's effective stiffness degrades, the damping force through that spring naturally decreases too. This means springs in the damage zone get progressively less numerical damping, which is physically appropriate — a failing spring shouldn't be a major source of damping.

---

## Fracture Energy Dissipation

When a spring breaks, the elastic strain energy it was storing needs to be tracked as dissipated energy. Simply zeroing the strain energy without accounting for it creates an energy hole in the [[Energy Conservation|energy balance]].

### Basic Fracture Energy

At the moment of failure, the spring's stored elastic energy is:

$$E_{fracture,j} = \frac{1}{2} k_j (\varepsilon_{fail} \cdot L_{0,j})^2$$

### Kevlar Fracture Energy Multiplier

For Kevlar fibers, the actual energy dissipated during failure exceeds the simple elastic strain energy. The literature-based multiplier is **1.5** (conservative):

$$E_{fracture,j}^{total} = 1.5 \times \frac{1}{2} k_j (\varepsilon_{fail} \cdot L_{0,j})^2$$

**Justification for the 1.5× multiplier:**

| Component | Contribution |
|---|---|
| Elastic strain energy | 1.0× — Kevlar is linear-elastic to failure, so the ratio is essentially 1.0 for pure strain energy |
| Inter-yarn friction | +0.2× — Energy dissipated by yarn-on-yarn sliding at crossover points during failure |
| Fibrillation energy | +0.2× — Kevlar fails by axial splitting (fibrillation), not clean fracture. The splitting process dissipates energy |
| Viscoelastic dissipation | +0.1× — Rate-dependent effects not captured by the quasi-static mass-spring model |

**Key references:**
- Cunniff (1999) — Dimensionless ballistic performance parameters
- Phoenix & Porwal (2003) — Membrane model for ballistic impact
- Duan et al. (2005/2006) — Finite element modeling of Kevlar fabric impact
- Bazhenov (1997) — Dissipation of energy by dry-friction in ballistic nylon fabrics

> [!TIP]
> The 1.5× multiplier is conservative. If your energy balance still shows unexplained losses after adding fracture tracking, the multiplier may need to increase toward 2.0×. But fix ghost forces and implement progressive damage first — those are larger effects.

---

## Ghost Force Fix

The current fused loop processes each timestep in this order:

1. Compute contact forces (line 175)
2. Compute spring forces (line 215) ← **spring may be past failure strain**
3. Compute damping forces (line 225)
4. Integrate velocities and positions (line 239)
5. Detect failures (line 255) ← **too late — force was already applied**

A spring that exceeded `failure_strain` during the previous timestep still computes and applies its full force in step 2, then gets flagged as failed in step 5. For one timestep, it acts as a "ghost" — applying a large force that shouldn't exist.

**The fix:** Move failure detection to **before** force computation:

1. Compute strains and detect failures ← **moved up**
2. Compute contact forces
3. Compute spring forces ← **now sees updated `failed` flags**
4. Compute damping forces
5. Integrate velocities and positions

This is a targeted reordering of the fused loop, not a structural change. The strain calculation already exists in the failure detection block — it just needs to happen earlier.

> [!IMPORTANT]
> The ghost force fix should be implemented before progressive damage. It's a simpler change with a larger impact on energy conservation. Progressive damage further smooths the remaining discontinuity, but without the ghost force fix, even progressive damage will have a one-timestep artifact.

---

## See Also

- [[Energy Conservation]] — How fracture energy fits into the full energy balance
- [[Kevlar Material Properties]] — Material constants that determine failure strain and stiffness
