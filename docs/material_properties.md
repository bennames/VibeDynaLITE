# Kevlar Material Properties — Reference Data

This document contains the material property values used in the KevlarGrid Explicit Solver's built-in material library. All values are sourced from published literature and manufacturer datasheets.

---

## 1. Fiber-Level Mechanical Properties

### Kevlar 29 (Heavy Ballistic)

| Property | Value | Unit | Source |
|----------|-------|------|--------|
| Fiber density ρ | 1.44 | g/cm³ | DuPont Kevlar Technical Guide |
| Tensile modulus E | 71.0 | GPa | DuPont Technical Guide (yarn value) |
| Ultimate tensile strength σ_ult | 2.92 | GPa | DuPont Technical Guide (yarn) |
| Failure strain ε_fail | 3.6 | % | DuPont Technical Guide |
| Longitudinal shear modulus G_LT | ~2.0 | GPa | Estimated from similar aramid fiber data |

### Kevlar 49 (High Modulus)

| Property | Value | Unit | Source |
|----------|-------|------|--------|
| Fiber density ρ | 1.44 | g/cm³ | DuPont Kevlar Technical Guide |
| Tensile modulus E | 112.4 | GPa | DuPont Technical Guide (yarn value) |
| Ultimate tensile strength σ_ult | 3.00 | GPa | DuPont Technical Guide (yarn) |
| Failure strain ε_fail | 2.4 | % | DuPont Technical Guide |
| Longitudinal shear modulus G_LT | ~2.0 | GPa | Estimated from similar aramid fiber data |

### Kevlar KM2 (High Performance Ballistic)

| Property | Value | Unit | Source |
|----------|-------|------|--------|
| Fiber density ρ | 1.44 | g/cm³ | DuPont; ARL reports |
| Tensile modulus E | 84.62 ± 4.18 | GPa | Cheng, Chen & Weerasooriya (2005) |
| Ultimate tensile strength σ_ult (virgin fiber) | 3.88 ± 0.40 | GPa | Cheng et al. (2005) |
| Ultimate tensile strength σ_ult (from woven fabric) | ~3.40 | GPa | Cheng et al. (2005); ARL reports |
| Failure strain ε_fail (virgin fiber) | 4.52 ± 0.37 | % | Cheng et al. (2005) |
| Failure strain ε_fail (from woven fabric) | ~3.55 | % | Cheng et al. (2005); ARL reports |
| Longitudinal shear modulus G_LT (single fiber) | 24.4 ± 2.4 | GPa | Cheng et al. (2005) |
| Transverse elastic modulus E_T | 1.34 ± 0.35 | GPa | Cheng et al. (2005) |

> **IMPORTANT — KM2 Weaving Degradation:** Weaving degrades fiber properties by approximately 15%. Virgin fiber values (3.88 GPa strength, 4.52% strain) should NOT be used for woven fabric modeling. Use the "from fabric" values (3.40 GPa, 3.55%) as conservative defaults.

---

## 2. Reference Woven Fabric Styles & Areal Densities

| Fabric Style | Fiber | Weave | Denier | Count (ypi) | Areal Density | Source |
|-------------|-------|-------|--------|-------------|----------------|--------|
| **Style 745** | Kevlar 29 | Plain | 3000 | ~17×17 | 0.47 kg/m² (≈13.9 oz/yd²) | Armorco; Fiberglass Supply TDS |
| **Style 328** | Kevlar 49 | Plain | 1140 | ~17×17 | 0.23 kg/m² (≈6.8 oz/yd²) | TMI-SLC TDS; CST Sales |
| **Style 706** | Kevlar KM2 | Plain | 600 | 34×34 | 0.180 kg/m² | ARL reports (DTIC) |

---

## 3. In-Plane Shear Modulus (Woven Fabric)

This is the most complex property. Dry woven fabric shear is **highly nonlinear** and structure-dependent, governed by yarn rotation at crossovers rather than fiber material properties.

> **CRITICAL DISTINCTION:** The fiber shear modulus (e.g., KM2: 24.4 GPa) is NOT the fabric in-plane shear modulus. Fabric shear stiffness is orders of magnitude lower for dry (unresinized) fabrics.

| Configuration | G value | G/E Ratio | Source |
|---------------|---------|-----------|--------|
| Dry woven Kevlar (initial linear region) | 0.01–0.05 GPa | ~0.0001–0.0007 | Picture frame test literature |
| Kevlar/epoxy composite laminate | 4–5 GPa | ~0.13–0.20 | Performance Composites data |

### Recommendations for the Mass-Spring Model

For **dry woven fabric** (the primary use case in this solver):
- Use linearized fabric G ≈ **0.03 GPa** for all three variants
- This yields G/E ratios of approximately:
  - Kevlar 29: G/E ≈ 0.0004
  - Kevlar 49: G/E ≈ 0.0003
  - Kevlar KM2: G/E ≈ 0.0004

These are the values used for diagonal (shear) spring stiffness derivation:
```
k_shear = k_ortho × (G_fabric / E_fiber)
```

> **NOTE:** These G/E ratios are much smaller than the 0.03–0.05 range initially assumed in the PRD. This is physically correct — dry fabric has very low shear resistance. The diagonal springs will be very compliant, primarily serving to prevent zero-energy hourglass modes rather than carrying significant load.

---

## 4. Axial Spring Stiffness Derivation

Spring stiffness is derived from fiber/yarn properties using the approach from Phoenix & Porwal (2003):

```
k_axial = (E_fiber × A_yarn) / L_element
```

Where:
- `E_fiber` = fiber tensile modulus (GPa)
- `A_yarn` = yarn cross-sectional area = `(Denier / 9000) / ρ_fiber` (m²)
- `L_element` = element/spring rest length in the mesh (m)

### Example: Kevlar KM2 Style 706

```
E       = 84.62 GPa = 84.62 × 10⁹ Pa
Denier  = 600
ρ_fiber = 1440 kg/m³
A_yarn  = (600 / 9000) / 1440 = 4.63 × 10⁻⁸ m²
```

**Per unit width** (yarn spacing `s = 1/34 inch = 0.747 mm`):
```
k_per_unit_width = E × A / s² = 84.62e9 × 4.63e-8 / (7.47e-4)²
                 ≈ 7.02 × 10⁶ N/m per meter of fabric width
                 ≈ 7.0 MN/m/m
```

### Crimp Correction

Real woven fabrics have crimped (undulating) yarns that produce a nonlinear initial toe region before the yarn fully straightens and engages its elastic stiffness. A crimp correction factor should be applied:

- **Crimp factor range:** 0.06–0.20 of elastic stiffness for the initial loading phase
- **Effective initial stiffness:** 0.42–1.4 MN/m per meter width (for KM2 Style 706)
- **Recommendation:** Use a crimp factor of **0.10** as default, user-configurable

---

## 5. Recommended Default Values for Built-in Library

These are the values that will be programmed into the solver's material dropdown:

| Property | Kevlar 29 | Kevlar 49 | Kevlar KM2 | Unit |
|----------|-----------|-----------|------------|------|
| **Fiber density ρ** | 1.44 | 1.44 | 1.44 | g/cm³ |
| **Tensile modulus E** | 71.0 | 112.4 | 84.62 | GPa |
| **Tensile strength σ_ult** | 2.92 | 3.00 | 3.40 | GPa |
| **Failure strain ε_fail** | 3.6 | 2.4 | 3.55 | % |
| **Reference fabric style** | 745 | 328 | 706 | — |
| **Fabric areal density** | 0.47 | 0.23 | 0.180 | kg/m² |
| **Fabric in-plane shear G** | 0.03 | 0.03 | 0.03 | GPa |
| **G/E ratio** | 0.0004 | 0.0003 | 0.0004 | — |
| **Fabric denier** | 3000 | 1140 | 600 | — |
| **Yarn count** | 17×17 | 17×17 | 34×34 | ypi |
| **Crimp factor** | 0.10 | 0.10 | 0.10 | — |

---

## 6. Source Citations

1. **DuPont Kevlar Technical Guide** — Fiber properties for K29, K49 (density, modulus, strength, elongation). Available from DuPont.com.

2. **Cheng, M., Chen, W., & Weerasooriya, T. (2005).** "Mechanical Properties of Kevlar KM2 Single Fiber." *J. Eng. Mater. Technol.*, 127(2), 197–203. — Primary source for KM2 single-fiber properties (E=84.62 GPa, σ=3.88 GPa, ε=4.52%, G_LT=24.4 GPa, E_T=1.34 GPa).

3. **Phoenix, S.L. & Porwal, P.K. (2003).** "A New Membrane Model for the Ballistic Impact Response and V50 Performance of Multi-Ply Fibrous Systems." *Int. J. Solids Struct.*, 40(24), 6723–6765. — Analytical membrane model framework and spring stiffness derivation approach.

4. **Cunniff, P.M. (1999).** "Dimensionless Parameters for Optimization of Textile-Based Body Armor Systems." *Proc. 18th Int. Symp. on Ballistics*, San Antonio, TX. — Cunniff U* parameter for fiber screening.

5. **Tabiei, A. & Nilakantan, G. (2008).** "Ballistic Impact of Dry Woven Fabric Composites: A Review." *Appl. Mech. Rev.*, 61(1), 010801. — Comprehensive review of FE modeling approaches and required input parameters.

6. **ARL Technical Reports (various, via DTIC)** — KM2 Style 706 fabric characterization, weaving degradation effects, dynamic properties.

7. **MIL-HDBK-17 / CMH-17 Volume 2** — Statistically-based aramid composite allowables. Useful for resin-impregnated laminate properties.

8. **FAA AC 20-128A** — "Design Considerations for Minimizing Hazards Caused by Uncontained Turbine Engine and APU Rotor Failure." Performance-based guidance for containment design methodology.

---

## 7. Known Discrepancies & Modeling Notes

- **K29 modulus:** DuPont reports 70.5 GPa (yarn) vs. 83 GPa (impregnated strand, ASTM D2343). We use **71 GPa** for dry yarn applications.
- **K49 modulus:** Ranges from 112–131 GPa depending on source. DuPont official yarn value is **112.4 GPa**.
- **KM2 strength/strain:** Virgin fiber values (3.88 GPa, 4.52%) are higher than woven fabric values (3.40 GPa, 3.55%) due to weaving damage. **We use fabric values** for woven barrier modeling.
- **Shear stiffness:** The diagonal spring stiffness derived from G/E ≈ 0.0003–0.0004 is very small. This is physically correct for dry fabric. The shear springs primarily serve as numerical stabilizers, not load carriers. Users modeling resin-impregnated fabric should override with G/E ≈ 0.15.
- **Strain-rate effects:** At strain rates >100 s⁻¹ (typical in ballistic impact), Kevlar fibers stiffen. The factor is approximately 1 + 0.02 × ln(rate/rate_ref). This is planned as a future enhancement and is NOT included in the MVP material library.
