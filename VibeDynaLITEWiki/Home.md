# VibeDynaLITE

**VibeDynaLITE** is an explicit dynamics simulation tool for modeling ballistic projectile impact on woven Kevlar fabric. It represents the fabric as a mass-spring grid where each node corresponds to a yarn crossover point, and springs capture the tensile, shear, and diagonal load paths between them. The solver uses [[Leapfrog Verlet Integration]] to march the system forward in time, with support for multiple [[Compute Backends]] (Numba, JAX, NumPy, Taichi).

The primary use case is predicting the **V50 ballistic limit** — the velocity at which a projectile has a 50% probability of penetrating a given fabric system — and understanding the deformation and failure mechanics of the fabric during impact.

---

## Theory & Physics

- [[Grid Sizing and Mesh Resolution]] — Choosing dx to balance speed and fidelity
- [[Infinite Boundary Conditions]] — Preventing spurious wave reflections from grid edges
- [[Damping Models]] — Viscous and Rayleigh damping for numerical stability
- [[Energy Conservation]] — Tracking kinetic, strain, and dissipated energy budgets
- [[Spring Failure Mechanics]] — Strain-based rupture criteria for yarn breakage
- [[Kevlar Material Properties]] — Fiber modulus, density, failure strain, and weave parameters
- [[Mass Scaling]] — Why we chose not to implement it, and what it would do
- [[Physics-Based Benchmarks]] — Continuous validation suite against first principles and analytical theories

## Solver Architecture

- [[Compute Backends]] — Numba, JAX, NumPy, and Taichi solver implementations
- [[CFL Stability Condition]] — Critical timestep calculation and the CFL safety factor
- [[Leapfrog Verlet Integration]] — The time integration scheme

## User Guide

- [[Simulation Configuration]] — Setting up materials, grids, projectiles, and solver parameters
- [[Interpreting Results]] — Reading energy plots, deformation snapshots, and V50 predictions

## Project History

- [[Changelog]] — Project change history and release notes
