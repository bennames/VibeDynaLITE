"""Solver subpackage — core simulation engine.

Exposes the principal public names used to set up and run an explicit
dynamic simulation of a Kevlar spring–mass grid.
"""

from __future__ import annotations

from kevlargrid.solver.boundary import apply_clamped_boundary, compute_min_radius
from kevlargrid.solver.damping import rayleigh_damping, viscous_damping
from kevlargrid.solver.energy import (
    compute_energy_balance,
    compute_kinetic_energy,
    compute_strain_energy,
)
from kevlargrid.solver.failure import check_failures
from kevlargrid.solver.forces import compute_spring_forces, compute_spring_strains
from kevlargrid.solver.grid import Grid, generate_rectangular_grid
from kevlargrid.solver.integrator import leapfrog_step
from kevlargrid.solver.projectile import Projectile, distribute_contact_forces, update_contact_zone
from kevlargrid.solver.timestep import compute_cfl_timestep

__all__ = [
    "Grid",
    "Projectile",
    "apply_clamped_boundary",
    "check_failures",
    "compute_cfl_timestep",
    "compute_energy_balance",
    "compute_kinetic_energy",
    "compute_min_radius",
    "compute_spring_forces",
    "compute_spring_strains",
    "compute_strain_energy",
    "distribute_contact_forces",
    "generate_rectangular_grid",
    "leapfrog_step",
    "rayleigh_damping",
    "update_contact_zone",
    "viscous_damping",
]
