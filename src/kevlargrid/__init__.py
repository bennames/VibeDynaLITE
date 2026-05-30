"""KevlarGrid Explicit Solver — ballistic fabric impact simulation.

A lumped-mass explicit dynamics solver for woven Kevlar® fabric panels
subjected to high-velocity projectile impact.  The solver uses a spring–mass
grid discretisation with central-difference (Störmer–Verlet) time integration,
strain-based yarn failure, and Rayleigh damping.
"""

from __future__ import annotations

__version__ = "0.1.0"
