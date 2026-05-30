"""Unit conversion utilities."""

from __future__ import annotations


def n_to_lbf(n: float) -> float:
    """Convert Newtons to Pounds-force.

    Args:
        n: Force in Newtons.

    Returns:
        float: Force in lbf.
    """
    return n * 0.2248089431


def lbf_to_n(lbf: float) -> float:
    """Convert Pounds-force to Newtons.

    Args:
        lbf: Force in lbf.

    Returns:
        float: Force in Newtons.
    """
    return lbf / 0.2248089431


def mm_to_m(mm: float) -> float:
    """Convert millimeters to meters.

    Args:
        mm: Length in mm.

    Returns:
        float: Length in meters.
    """
    return mm / 1000.0


def gpa_to_pa(gpa: float) -> float:
    """Convert Gigapascals to Pascals.

    Args:
        gpa: Pressure/modulus in GPa.

    Returns:
        float: Pressure/modulus in Pascals.
    """
    return gpa * 1e9
