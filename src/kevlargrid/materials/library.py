"""Built-in material library for KevlarGrid."""

from __future__ import annotations

MATERIALS: dict[str, dict[str, str | float | tuple[int, int]]] = {
    "Kevlar 29": {
        "tensile_modulus_gpa": 71.0,
        "failure_strain": 0.036,
        "tensile_strength_gpa": 2.92,
        "fiber_density_gcc": 1.44,
        "areal_density_kgm2": 0.47,
        "shear_ratio": 0.002,
        "fabric_style": "745",
        "denier": 3000,
        "yarn_count": (17, 17),
        "crimp_factor": 0.10,
    },
    "Kevlar 49": {
        "tensile_modulus_gpa": 112.4,
        "failure_strain": 0.024,
        "tensile_strength_gpa": 3.00,
        "fiber_density_gcc": 1.44,
        "areal_density_kgm2": 0.23,
        "shear_ratio": 0.002,
        "fabric_style": "328",
        "denier": 1140,
        "yarn_count": (17, 17),
        "crimp_factor": 0.10,
    },
    "Kevlar KM2": {
        "tensile_modulus_gpa": 84.62,
        "failure_strain": 0.0355,
        "tensile_strength_gpa": 3.40,
        "fiber_density_gcc": 1.44,
        "areal_density_kgm2": 0.180,
        "shear_ratio": 0.002,
        "fabric_style": "706",
        "denier": 600,
        "yarn_count": (34, 34),
        "crimp_factor": 0.10,
    },
    "Corten Steel (14 Gauge)": {
        "tensile_modulus_gpa": 200.0,
        "failure_strain": 0.20,
        "tensile_strength_gpa": 0.485,
        "fiber_density_gcc": 7.85,
        "areal_density_kgm2": 15.7,
        "shear_ratio": 0.38,
        "crimp_factor": 0.0,
        "yarn_count": (1, 1),
        "material_model": "j2_plasticity",
        "yield_strength_gpa": 0.345,
        "hardening_modulus_gpa": 1.0,
        "ultimate_strain": 0.20,
        "poisson_ratio": 0.30,
    },
}


def get_material(name: str) -> dict[str, str | float | tuple[int, int]]:
    """Retrieve material properties by name.

    Args:
        name: Name of the Kevlar material.

    Returns:
        dict: Properties dictionary.
    """
    if name not in MATERIALS:
        raise KeyError(f"Material '{name}' not found. Available: {list(MATERIALS.keys())}")
    return MATERIALS[name]


def list_materials() -> list[str]:
    """Get the names of all registered materials.

    Returns:
        list[str]: Registered material names.
    """
    return list(MATERIALS.keys())
