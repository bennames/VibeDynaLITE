"""JSON config save, load, and validation.

Provides dynamic serialization, deserialization, and schema validation
for KevlarGrid simulation parameters.
"""

from __future__ import annotations

import json
import os

from kevlargrid.utils import get_logger

logger = get_logger("io.config")


class ValidationError(ValueError):
    """Raised when configuration validation fails."""

    pass


def save_config(config: dict, path: str) -> None:
    """Save configuration to JSON file.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    path : str
        Output file path.
    """
    # Create parent directories if they don't exist
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    logger.info("Configuration saved successfully to path: %s", path)


def load_config(path: str) -> dict:
    """Load configuration from JSON file.

    Parameters
    ----------
    path : str
        Path to configuration file.

    Returns
    -------
    dict
        Loaded configuration dictionary.
    """
    if not os.path.exists(path):
        logger.error("Configuration file not found at path: %s", path)
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, encoding="utf-8") as f:
        config = json.load(f)
    logger.info("Configuration loaded successfully from path: %s", path)
    return config  # type: ignore[no-any-return]


def validate_config(config: dict) -> bool:
    """Validate configuration format and parameter ranges.

    Parameters
    ----------
    config : dict
        Configuration dictionary.

    Returns
    -------
    bool
        True if valid, otherwise raises ValidationError.
    """
    # 1. Structural requirements
    required_sections = ["material", "grid", "projectile", "simulation"]
    for sec in required_sections:
        if sec not in config or not isinstance(config[sec], dict):
            msg = f"Missing or invalid section: '{sec}'"
            logger.warning("Configuration validation failed: %s", msg)
            raise ValidationError(msg)

    # 2. Material validation
    mat = config["material"]
    required_mat = [
        "name",
        "tensile_modulus_gpa",
        "failure_strain",
        "tensile_strength_gpa",
        "fiber_density_gcc",
        "areal_density_kgm2",
        "shear_ratio",
    ]
    for key in required_mat:
        if key not in mat:
            raise ValidationError(f"Material section missing required key: '{key}'")

    if not isinstance(mat["name"], str) or not mat["name"]:
        raise ValidationError("Material name must be a non-empty string.")

    for key in required_mat[1:]:
        val = mat[key]
        if not isinstance(val, (int, float)) or val <= 0.0:
            raise ValidationError(
                f"Material property '{key}' must be a positive number (got {val})."
            )

    if "crimp_factor" in mat:
        cf = mat["crimp_factor"]
        if not isinstance(cf, (int, float)) or cf < 0.0:
            raise ValidationError(f"crimp_factor must be non-negative (got {cf}).")

    if "yarn_count" in mat:
        yc = mat["yarn_count"]
        if (
            not isinstance(yc, list)
            or len(yc) != 2
            or not all(isinstance(x, int) and x > 0 for x in yc)
        ):
            raise ValidationError(f"yarn_count must be a list of two positive integers (got {yc}).")

    # 3. Grid validation
    grid = config["grid"]
    for key in ["nx", "ny", "dx", "n_plies", "boundary_type"]:
        if key not in grid:
            raise ValidationError(f"Grid section missing required key: '{key}'")

    for key in ["nx", "ny", "n_plies"]:
        val = grid[key]
        if not isinstance(val, int) or val <= 0:
            raise ValidationError(f"Grid parameter '{key}' must be a positive integer (got {val}).")

    if not isinstance(grid["dx"], (int, float)) or grid["dx"] <= 0.0:
        raise ValidationError(f"Grid parameter 'dx' must be a positive number (got {grid['dx']}).")

    if grid["boundary_type"] not in ["fixed", "infinite"]:
        raise ValidationError(
            f"Grid boundary_type must be 'fixed' or 'infinite' (got '{grid['boundary_type']}')."
        )

    if "t_ply" in grid and grid["t_ply"] is not None:
        t_ply = grid["t_ply"]
        if not isinstance(t_ply, (int, float)) or t_ply <= 0.0:
            raise ValidationError(
                f"Grid parameter 't_ply' must be a positive number or null (got {t_ply})."
            )

    # 4. Projectile validation
    proj = config["projectile"]
    for key in ["mass", "velocity", "position", "blade_width", "edge_thickness"]:
        if key not in proj:
            raise ValidationError(f"Projectile section missing required key: '{key}'")

    for key in ["mass", "blade_width", "edge_thickness"]:
        val = proj[key]
        if not isinstance(val, (int, float)) or val <= 0.0:
            raise ValidationError(
                f"Projectile parameter '{key}' must be a positive number (got {val})."
            )

    for key in ["velocity", "position"]:
        val = proj[key]
        if (
            not isinstance(val, list)
            or len(val) != 3
            or not all(isinstance(x, (int, float)) for x in val)
        ):
            raise ValidationError(
                f"Projectile parameter '{key}' must be a list of three numbers (got {val})."
            )

    # 5. Simulation validation
    sim = config["simulation"]
    for key in ["duration", "cfl_factor", "damping_coefficient"]:
        if key not in sim:
            raise ValidationError(f"Simulation section missing required key: '{key}'")

    if not isinstance(sim["duration"], (int, float)) or sim["duration"] <= 0.0:
        raise ValidationError(
            f"Simulation parameter 'duration' must be a positive number (got {sim['duration']})."
        )

    cfl = sim["cfl_factor"]
    if not isinstance(cfl, (int, float)) or cfl <= 0.0 or cfl > 1.0:
        raise ValidationError(
            f"Simulation parameter 'cfl_factor' must be in the range (0.0, 1.0] (got {cfl})."
        )

    damp = sim["damping_coefficient"]
    if not isinstance(damp, (int, float)) or damp < 0.0:
        raise ValidationError(
            f"Simulation parameter 'damping_coefficient' must be a non-negative number (got {damp})."
        )

    logger.info("Configuration validation succeeded.")
    return True
