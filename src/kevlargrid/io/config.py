"""TOML config save, load, and validation.

Provides dynamic serialization, deserialization, and schema validation
for KevlarGrid simulation parameters.
"""

from __future__ import annotations

import json
import os
import re
import tomllib
from typing import Any

from kevlargrid.utils import get_logger

logger = get_logger("io.config")


class ValidationError(ValueError):
    """Raised when configuration validation fails."""

    pass


UNIT_PATTERN = re.compile(r"^\s*([0-9.eE+-]+)\s*([a-zA-Z0-9*/³²^\-\[\]]+)?\s*$")


def parse_unit_value(val: Any, expected_base_unit: str) -> float | int:
    """Parse a value that may be a float, int, or a string with units, and convert to base units."""
    if isinstance(val, (int, float)):
        return val
    if not isinstance(val, str):
        raise ValidationError(
            f"Value must be a number or string with units, got {type(val).__name__}"
        )

    match = UNIT_PATTERN.match(val)
    if not match:
        raise ValidationError(f"Invalid format for value with units: '{val}'")

    num_str, unit_str = match.groups()
    try:
        num = float(num_str) if ("." in num_str or "e" in num_str.lower()) else int(num_str)
    except ValueError as e:
        raise ValidationError(f"Invalid numeric part in '{val}': {e}") from e

    if not unit_str:
        return num

    unit = unit_str.lower().strip()

    if expected_base_unit == "m":
        if unit in ("m", "meter", "meters"):
            return num
        elif unit in ("mm", "millimeter", "millimeters"):
            return num * 1e-3
        elif unit in ("cm", "centimeter", "centimeters"):
            return num * 1e-2
    elif expected_base_unit == "kg":
        if unit in ("kg", "kilogram", "kilograms"):
            return num
        elif unit in ("g", "gram", "grams"):
            return num * 1e-3
    elif expected_base_unit == "m/s":
        if unit in ("m/s", "ms-1", "m*s-1", "m/sec", "meters/second"):
            return num
    elif expected_base_unit == "s":
        if unit in ("s", "sec", "second", "seconds"):
            return num
        elif unit in ("ms", "millisecond", "milliseconds"):
            return num * 1e-3
        elif unit in ("us", "microsecond", "microseconds"):
            return num * 1e-6
        elif unit in ("ns", "nanosecond", "nanoseconds"):
            return num * 1e-9
    elif expected_base_unit == "gpa":
        if unit in ("gpa", "gigapascal", "gigapascals"):
            return num
        elif unit in ("pa", "pascal", "pascals"):
            return num * 1e-9
        elif unit in ("mpa", "megapascal", "megapascals"):
            return num * 1e-3
    elif expected_base_unit == "gcc":
        if unit in ("gcc", "g/cm3", "g/cm^3", "g/cc", "grams/cc"):
            return num
        elif unit in ("kg/m3", "kg/m^3"):
            return num * 1e-3
    elif expected_base_unit == "kgm2":
        if unit in ("kgm2", "kg/m2", "kg/m^2"):
            return num
        elif unit in ("g/m2", "g/m^2"):
            return num * 1e-3

    raise ValidationError(
        f"Unknown or incompatible unit '{unit_str}' for expected unit type '{expected_base_unit}'"
    )


def normalize_old_config_keys(config: dict) -> dict:
    """Normalize old config key layouts to the current schema format."""
    # Ensure all required sections exist or create them
    for sec in ["material", "grid", "projectile", "simulation"]:
        if sec not in config or not isinstance(config[sec], dict):
            config[sec] = {}

    # 1. Projectile section mapping
    proj = config["projectile"]
    if "mass_kg" in proj:
        proj["mass"] = proj.pop("mass_kg")
    if "velocity_ms" in proj:
        v_ms = proj.pop("velocity_ms")
        if isinstance(v_ms, (int, float)):
            proj["velocity"] = [0.0, 0.0, float(v_ms)]
        elif isinstance(v_ms, str):
            proj["velocity"] = [0.0, 0.0, v_ms]
    if "blade_width_mm" in proj:
        proj["blade_width"] = proj.pop("blade_width_mm")
    if "edge_thickness_mm" in proj:
        proj["edge_thickness"] = proj.pop("edge_thickness_mm")

    # 2. Simulation -> Grid section mapping
    sim = config["simulation"]
    grid = config["grid"]
    if "plies" in sim:
        grid["n_plies"] = sim.pop("plies")
    if "boundary" in sim:
        grid["boundary_type"] = sim.pop("boundary")
    if "cfl" in sim:
        sim["cfl_factor"] = sim.pop("cfl")
    if "ply_spacing_mm" in sim:
        grid["t_ply"] = sim.pop("ply_spacing_mm")

    # Fill defaults for missing grid sizes if converting old format
    if "nx" not in grid:
        grid["nx"] = 11
    if "ny" not in grid:
        grid["ny"] = 11
    if "dx" not in grid:
        grid["dx"] = "10 mm"
    if "boundary_type" not in grid:
        grid["boundary_type"] = "fixed"

    # 3. Damping section mapping (moves to simulation)
    if "damping" in config and isinstance(config["damping"], dict):
        damp = config.pop("damping")
        if "model" in damp:
            sim["damping_model"] = damp["model"]
        if "coefficient" in damp:
            coeff = damp["coefficient"]
            if coeff == "auto":
                sim["damping_coefficient"] = 0.05
            else:
                sim["damping_coefficient"] = coeff

    return config


def normalize_config_units(config: dict) -> None:
    """Parse and normalize all unit strings in the config to numerical base values."""
    if "material" in config and isinstance(config["material"], dict):
        mat = config["material"]
        if "tensile_modulus_gpa" in mat:
            mat["tensile_modulus_gpa"] = parse_unit_value(mat["tensile_modulus_gpa"], "gpa")
        if "tensile_strength_gpa" in mat:
            mat["tensile_strength_gpa"] = parse_unit_value(mat["tensile_strength_gpa"], "gpa")
        if "fiber_density_gcc" in mat:
            mat["fiber_density_gcc"] = parse_unit_value(mat["fiber_density_gcc"], "gcc")
        if "areal_density_kgm2" in mat:
            mat["areal_density_kgm2"] = parse_unit_value(mat["areal_density_kgm2"], "kgm2")

    if "grid" in config and isinstance(config["grid"], dict):
        grid = config["grid"]
        if "dx" in grid:
            grid["dx"] = parse_unit_value(grid["dx"], "m")
        if "t_ply" in grid and grid["t_ply"] is not None:
            grid["t_ply"] = parse_unit_value(grid["t_ply"], "m")

    if "projectile" in config and isinstance(config["projectile"], dict):
        proj = config["projectile"]
        if "mass" in proj:
            proj["mass"] = parse_unit_value(proj["mass"], "kg")
        if "blade_width" in proj:
            proj["blade_width"] = parse_unit_value(proj["blade_width"], "m")
        if "edge_thickness" in proj:
            proj["edge_thickness"] = parse_unit_value(proj["edge_thickness"], "m")
        if "caliber" in proj:
            proj["caliber"] = parse_unit_value(proj["caliber"], "m")
        if "total_length" in proj:
            proj["total_length"] = parse_unit_value(proj["total_length"], "m")
        if "edge_radius" in proj:
            proj["edge_radius"] = parse_unit_value(proj["edge_radius"], "m")
        if "span" in proj:
            proj["span"] = parse_unit_value(proj["span"], "m")
        if "root_chord" in proj:
            proj["root_chord"] = parse_unit_value(proj["root_chord"], "m")
        if "tip_chord" in proj:
            proj["tip_chord"] = parse_unit_value(proj["tip_chord"], "m")
        if "tip_radius" in proj:
            proj["tip_radius"] = parse_unit_value(proj["tip_radius"], "m")
        if "radius" in proj:
            proj["radius"] = parse_unit_value(proj["radius"], "m")
        if "length" in proj:
            proj["length"] = parse_unit_value(proj["length"], "m")
        if "velocity" in proj and isinstance(proj["velocity"], list):
            proj["velocity"] = [parse_unit_value(v, "m/s") for v in proj["velocity"]]
        if "position" in proj and isinstance(proj["position"], list):
            proj["position"] = [parse_unit_value(p, "m") for p in proj["position"]]
        if "omega" in proj and isinstance(proj["omega"], list):
            proj["omega"] = [parse_unit_value(w, "m/s") if isinstance(w, str) else float(w) for w in proj["omega"]]
        if "quat" in proj and isinstance(proj["quat"], list):
            proj["quat"] = [float(q) for q in proj["quat"]]

    if "simulation" in config and isinstance(config["simulation"], dict):
        sim = config["simulation"]
        if "duration" in sim:
            sim["duration"] = parse_unit_value(sim["duration"], "s")
        if "dt" in sim:
            sim["dt"] = parse_unit_value(sim["dt"], "s")
        if "rayleigh_beta" in sim:
            sim["rayleigh_beta"] = parse_unit_value(sim["rayleigh_beta"], "s")


def serialize_toml_val(v: Any) -> str:
    """Helper to serialize values to TOML compliant string syntax."""
    if isinstance(v, bool):
        return "true" if v else "false"
    elif isinstance(v, (int, float)):
        return str(v)
    elif isinstance(v, str):
        escaped = v.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    elif isinstance(v, list):
        items = [serialize_toml_val(x) for x in v]
        return f"[{', '.join(items)}]"
    elif v is None:
        return '""'
    else:
        raise TypeError(f"Unsupported TOML type: {type(v)}")


def serialize_toml(config: dict) -> str:
    """Custom serializer of dict to TOML string representation."""
    lines = []
    # Write top-level metadata first if any (e.g. version)
    for k, v in config.items():
        if not isinstance(v, dict):
            lines.append(f"{k} = {serialize_toml_val(v)}")

    # Write tables
    for section, content in config.items():
        if isinstance(content, dict):
            if lines:
                lines.append("")
            lines.append(f"[{section}]")
            for k, v in content.items():
                if v is not None:  # Omit None values
                    lines.append(f"{k} = {serialize_toml_val(v)}")
    return "\n".join(lines) + "\n"


def save_config(config: dict, path: str) -> None:
    """Save configuration to TOML file.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    path : str
        Output file path.
    """
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)

    toml_str = serialize_toml(config)

    with open(path, "w", encoding="utf-8") as f:
        f.write(toml_str)
    logger.info("Configuration saved successfully to TOML path: %s", path)


def load_config(path: str) -> dict:
    """Load configuration from TOML (or legacy JSON) file.

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

    with open(path, "rb") as f:
        content = f.read()

    # Try JSON fallback first if path ends with .json
    if path.endswith(".json"):
        try:
            config = json.loads(content.decode("utf-8"))
            logger.info("Configuration loaded from legacy JSON at path: %s", path)
            config = normalize_old_config_keys(config)
            normalize_config_units(config)
            return config
        except Exception as e:
            logger.warning("Failed to parse JSON file: %s. Trying TOML...", e)

    try:
        config = tomllib.loads(content.decode("utf-8"))
        logger.info("Configuration loaded from TOML at path: %s", path)
    except Exception as toml_err:
        # Fallback to JSON load
        try:
            config = json.loads(content.decode("utf-8"))
            logger.info("Configuration loaded from legacy JSON fallback at path: %s", path)
        except Exception as json_err:
            raise toml_err from json_err

    # Normalize old keys and units to float
    config = normalize_old_config_keys(config)
    normalize_config_units(config)
    return config


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

    if "fracture_energy_multiplier" in mat:
        fem = mat["fracture_energy_multiplier"]
        if not isinstance(fem, (int, float)) or fem < 1.0:
            raise ValidationError(
                f"fracture_energy_multiplier must be a number >= 1.0 (got {fem})."
            )

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

    if grid["boundary_type"] not in ["fixed", "infinite", "non-reflecting"]:
        raise ValidationError(
            f"Grid boundary_type must be 'fixed', 'infinite', or 'non-reflecting' (got '{grid['boundary_type']}')."
        )

    if "t_ply" in grid and grid["t_ply"] is not None:
        t_ply = grid["t_ply"]
        if not isinstance(t_ply, (int, float)) or t_ply <= 0.0:
            raise ValidationError(
                f"Grid parameter 't_ply' must be a positive number or null (got {t_ply})."
            )

    # 4. Projectile validation
    # 4. Projectile validation
    proj = config["projectile"]
    shape = proj.get("shape", "box").lower()
    if shape not in ["box", "sphere", "cylinder", "bullet", "propeller"]:
        raise ValidationError(f"Invalid projectile shape: '{shape}'")

    required_keys = ["mass", "velocity", "position"]
    if shape == "box":
        required_keys += ["blade_width", "edge_thickness"]

    for key in required_keys:
        if key not in proj:
            raise ValidationError(f"Projectile section missing required key: '{key}' for shape '{shape}'")

    for key in ["mass"] + (["blade_width", "edge_thickness"] if shape == "box" else []):
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

    if "omega" in proj:
        val = proj["omega"]
        if (
            not isinstance(val, list)
            or len(val) != 3
            or not all(isinstance(x, (int, float)) for x in val)
        ):
            raise ValidationError(
                f"Projectile parameter 'omega' must be a list of three numbers (got {val})."
            )

    if "quat" in proj:
        val = proj["quat"]
        if (
            not isinstance(val, list)
            or len(val) != 4
            or not all(isinstance(x, (int, float)) for x in val)
        ):
            raise ValidationError(
                f"Projectile parameter 'quat' must be a list of four numbers (got {val})."
            )

    # Shape specific validations
    if shape == "sphere":
        # Check radius or caliber
        r = proj.get("radius", proj.get("caliber", None))
        if r is not None and (not isinstance(r, (int, float)) or r <= 0.0):
            raise ValidationError(f"Sphere radius/caliber must be a positive number (got {r}).")
    elif shape == "cylinder":
        r = proj.get("radius", proj.get("caliber", None))
        l = proj.get("length", proj.get("total_length", None))
        if r is not None and (not isinstance(r, (int, float)) or r <= 0.0):
            raise ValidationError(f"Cylinder radius/caliber must be a positive number (got {r}).")
        if l is not None and (not isinstance(l, (int, float)) or l <= 0.0):
            raise ValidationError(f"Cylinder length/total_length must be a positive number (got {l}).")
    elif shape == "bullet":
        r = proj.get("radius", proj.get("caliber", None))
        l = proj.get("length", proj.get("total_length", None))
        m = proj.get("ogive_multiplier", 2.0)
        if r is not None and (not isinstance(r, (int, float)) or r <= 0.0):
            raise ValidationError(f"Bullet radius/caliber must be a positive number (got {r}).")
        if l is not None and (not isinstance(l, (int, float)) or l <= 0.0):
            raise ValidationError(f"Bullet length/total_length must be a positive number (got {l}).")
        if not isinstance(m, (int, float)) or m <= 0.0:
            raise ValidationError(f"Bullet ogive_multiplier must be a positive number (got {m}).")
    elif shape == "propeller":
        span = proj.get("span", 0.05)
        r_c = proj.get("root_chord", 0.01)
        t_c = proj.get("tip_chord", 0.005)
        twist = proj.get("twist", 15.0)
        thick = proj.get("thickness_ratio", 12.0)
        t_r = proj.get("tip_radius", 0.002)
        if not isinstance(span, (int, float)) or span <= 0.0:
            raise ValidationError(f"Propeller span must be a positive number (got {span}).")
        if not isinstance(r_c, (int, float)) or r_c <= 0.0:
            raise ValidationError(f"Propeller root_chord must be a positive number (got {r_c}).")
        if not isinstance(t_c, (int, float)) or t_c <= 0.0:
            raise ValidationError(f"Propeller tip_chord must be a positive number (got {t_c}).")
        if not isinstance(twist, (int, float)):
            raise ValidationError(f"Propeller twist must be a number (got {twist}).")
        if not isinstance(thick, (int, float)) or thick <= 0.0:
            raise ValidationError(f"Propeller thickness_ratio must be a positive number (got {thick}).")
        if not isinstance(t_r, (int, float)) or t_r <= 0.0:
            raise ValidationError(f"Propeller tip_radius must be a positive number (got {t_r}).")

    # 5. Simulation validation
    sim = config["simulation"]
    model = sim.get("damping_model")
    if model is None:
        if "rayleigh_beta" in sim and sim["rayleigh_beta"] > 0.0:
            model = "rayleigh"
        elif "damping_coefficient" in sim:
            model = "viscous"
        else:
            model = "rayleigh"
    sim["damping_model"] = model

    if "damping_coefficient" not in sim:
        sim["damping_coefficient"] = 0.05
    if "rayleigh_alpha" not in sim:
        sim["rayleigh_alpha"] = 0.0
    if "rayleigh_beta" not in sim:
        sim["rayleigh_beta"] = 1e-9
    if "auto_cfl" not in sim:
        sim["auto_cfl"] = True
    if "dt" not in sim:
        sim["dt"] = 1.5e-7

    for key in [
        "duration",
        "cfl_factor",
        "damping_model",
        "damping_coefficient",
        "rayleigh_alpha",
        "rayleigh_beta",
        "auto_cfl",
        "dt",
    ]:
        if key not in sim:
            raise ValidationError(f"Simulation section missing required key: '{key}'")

    if not isinstance(sim["auto_cfl"], bool):
        raise ValidationError(
            f"Simulation parameter 'auto_cfl' must be a boolean (got {type(sim['auto_cfl']).__name__})."
        )

    if not isinstance(sim["dt"], (int, float)) or sim["dt"] <= 0.0:
        raise ValidationError(
            f"Simulation parameter 'dt' must be a positive number (got {sim['dt']})."
        )

    if not isinstance(sim["duration"], (int, float)) or sim["duration"] <= 0.0:
        raise ValidationError(
            f"Simulation parameter 'duration' must be a positive number (got {sim['duration']})."
        )

    cfl = sim["cfl_factor"]
    if not isinstance(cfl, (int, float)) or cfl <= 0.0 or cfl > 1.0:
        raise ValidationError(
            f"Simulation parameter 'cfl_factor' must be in the range (0.0, 1.0] (got {cfl})."
        )

    damp_model = sim["damping_model"]
    if damp_model not in ["rayleigh", "viscous"]:
        raise ValidationError(
            f"Simulation parameter 'damping_model' must be 'rayleigh' or 'viscous' (got '{damp_model}')."
        )

    coeff = sim["damping_coefficient"]
    if not isinstance(coeff, (int, float)) or coeff < 0.0:
        raise ValidationError(
            f"Simulation parameter 'damping_coefficient' must be a non-negative number (got {coeff})."
        )

    alpha = sim["rayleigh_alpha"]
    if not isinstance(alpha, (int, float)) or alpha < 0.0:
        raise ValidationError(
            f"Simulation parameter 'rayleigh_alpha' must be a non-negative number (got {alpha})."
        )

    beta = sim["rayleigh_beta"]
    if not isinstance(beta, (int, float)) or beta < 0.0:
        raise ValidationError(
            f"Simulation parameter 'rayleigh_beta' must be a non-negative number (got {beta})."
        )

    logger.info("Configuration validation succeeded.")
    return True
