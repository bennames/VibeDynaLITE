"""High-performance trajectory archiver using HDF5.

Saves multi-ply mass-spring grid positions, velocities, energies, spring statuses,
and solver configurations into an open hierarchical binary format.
"""

from __future__ import annotations

import json
from typing import Any

import h5py
import numpy as np


def export_to_h5(
    config: dict[str, Any],
    results_report: dict[str, Any],
    history: list[dict[str, Any]],
    filepath: str,
) -> None:
    """Export simulation results and history data to an open HDF5 format.

    Parameters
    ----------
    config : dict
        The active simulation configuration dictionary.
    results_report : dict
        A dictionary containing final summary outcomes and metrics.
    history : list of dict
        The frame history containing node coordinates, failures, and energies.
    filepath : str
        Output file path.
    """
    if not history:
        raise ValueError("Cannot export empty simulation history.")

    # Prepare arrays
    steps = len(history)
    first_frame = history[0]
    n_nodes = first_frame["nodes"].shape[0]
    n_springs = first_frame["failed"].shape[0]

    times = np.zeros(steps, dtype=np.float32)
    positions = np.zeros((steps, n_nodes, 3), dtype=np.float32)
    failed = np.zeros((steps, n_springs), dtype=bool)
    proj_pos = np.zeros((steps, 3), dtype=np.float32)

    # Energies matrix: KE, SE, Damped, Contact, Total
    energies = np.zeros((steps, 5), dtype=np.float32)

    for i, frame in enumerate(history):
        times[i] = float(frame["time"])
        positions[i] = frame["nodes"]
        failed[i] = frame["failed"]
        proj_pos[i] = frame["projectile_pos"]

        energies[i, 0] = float(frame.get("ke", 0.0))
        energies[i, 1] = float(frame.get("se", 0.0))
        energies[i, 2] = float(frame.get("damped", 0.0))
        energies[i, 3] = float(frame.get("contact", 0.0))
        energies[i, 4] = float(frame.get("total", 0.0))

    # Write HDF5 File
    with h5py.File(filepath, "w") as f:
        # 1. Metadata JSON
        meta_grp = f.create_group("metadata")
        meta_grp.create_dataset(
            "config_json",
            data=json.dumps(config, indent=2),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )

        # 2. Results Summary Group
        res_grp = f.create_group("results")
        res_grp.attrs["arrested"] = bool(results_report.get("arrested", False))
        res_grp.attrs["peak_deceleration_g"] = float(results_report.get("peak_deceleration_g", 0.0))
        res_grp.attrs["yarn_rupture_percentage"] = float(
            results_report.get("yarn_rupture_percentage", 0.0)
        )
        res_grp.attrs["residual_velocity_ms"] = float(
            results_report.get("residual_velocity_ms", 0.0)
        )
        res_grp.attrs["energy_dissipation_efficiency"] = float(
            results_report.get("energy_dissipation_efficiency", 0.0)
        )
        res_grp.attrs["max_layer_perforated"] = int(results_report.get("max_layer_perforated", -1))

        # 3. Time History Group (with gzip compression for efficiency)
        hist_grp = f.create_group("time_history")
        hist_grp.create_dataset("time", data=times, compression="gzip")
        hist_grp.create_dataset("positions", data=positions, compression="gzip")
        hist_grp.create_dataset("spring_failures", data=failed, compression="gzip")
        hist_grp.create_dataset("projectile_pos", data=proj_pos, compression="gzip")

        # Energies matrix
        e_dataset = hist_grp.create_dataset("energies", data=energies, compression="gzip")
        e_dataset.attrs["columns"] = [
            "Kinetic Energy (J)",
            "Strain Energy (J)",
            "Viscous Damping (J)",
            "Contact Potential (J)",
            "Total Energy (J)",
        ]
