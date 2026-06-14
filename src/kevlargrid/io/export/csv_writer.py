"""Comma-Separated Excel Summary Matrix Exporter.

Writes simulation timestep history data as a tabular CSV file
for clean loading and analysis in Excel or spreadsheet viewers.
"""

from __future__ import annotations

import csv
from typing import Any


def export_to_csv(history: list[dict[str, Any]], filepath: str) -> None:
    """Export time-series summary telemetry data to a tabular CSV format.

    Parameters
    ----------
    history : list of dict
        The step-by-step frame history of solver coordinates and energies.
    filepath : str
        Output file path.
    """
    if not history:
        raise ValueError("Cannot export empty simulation history.")

    headers = [
        "Time (s)",
        "Peak Strain",
        "Kinetic Energy (J)",
        "Strain Energy (J)",
        "Damping Energy (J)",
        "Projectile Z Position (m)",
        "Projectile Velocity (m/s)",
        "Rupture Energy (J)",
        "Clamping Energy (J)",
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for frame in history:
            time_val = frame["time"]
            peak_strain = frame.get("peak_strain", 0.0)
            ke = frame.get("ke", 0.0)
            se = frame.get("se", 0.0)
            damped = frame.get("damped", 0.0)
            rupture = frame.get("failure_dissipated", 0.0)
            clamp = frame.get("clamp_dissipated", 0.0)

            # Projectile details
            proj_z = frame["projectile_pos"][2]
            # Estimate velocity from projectile state if stored, otherwise 0.0
            proj_vz = frame.get("projectile_vz", 0.0)

            writer.writerow(
                [
                    f"{time_val:.8e}",
                    f"{peak_strain:.6f}",
                    f"{ke:.4f}",
                    f"{se:.4f}",
                    f"{damped:.4f}",
                    f"{proj_z:.6f}",
                    f"{proj_vz:.4f}",
                    f"{rupture:.4f}",
                    f"{clamp:.4f}",
                ]
            )
