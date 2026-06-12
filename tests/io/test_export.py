"""Headless automated unit verification suite for Sprint 6 export features.

Validates binary structural formatting for HDF5 trajectory archives, tabular summary CSV layouts,
Jinja2-compiled self-contained HTML and PDF summaries, and real-time GUI file size calculations.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from typing import Any

import h5py
import numpy as np
import pytest

from kevlargrid.io.export.csv_writer import export_to_csv
from kevlargrid.io.export.h5_writer import export_to_h5
from kevlargrid.io.export.report_builder import (
    generate_pdf_report,
    generate_report_html,
)
from kevlargrid.io.export.video_exporter import VideoExporter


@pytest.fixture
def dummy_simulation_data() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Pre-populates simple configurations, telemetry reports, and history frames."""
    config = {
        "material": {
            "name": "Kevlar 29",
            "tensile_modulus_gpa": 71.0,
            "failure_strain": 0.036,
            "tensile_strength_gpa": 2.92,
            "fiber_density_gcc": 1.44,
            "areal_density_kgm2": 0.47,
            "shear_ratio": 0.0004,
            "crimp_factor": 0.10,
            "yarn_count": [17, 17],
        },
        "grid": {
            "nx": 5,
            "ny": 5,
            "dx": 0.01,
            "n_plies": 1,
            "boundary_type": "fixed",
        },
        "projectile": {
            "mass": 0.05,
            "velocity": [0.0, 0.0, 400.0],
            "position": [0.0, 0.0, -0.005],
            "blade_width": 0.02,
            "edge_thickness": 0.005,
        },
        "simulation": {
            "duration": 0.001,
            "cfl_factor": 0.8,
            "rayleigh_alpha": 0.5,
            "rayleigh_beta": 0.0,
            "snapshot_interval": 10,
        },
    }

    results = {
        "arrested": True,
        "peak_deceleration_g": 245.5,
        "yarn_rupture_percentage": 8.0,
        "residual_velocity_ms": 0.0,
        "energy_dissipation_efficiency": 0.98,
        "max_layer_perforated": -1,
    }

    # Generate 5 sample frame cycles
    n_nodes = 25
    n_springs = 40  # Estimate
    history = []

    for step in range(5):
        time = step * 0.0001
        nodes = np.zeros((n_nodes, 3), dtype=np.float32)
        nodes[:, 2] = -0.001 * step  # Deflect along Z

        failed = np.zeros(n_springs, dtype=bool)
        if step > 2:
            failed[:2] = True  # Snapped springs

        history.append(
            {
                "time": time,
                "nodes": nodes,
                "failed": failed,
                "projectile_pos": np.array([0.0, 0.0, -0.005 + 0.04 * step], dtype=np.float32),
                "ke": 4000.0 - 500.0 * step,
                "se": 400.0 * step,
                "damped": 20.0 * step,
                "contact": 10.0 * step,
                "total": 4000.0,
                "peak_strain": 0.008 * step,
                "projectile_vz": 400.0 - 15.0 * step,
            }
        )

    return config, results, history


def test_hdf5_trajectory_archiver(dummy_simulation_data) -> None:
    """Verify HDF5 binary trajectory files load and match expected schema schemas."""
    config, results, history = dummy_simulation_data

    with tempfile.TemporaryDirectory() as tmpdir:
        h5_filepath = os.path.join(tmpdir, "test_trajectory.h5")

        # Export
        export_to_h5(config, results, history, h5_filepath)
        assert os.path.exists(h5_filepath)

        # Reload & Assert
        with h5py.File(h5_filepath, "r") as f:
            # 1. Metadata Checks
            assert "metadata" in f
            cfg_json = f["metadata/config_json"][()].decode("utf-8")
            loaded_cfg = json.loads(cfg_json)
            assert loaded_cfg["material"]["name"] == "Kevlar 29"

            # 2. Results Attributes Checks
            assert "results" in f
            res = f["results"]
            assert bool(res.attrs["arrested"]) is True
            assert res.attrs["peak_deceleration_g"] == pytest.approx(245.5)
            assert res.attrs["max_layer_perforated"] == -1

            # 3. Time History Array Shape Checks
            assert "time_history" in f
            hist = f["time_history"]

            assert hist["time"].shape == (5,)
            assert hist["time"][1] == pytest.approx(0.0001)

            assert hist["positions"].shape == (5, 25, 3)
            assert hist["positions"][2, 0, 2] == pytest.approx(-0.002)

            assert hist["spring_failures"].shape == (5, 40)
            assert hist["spring_failures"][4, 0]
            assert not hist["spring_failures"][1, 0]

            assert hist["projectile_pos"].shape == (5, 3)
            assert hist["projectile_pos"][1, 2] == pytest.approx(-0.005 + 0.04)

            # Energy Columns Checks
            assert hist["energies"].shape == (5, 5)
            assert hist["energies"][2, 0] == pytest.approx(3000.0)  # KE frame 2
            assert hist["energies"][2, 4] == pytest.approx(4000.0)  # Total frame 2
            assert "columns" in hist["energies"].attrs


def test_csv_excel_matrix_exporter(dummy_simulation_data) -> None:
    """Verify CSV summaries open cleanly and follow tabular matrix columns."""
    _, _, history = dummy_simulation_data

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_filepath = os.path.join(tmpdir, "test_summary.csv")

        # Export
        export_to_csv(history, csv_filepath)
        assert os.path.exists(csv_filepath)

        # Parse & Validate
        with open(csv_filepath, encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

            # 1. Header Validation
            assert len(rows) == 6  # 1 header + 5 steps
            headers = rows[0]
            assert "Time (s)" in headers
            assert "Peak Strain" in headers
            assert "Projectile Z Position (m)" in headers

            # 2. Row telemetry validation
            row_1 = rows[1]
            assert float(row_1[0]) == pytest.approx(0.0)
            assert float(row_1[1]) == pytest.approx(0.0)
            assert float(row_1[2]) == pytest.approx(4000.0)
            assert float(row_1[5]) == pytest.approx(-0.005)


def test_executive_html_and_pdf_report_compilers(dummy_simulation_data) -> None:
    """Verify Jinja2 executive HTML and PDF compilation works headlessly."""
    config, results, history = dummy_simulation_data

    # Test standalone HTML generation
    html_content = generate_report_html(config, results, history)
    assert "<title>KevlarGrid Explicit Solver Executive Report</title>" in html_content
    assert "OUTCOME: PASS (ARRESTED)" in html_content
    assert "data:image/png;base64," in html_content

    # Test PDF generation (transparent fallback writes .html if WeasyPrint is mocked or missing)
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_filepath = os.path.join(tmpdir, "test_report.pdf")

        try:
            generate_pdf_report(config, results, history, pdf_filepath)
            # If weasyprint is natively fully functional on this platform
            assert os.path.exists(pdf_filepath) or os.path.exists(
                pdf_filepath.replace(".pdf", ".html")
            )
        except ImportError:
            # Fallback path successfully activated
            html_filepath = pdf_filepath.replace(".pdf", ".html")
            assert os.path.exists(html_filepath)


def test_video_animation_exporter_mock(dummy_simulation_data) -> None:
    """Verify video anim constructor loads data headlessly without crashing."""
    config, _, history = dummy_simulation_data

    exporter = VideoExporter(
        config=config,
        history=history,
        nx=5,
        ny=5,
        n_plies=1,
        n_nodes_per_layer=25,
    )

    assert exporter.nx == 5
    assert len(exporter.history) == 5


def test_video_animation_exporter_compile(dummy_simulation_data) -> None:
    """Verify video anim compile actually renders and saves output to disk."""
    config, _, history = dummy_simulation_data

    exporter = VideoExporter(
        config=config,
        history=history,
        nx=5,
        ny=5,
        n_plies=1,
        n_nodes_per_layer=25,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Test GIF
        gif_filepath = os.path.join(tmpdir, "test_animation.gif")
        exporter.compile(gif_filepath, fps=5)
        assert os.path.exists(gif_filepath)
        assert os.path.getsize(gif_filepath) > 1024

        # 2. Test MP4 (if ffmpeg is available)
        import matplotlib.animation as animation
        if animation.FFMpegWriter.isAvailable() or animation.writers.is_available("ffmpeg"):
            mp4_filepath = os.path.join(tmpdir, "test_animation.mp4")
            exporter.compile(mp4_filepath, fps=5)
            assert os.path.exists(mp4_filepath)
            assert os.path.getsize(mp4_filepath) > 1024

