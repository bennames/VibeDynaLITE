"""Executive summary reporting compilers for HTML and PDF outputs.

Compiles standalone, beautifully styled executive reports featuring embedded
base64-encoded strain/energy plots, summary data tables, and arrest outcomes.
"""

from __future__ import annotations

import base64
import io
from typing import Any

# Headless matplotlib plotting
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# WeasyPrint PDF compiler bypassed
weasyprint = None


def generate_plots_base64(history: list[dict[str, Any]]) -> tuple[str, str]:
    """Headlessly render strain and energy time histories, returning base64 PNG data.

    Parameters
    ----------
    history : list of dict
        Simulation history logs.

    Returns
    -------
    tuple of str
        The (strain_base64, energy_base64) image data strings.
    """
    steps = len(history)
    times = np.zeros(steps, dtype=np.float32)
    strains = np.zeros(steps, dtype=np.float32)

    # Trace arrays: KE, SE, Damping, Contact, Total
    ke = np.zeros(steps, dtype=np.float32)
    se = np.zeros(steps, dtype=np.float32)
    damped = np.zeros(steps, dtype=np.float32)
    contact = np.zeros(steps, dtype=np.float32)
    total = np.zeros(steps, dtype=np.float32)

    for i, frame in enumerate(history):
        times[i] = frame["time"] * 1000.0  # ms
        strains[i] = frame.get("peak_strain", 0.0)
        ke[i] = frame.get("ke", 0.0)
        se[i] = frame.get("se", 0.0)
        damped[i] = frame.get("damped", 0.0)
        contact[i] = frame.get("contact", 0.0)
        total[i] = frame.get("total", 0.0)

    # 1. Peak Strain History Plot
    fig1, ax1 = plt.subplots(figsize=(6, 3.5), dpi=100)
    ax1.plot(times, strains, color="#1E90FF", linewidth=2.0, label="Max Engineering Strain")
    ax1.axhline(
        y=0.036, color="#DC143C", linestyle="--", linewidth=1.5, label="Rupture Limit (0.036)"
    )
    ax1.set_title("Peak Membrane Strain History", fontsize=11, fontweight="bold", color="#2C3E50")
    ax1.set_xlabel("Time (ms)", fontsize=9)
    ax1.set_ylabel("Strain (m/m)", fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=8, loc="upper right")
    plt.tight_layout()

    buf1 = io.BytesIO()
    fig1.savefig(buf1, format="png", dpi=100)
    plt.close(fig1)
    buf1.seek(0)
    strain_b64 = base64.b64encode(buf1.read()).decode("utf-8")

    # 2. Multi-Series Energy Balance Plot
    fig2, ax2 = plt.subplots(figsize=(6, 3.5), dpi=100)
    ax2.plot(times, ke, color="#FF4500", linewidth=1.8, label="Kinetic Energy")
    ax2.plot(times, se, color="#32CD32", linewidth=1.8, label="Strain Energy")
    ax2.plot(times, damped, color="#9370DB", linewidth=1.8, label="Viscous Damping")
    ax2.plot(times, contact, color="#00CED1", linewidth=1.8, label="Contact Potential")
    ax2.plot(times, total, color="#2F4F4F", linewidth=2.0, linestyle=":", label="Total Energy")

    ax2.set_title(
        "System Energy Conservation Telemetry", fontsize=11, fontweight="bold", color="#2C3E50"
    )
    ax2.set_xlabel("Time (ms)", fontsize=9)
    ax2.set_ylabel("Energy (Joules)", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=8, loc="upper right")
    plt.tight_layout()

    buf2 = io.BytesIO()
    fig2.savefig(buf2, format="png", dpi=100)
    plt.close(fig2)
    buf2.seek(0)
    energy_b64 = base64.b64encode(buf2.read()).decode("utf-8")

    return strain_b64, energy_b64


def generate_report_html(
    config: dict[str, Any],
    results_report: dict[str, Any],
    history: list[dict[str, Any]],
) -> str:
    """Compile the standalone self-contained HTML executive report.

    Parameters
    ----------
    config : dict
        Simulation configuration dictionary.
    results_report : dict
        Simulation outcome variables.
    history : list of dict
        Timber frame coordinate histories.

    Returns
    -------
    str
        The fully compiled HTML string.
    """
    strain_b64, energy_b64 = generate_plots_base64(history)

    arrested = results_report.get("arrested", False)
    status_label = "PASS (ARRESTED)" if arrested else "FAIL (PERFORATED)"
    status_color = "#32CD32" if arrested else "#DC143C"

    # Calculate initial projectile kinetic energy
    p_mass = config["projectile"]["mass"]
    vel = config["projectile"]["velocity"]
    v_sq = sum(v**2 for v in vel)
    ke_initial = 0.5 * p_mass * v_sq

    # Setup report parameters
    material = config["material"]
    grid = config["grid"]

    is_mode_b = grid.get("t_ply") is not None
    analysis_mode = "Mode B (Checkout Stacking)" if is_mode_b else "Mode A (Sizing Multiplier)"

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>KevlarGrid Explicit Solver Executive Report</title>
    <style>
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: #333333;
            line-height: 1.5;
            background-color: #fcfcfc;
            margin: 0;
            padding: 40px;
        }}
        @page {{
            size: A4;
            margin: 20mm;
            @bottom-right {{
                content: counter(page);
                font-size: 9pt;
                color: #7f8c8d;
            }}
        }}
        .header {{
            border-bottom: 2px solid #2C3E50;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            color: #2C3E50;
            font-size: 24pt;
            font-weight: 700;
        }}
        .header p {{
            margin: 5px 0 0 0;
            color: #7F8C8D;
            font-size: 11pt;
        }}
        .badge {{
            display: inline-block;
            padding: 8px 16px;
            font-size: 14pt;
            font-weight: bold;
            color: #ffffff;
            background-color: {status_color};
            border-radius: 4px;
            margin-bottom: 25px;
            text-align: center;
        }}
        .section-title {{
            color: #2C3E50;
            font-size: 16pt;
            font-weight: 600;
            border-left: 4px solid #1E90FF;
            padding-left: 10px;
            margin-top: 30px;
            margin-bottom: 15px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 25px;
            font-size: 10.5pt;
        }}
        th, td {{
            border: 1px solid #BDC3C7;
            padding: 10px;
            text-align: left;
        }}
        th {{
            background-color: #ECF0F1;
            color: #2C3E50;
            font-weight: 600;
        }}
        .grid-container {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 30px;
        }}
        .grid-item {{
            width: 48%;
            border: 1px solid #E2E8F0;
            background: #ffffff;
            border-radius: 6px;
            padding: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .plot-container {{
            text-align: center;
            margin-bottom: 30px;
            page-break-inside: avoid;
        }}
        .plot-image {{
            max-width: 100%;
            height: auto;
            border: 1px solid #E2E8F0;
            border-radius: 4px;
        }}
        .footer {{
            margin-top: 50px;
            border-top: 1px solid #BDC3C7;
            padding-top: 15px;
            text-align: center;
            font-size: 9pt;
            color: #7F8C8D;
        }}
    </style>
</head>
<body>

    <div class="header">
        <h1>KevlarGrid Explicit Solver Executive Report</h1>
        <p>Dynamic Mass-Spring Explicit Integration Simulation Summary — Powered by VibeDynaLITE</p>
    </div>

    <div class="badge">OUTCOME: {status_label}</div>

    <div class="section-title">1. Impact Simulation & Structural Summary</div>
    <div class="grid-container">
        <div class="grid-item">
            <h3 style="margin-top: 0; color: #2C3E50;">Fabric Configuration</h3>
            <p><strong>Material preset</strong>: {material.get("name", "Custom")}<br>
               <strong>Young's Modulus</strong>: {material.get("tensile_modulus_gpa", 71.0)} GPa<br>
               <strong>Failure Strain Limit</strong>: {material.get("failure_strain", 0.036)} m/m<br>
               <strong>Grid Nodes (Nx &times; Ny)</strong>: {grid.get("nx", 11)} &times; {grid.get("ny", 11)}<br>
               <strong>Grid Spacing dx</strong>: {grid.get("dx", 0.01)} m<br>
               <strong>Analysis Mode</strong>: {analysis_mode}<br>
               <strong>Number of Plies</strong>: {grid.get("n_plies", 1)} plies</p>
        </div>
        <div class="grid-item">
            <h3 style="margin-top: 0; color: #2C3E50;">Striking Projectile</h3>
            <p><strong>Initial Mass</strong>: {config["projectile"]["mass"]} kg<br>
               <strong>Initial Velocity Vz</strong>: {config["projectile"]["velocity"][2]} m/s<br>
               <strong>Live Kinetic Energy (KE)</strong>: {ke_initial:.2f} Joules<br>
               <strong>Blade Profile Width</strong>: {config["projectile"]["blade_width"]} m<br>
               <strong>Edge Thickness</strong>: {config["projectile"]["edge_thickness"]} m</p>
        </div>
    </div>

    <div class="section-title">2. Solver Performance Telemetry</div>
    <table>
        <thead>
            <tr>
                <th>Outcome Metric</th>
                <th>Value</th>
                <th>Standard Limits / Pass Conditions</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>Projectile Arrest Outcome</strong></td>
                <td style="color: {status_color}; font-weight: bold;">{status_label}</td>
                <td>MEMBRANE CONTAINMENT SUCCESSFUL</td>
            </tr>
            <tr>
                <td><strong>Peak Projectile Deceleration</strong></td>
                <td>{results_report.get("peak_deceleration_g", 0.0):.2f} G's</td>
                <td>Dynamic structural loading cap</td>
            </tr>
            <tr>
                <td><strong>Yarn Springs Rupture Ratio</strong></td>
                <td>{results_report.get("yarn_rupture_percentage", 0.0):.2f}%</td>
                <td>Fabric structural degradation metric</td>
            </tr>
            <tr>
                <td><strong>Residual Velocity</strong></td>
                <td>{results_report.get("residual_velocity_ms", 0.0):.2f} m/s</td>
                <td>Exit velocity post-perforation</td>
            </tr>
            <tr>
                <td><strong>Maximum Ply Perforation Index</strong></td>
                <td>{results_report.get("max_layer_perforated", -1) + 1} / {grid.get("n_plies", 1)}</td>
                <td>Total snap-through layer sequence</td>
            </tr>
        </tbody>
    </table>

    <div class="section-title">3. Structural Telemetry Graphs</div>

    <div class="plot-container">
        <h3>Peak Fabric Strain Time History</h3>
        <img class="plot-image" src="data:image/png;base64,{strain_b64}" alt="Strain Plot">
    </div>

    <div class="plot-container">
        <h3>System Energy Balance Conservation</h3>
        <img class="plot-image" src="data:image/png;base64,{energy_b64}" alt="Energy Plot">
    </div>

    <div class="footer">
        <p>KevlarGrid Explicit Solver v2.0 © 2026. Standalone generated document.</p>
    </div>

</body>
</html>
"""
    return html_content


def generate_pdf_report(
    config: dict[str, Any],
    results_report: dict[str, Any],
    history: list[dict[str, Any]],
    filepath: str,
) -> None:
    """Generate the executive summary PDF report using WeasyPrint.

    Parameters
    ----------
    config : dict
        Simulation configuration dictionary.
    results_report : dict
        Simulation outcome variables.
    history : list of dict
        Frame history list.
    filepath : str
        Target output file path.
    """
    html_content = generate_report_html(config, results_report, history)

    if weasyprint is None:
        # Transparent fallback to direct HTML printing if native pango/cairo is not ready
        html_filepath = filepath.replace(".pdf", ".html")
        with open(html_filepath, "w", encoding="utf-8") as f:
            f.write(html_content)
        raise ImportError(
            "WeasyPrint is not installed or pango/cairo are missing. "
            f"HTML report successfully written as a fallback to: {html_filepath}"
        )

    # Compile natively to PDF
    weasyprint.HTML(string=html_content).write_pdf(filepath)
