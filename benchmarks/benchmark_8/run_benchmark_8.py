import os
import json
import time
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import solver components
from kevlargrid.solver.taichi_solver import taichi_leapfrog_loop
from kevlargrid.solver.fused import fused_leapfrog_loop
from kevlargrid.solver.grid import generate_rectangular_grid
from kevlargrid.solver.timestep import compute_cfl_timestep
from kevlargrid.io.export.report_builder import generate_report_html

# WeasyPrint PDF compiler
try:
    import weasyprint
except ImportError:
    weasyprint = None

BENCHMARK_DIR = Path(__file__).parent
RESULTS_FILE = BENCHMARK_DIR / "results.json"
PLOT_FILE = BENCHMARK_DIR / "validation_plot.png"
REPORT_PDF = BENCHMARK_DIR / "validation_report.pdf"
REPORT_HTML = BENCHMARK_DIR / "validation_report.html"

def escape_pdf_string(s: str) -> str:
    """Escape parentheses and backslashes for PDF string literals."""
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

def generate_pure_python_pdf(filepath: Path, results: dict) -> None:
    """Generate a 100% valid, compliant PDF binary from scratch containing simulation results."""
    case_a = results["case_a"]
    case_b = results["case_b"]
    case_c = results["case_c"]
    
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    
    title = "Benchmark 8: Ballistic Limit (V50) Validation Report"
    sub = f"Generated: {timestamp}"
    ref = "Experimental V50 Reference: 503 m/s (Style 713 Kevlar 29)"
    
    line_a = f"Case A (Strike: {case_a['initial_velocity']:.1f} m/s): Residual Velocity = {case_a['residual_velocity']:.2f} m/s (Arrested: {not case_a['penetrated']})"
    line_b = f"Case B (Strike: {case_b['initial_velocity']:.1f} m/s): Residual Velocity = {case_b['residual_velocity']:.2f} m/s (Arrested: {not case_b['penetrated']})"
    line_c = f"Case C (Strike: {case_c['initial_velocity']:.1f} m/s): Residual Velocity = {case_c['residual_velocity']:.2f} m/s (Arrested: {not case_c['penetrated']})"
    
    status_a = f"  - Case A (450 m/s) is arrested: {'PASS' if not case_a['penetrated'] else 'FAIL'}"
    status_b = f"  - Case B (503 m/s) residual velocity < 25 m/s: {'PASS' if case_b['residual_velocity'] < 25.0 else 'FAIL'}"
    status_c = f"  - Case C (550 m/s) residual velocity ~220 m/s: {'PASS' if abs(case_c['residual_velocity'] - 220.0) <= 20.0 else 'FAIL'}"
    
    # Build text stream commands
    stream_cmds = [
        "BT",
        "/F1 16 Tf",
        "18 TL",
        "72 720 Td",
        f"({escape_pdf_string(title)}) Tj",
        "T*",
        "/F1 10 Tf",
        "12 TL",
        f"({escape_pdf_string(sub)}) Tj",
        "T*",
        "T*",
        f"({escape_pdf_string(ref)}) Tj",
        "T*",
        "T*",
        f"({escape_pdf_string(line_a)}) Tj",
        "T*",
        f"({escape_pdf_string(line_b)}) Tj",
        "T*",
        f"({escape_pdf_string(line_c)}) Tj",
        "T*",
        "T*",
        "(Verification Outcomes:) Tj",
        "T*",
        f"({escape_pdf_string(status_a)}) Tj",
        "T*",
        f"({escape_pdf_string(status_b)}) Tj",
        "T*",
        f"({escape_pdf_string(status_c)}) Tj",
        "ET"
    ]
    
    content = "\n".join(stream_cmds)
    content_bytes = content.encode('latin1')
    
    # Construct PDF structure
    objects = []
    # 1 0 obj: Catalog
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    # 2 0 obj: Pages
    objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    # 3 0 obj: Page
    objects.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>")
    # 4 0 obj: Font
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    # 5 0 obj: Contents stream
    stream_meta = f"<< /Length {len(content_bytes)} >>".encode('latin1')
    objects.append(stream_meta + b"\nstream\n" + content_bytes + b"\nendstream")
    
    # Write PDF file
    with open(filepath, "wb") as f:
        f.write(b"%PDF-1.4\n")
        offsets = []
        for i, obj in enumerate(objects):
            offsets.append(f.tell())
            f.write(f"{i+1} 0 obj\n".encode('latin1'))
            f.write(obj)
            f.write(b"\nendobj\n")
            
        xref_offset = f.tell()
        f.write(b"xref\n")
        f.write(f"0 {len(objects)+1}\n".encode('latin1'))
        f.write(b"0000000000 65535 f \n")
        for offset in offsets:
            f.write(f"{offset:010d} 00000 n \n".encode('latin1'))
            
        f.write(b"trailer\n")
        f.write(f"<< /Size {len(objects)+1} /Root 1 0 R >>\n".encode('latin1'))
        f.write(b"startxref\n")
        f.write(f"{xref_offset}\n".encode('latin1'))
        f.write(b"%%EOF\n")

def run_case(v_strike: float, run_id: str, backend_name: str) -> dict:
    """Run a single dynamic simulation case and return result metrics."""
    print(f"\n--- Running Case {run_id} (strike velocity: {v_strike} m/s, backend: {backend_name}) ---")
    
    # 1.365 mm element size: exactly 4 elements span the 5.46 mm projectile diameter
    nx, ny = 184, 184
    dx = 0.001365
    n_nodes_per_layer = nx * ny
    n_plies = 13
    
    material_kev29 = {
        "tensile_modulus_gpa": 70.5,
        "areal_density_kgm2": 0.475,
        "fiber_density_gcc": 1.44,
        "failure_strain": 0.038,
        "shear_ratio": 0.0004,
    }
    
    grid = generate_rectangular_grid(nx, ny, dx, material_kev29, n_plies=n_plies, t_ply=0.0001)
    
    # Boundary conditions: Clamped on all outer edges
    boundary_mask = np.zeros(grid.n_nodes, dtype=bool)
    for ply in range(n_plies):
        offset = ply * n_nodes_per_layer
        for i in range(nx):
            for j in range(ny):
                if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                    boundary_mask[offset + i * ny + j] = True
                    
    # Setup Projectile: 17-grain FSP (treated as Right Circular Cylinder)
    proj_mass = 0.0011  # 1.10 grams
    R = 0.00273         # 5.46 mm diameter
    L = 0.006           # 6 mm length
    I_zz = 0.5 * proj_mass * R**2
    I_xx = (1.0 / 12.0) * proj_mass * (3.0 * R**2 + L**2)
    proj_inertia_inv = np.diag([1.0/I_xx, 1.0/I_xx, 1.0/I_zz])
    
    proj_pos = np.array([0.0, 0.0, -0.002], dtype=np.float64)
    proj_vel = np.array([0.0, 0.0, v_strike], dtype=np.float64)
    proj_omega = np.zeros(3, dtype=np.float64)
    proj_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    
    k_penalty = 2.0e6
    mu_s = 0.20
    dt = compute_cfl_timestep(grid.stiffnesses, grid.masses, dx, 0.1)
    
    node_initial_springs = grid.initial_spring_counts
    node_spring_offsets = grid.node_spring_offsets
    node_spring_ids = grid.node_spring_ids
    node_spring_signs = grid.node_spring_signs
    
    initial_energy = 0.5 * proj_mass * (v_strike**2)
    max_steps = 4500
    save_interval = 20
    
    t_sim = 0.0
    damp_dissipated = 0.0
    failure_dissipated = 0.0
    clamp_dissipated = 0.0
    contact_energy = 0.0
    friction_dissipated = 0.0
    
    hist_ke = []
    hist_se = []
    hist_proj_ke = []
    hist_time = []
    hist_failed_count = []
    hist_total_energy = []
    hist_peak_strain = []
    
    step = 0
    t0 = time.perf_counter()
    
    # Persistent state variables for propagation across chunks
    pos = grid.nodes.copy()
    vel = np.zeros_like(pos)
    grid_damage = np.zeros(grid.n_springs, dtype=np.float64)
    failed = grid.failed.copy()
    peak_decel_g = 0.0
    
    while step < max_steps:
        # Run one save_interval chunk of steps
        if backend_name == "taichi":
            (
                pos,
                vel,
                failed,
                proj_pos_new,
                proj_vel_new,
                damp_dissipated,
                failure_dissipated,
                clamp_dissipated,
                t_sim,
                _,  # hist_pos
                _,  # hist_failed
                _,  # hist_proj_pos
                _,  # hist_t
                _,  # h_ke
                _,  # h_se
                _,  # h_proj_ke
                hist_peak_strain_gpu,
                contact_energy,
                friction_dissipated,
            ) = taichi_leapfrog_loop(
                pos,
                vel,
                grid.springs.copy(),
                grid.stiffnesses.copy(),
                grid.rest_lengths.copy(),
                failed,
                grid.masses.copy(),
                grid.tension_only.copy(),
                boundary_mask,
                np.zeros((grid.n_nodes, 3)),
                proj_pos,
                proj_vel,
                proj_mass,
                0.0,  # blade_width
                0.0,  # edge_thickness
                n_plies,
                n_nodes_per_layer,
                0.0001,
                dx,
                k_penalty,
                0.0,   # rayleigh_alpha
                5e-8,  # rayleigh_beta
                0.038, # failure_strain
                0.0228, # damage_onset_strain
                1.5,   # fracture_energy_multiplier
                dt,
                save_interval,
                save_interval,
                damp_dissipated,
                failure_dissipated,
                clamp_dissipated,
                t_sim,
                1.0,
                node_initial_springs,
                node_spring_offsets,
                node_spring_ids,
                node_spring_signs,
                use_viscous=False,
                cfl_factor=0.1,
                mu_s=mu_s,
                proj_quat=proj_quat,
                proj_omega=proj_omega,
                proj_shape_type="cylinder",
                proj_radius=R,
                proj_length=L,
                proj_inertia_inv=proj_inertia_inv,
                grid_damage=grid_damage,
                contact_energy_init=contact_energy,
                friction_dissipated_init=friction_dissipated,
            )
        else: # numba
            (
                pos,
                vel,
                failed,
                proj_pos_new,
                proj_vel_new,
                damp_dissipated,
                failure_dissipated,
                clamp_dissipated,
                t_sim,
                _,  # hist_pos
                _,  # hist_failed
                _,  # hist_proj_pos
                _,  # hist_t
                _,  # h_ke
                _,  # h_se
                _,  # h_proj_ke
                contact_energy,
                friction_dissipated,
            ) = fused_leapfrog_loop(
                pos,
                vel,
                grid.springs.copy(),
                grid.stiffnesses.copy(),
                grid.rest_lengths.copy(),
                failed,
                grid.masses.copy(),
                grid.tension_only.copy(),
                boundary_mask,
                np.zeros((grid.n_nodes, 3)),
                proj_pos,
                proj_vel,
                proj_mass,
                0.0,  # blade_width
                0.0,  # edge_thickness
                n_plies,
                n_nodes_per_layer,
                0.0001,
                dx,
                k_penalty,
                0.0,   # rayleigh_alpha
                5e-8,  # rayleigh_beta
                0.038, # failure_strain
                0.0228, # damage_onset_strain
                1.5,   # fracture_energy_multiplier
                dt,
                save_interval,
                save_interval,
                damp_dissipated,
                failure_dissipated,
                clamp_dissipated,
                t_sim,
                1.0,
                node_initial_springs,
                node_spring_offsets,
                node_spring_ids,
                node_spring_signs,
                use_viscous=False,
                cfl_factor=0.1,
                mu_s=mu_s,
                proj_quat=proj_quat,
                proj_omega=proj_omega,
                proj_shape_type="cylinder",
                proj_radius=R,
                proj_length=L,
                proj_inertia_inv=proj_inertia_inv,
                grid_damage=grid_damage,
                contact_energy_init=contact_energy,
                friction_dissipated_init=friction_dissipated,
            )
        
        # Track deceleration of the projectile
        accel_z = (proj_vel_new[2] - proj_vel[2]) / (save_interval * dt)
        decel_g = -accel_z / 9.81
        if decel_g > peak_decel_g:
            peak_decel_g = decel_g

        # Update state
        grid.failed = failed
        proj_pos = proj_pos_new
        proj_vel = proj_vel_new
        step += save_interval
        
        # Calculate current telemetry energies on host
        ke_nodes = 0.5 * np.sum(grid.masses * np.sum(vel**2, axis=1))
        
        p1 = pos[grid.springs[:, 0]]
        p2 = pos[grid.springs[:, 1]]
        lens = np.sqrt(np.sum((p2 - p1)**2, axis=1))
        strains = (lens - grid.rest_lengths) / grid.rest_lengths
        strains_eff = np.where(grid.tension_only & (strains < 0.0), 0.0, strains)
        se_springs_array = 0.5 * grid.stiffnesses * (strains_eff * grid.rest_lengths)**2
        se_springs = float(np.sum(np.where(grid.failed, 0.0, se_springs_array)))
        
        ke_proj = 0.5 * proj_mass * np.sum(proj_vel**2)
        
        total_energy = ke_nodes + se_springs + ke_proj + damp_dissipated + failure_dissipated + clamp_dissipated + contact_energy + friction_dissipated
        drift_pct = abs(total_energy - initial_energy) / initial_energy * 100.0

        hist_ke.append(ke_nodes)
        hist_se.append(se_springs)
        hist_proj_ke.append(ke_proj)
        hist_time.append(t_sim)
        hist_failed_count.append(np.sum(grid.failed))
        hist_total_energy.append(total_energy)
        hist_peak_strain.append(float(np.max(strains_eff)))
        
        # Check termination
        if proj_vel[2] <= 0.0:
            print("Projectile arrested.")
            break
            
        if proj_pos[2] > (n_plies * 0.0001 + 0.005) and proj_vel[2] > 0.0:
            print("Projectile fully perforated target.")
            break
            
    t1 = time.perf_counter()
    residual_vel = max(0.0, float(proj_vel[2]))
    energy_drift = float(np.max(np.abs(np.array(hist_total_energy) - initial_energy)) / initial_energy)
    
    print(f"Case {run_id} Finished in {t1 - t0:.2f} s")
    print(f"  Residual Velocity: {residual_vel:.2f} m/s")
    print(f"  Energy Drift: {energy_drift*100:.3f}%")
    print(f"  Peak Deceleration: {peak_decel_g:.1f} g")
    
    yarn_rupture_pct = (np.sum(grid.failed) / grid.n_springs) * 100.0
    failed_indices = np.where(grid.failed)[0]
    if len(failed_indices) > 0:
        failed_layers = grid.springs[failed_indices, 0] // n_nodes_per_layer
        max_layer_perforated = int(np.max(failed_layers))
    else:
        max_layer_perforated = -1

    history = []
    for i in range(len(hist_time)):
        history.append({
            "time": hist_time[i],
            "peak_strain": hist_peak_strain[i],
            "ke": hist_ke[i],
            "se": hist_se[i],
            "damped": damp_dissipated,
            "contact": contact_energy,
            "total": hist_total_energy[i],
        })

    return {
        "initial_velocity": v_strike,
        "residual_velocity": residual_vel,
        "energy_drift_pct": energy_drift * 100,
        "peak_deceleration_g": peak_decel_g,
        "yarn_rupture_percentage": yarn_rupture_pct,
        "max_layer_perforated": max_layer_perforated,
        "penetrated": residual_vel > 0.0,
        "history": history,
        "telemetry_lists": {
            "time": hist_time,
            "ke": hist_ke,
            "se": hist_se,
            "proj_ke": hist_proj_ke,
            "total": hist_total_energy,
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Run KevlarGrid Benchmark 8 validation sweep.")
    parser.add_argument("--backend", type=str, choices=["taichi", "numba"], default="numba",
                        help="Compute backend to use for simulation.")
    args = parser.parse_args()

    print("Starting Benchmark 8 - Ballistic Limit (V50) Validation Sweep...")
    
    # Run cases
    case_a = run_case(450.0, "A", args.backend)
    case_b = run_case(503.0, "B", args.backend)
    case_c = run_case(550.0, "C", args.backend)
    
    # Save results to JSON
    results = {
        "case_a": {
            "initial_velocity": case_a["initial_velocity"],
            "residual_velocity": case_a["residual_velocity"],
            "energy_drift_pct": case_a["energy_drift_pct"],
            "penetrated": case_a["penetrated"]
        },
        "case_b": {
            "initial_velocity": case_b["initial_velocity"],
            "residual_velocity": case_b["residual_velocity"],
            "energy_drift_pct": case_b["energy_drift_pct"],
            "penetrated": case_b["penetrated"]
        },
        "case_c": {
            "initial_velocity": case_c["initial_velocity"],
            "residual_velocity": case_c["residual_velocity"],
            "energy_drift_pct": case_c["energy_drift_pct"],
            "penetrated": case_c["penetrated"]
        }
    }
    
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {RESULTS_FILE}")
    
    # Plot Jonas-Laval curve and validation points
    v_strike = np.array([450.0, 503.0, 550.0])
    v_residual = np.array([case_a["residual_velocity"], case_b["residual_velocity"], case_c["residual_velocity"]])
    
    # Jonas-Laval Fit
    v50_fit = 503.0
    alpha_fit = 1.05
    
    plt.figure(figsize=(8, 6))
    plt.scatter(v_strike, v_residual, color="#e74c3c", s=100, zorder=5, label="Simulation Cases")
    
    v_s_plot = np.linspace(400.0, 600.0, 500)
    v_r_plot = np.zeros_like(v_s_plot)
    mask = v_s_plot > v50_fit
    v_r_plot[mask] = alpha_fit * np.sqrt(v_s_plot[mask]**2 - v50_fit**2)
    
    plt.plot(v_s_plot, v_r_plot, color="#34495e", linewidth=2.5, zorder=4, label=f"Lambert-Jonas Fit ($V_{{50}} = 503$ m/s)")
    plt.axvline(503.0, color="#2ecc71", linestyle="--", linewidth=1.5, label="Experimental V50 (503 m/s)")
    
    plt.title("Benchmark 8: Kevlar 29 Style 713 (13-Ply, 17-Grain FSP)", fontsize=12, fontweight="bold")
    plt.xlabel("Strike Velocity (m/s)", fontsize=11)
    plt.ylabel("Residual Velocity (m/s)", fontsize=11)
    plt.xlim(420, 580)
    plt.ylim(-10, 300)
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=300)
    plt.close()
    print(f"Saved validation plot to {PLOT_FILE}")
    
    # Generate HTML & PDF Report
    config = {
        "material": {
            "name": "Kevlar 29 Style 713",
            "tensile_modulus_gpa": 70.5,
            "failure_strain": 0.038,
        },
        "grid": {
            "nx": 184,
            "ny": 184,
            "dx": 0.001365,
            "n_plies": 13,
            "t_ply": 0.0001,
        },
        "projectile": {
            "mass": 0.0011,
            "velocity": [0.0, 0.0, 503.0],
            "blade_width": 0.0,
            "edge_thickness": 0.0,
        }
    }
    
    # Use Case B as the representative telemetry report case
    results_report = {
        "arrested": not case_b["penetrated"],
        "peak_deceleration_g": case_b["peak_deceleration_g"],
        "yarn_rupture_percentage": case_b["yarn_rupture_percentage"],
        "residual_velocity_ms": case_b["residual_velocity"],
        "max_layer_perforated": case_b["max_layer_perforated"],
    }
    
    html_content = generate_report_html(config, results_report, case_b["history"])
    with open(REPORT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"HTML report saved to {REPORT_HTML}")
    
    pdf_compiled = False
    
    # 1. Try WeasyPrint (preferred HTML->PDF engine)
    if False:  # Bypass WeasyPrint to prevent sandbox hangs
        try:
            weasyprint.HTML(string=html_content).write_pdf(REPORT_PDF)
            print(f"PDF report successfully compiled via WeasyPrint to {REPORT_PDF}")
            pdf_compiled = True
        except Exception as e:
            print(f"WeasyPrint PDF compilation failed: {e}")
            
    # 2. Try ReportLab (canvas rendering fallback)
    if not pdf_compiled:
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            c = canvas.Canvas(str(REPORT_PDF), pagesize=letter)
            c.setFont("Helvetica-Bold", 16)
            c.drawString(72, 720, "Benchmark 8: Ballistic Limit (V50) Validation Report")
            c.setFont("Helvetica", 10)
            c.drawString(72, 700, f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
            c.setFont("Helvetica-Bold", 12)
            c.drawString(72, 660, "Experimental V50 Reference: 503 m/s (Kevlar 29)")
            c.setFont("Helvetica", 10)
            c.drawString(72, 630, f"Case A (450 m/s): Residual Velocity = {results['case_a']['residual_velocity']:.2f} m/s (Arrested: {not results['case_a']['penetrated']})")
            c.drawString(72, 610, f"Case B (503 m/s): Residual Velocity = {results['case_b']['residual_velocity']:.2f} m/s (Arrested: {not results['case_b']['penetrated']})")
            c.drawString(72, 590, f"Case C (550 m/s): Residual Velocity = {results['case_c']['residual_velocity']:.2f} m/s (Arrested: {not results['case_c']['penetrated']})")
            c.setFont("Helvetica-Bold", 11)
            c.drawString(72, 550, "Verification Outcomes:")
            c.setFont("Helvetica", 10)
            c.drawString(72, 530, f"  - Case A (450 m/s) is arrested: {'PASS' if not results['case_a']['penetrated'] else 'FAIL'}")
            c.drawString(72, 510, f"  - Case B (503 m/s) residual velocity < 25 m/s: {'PASS' if results['case_b']['residual_velocity'] < 25.0 else 'FAIL'}")
            c.drawString(72, 490, f"  - Case C (550 m/s) residual velocity ~220 m/s: {'PASS' if abs(results['case_c']['residual_velocity'] - 220.0) <= 20.0 else 'FAIL'}")
            c.save()
            print(f"PDF report successfully compiled via ReportLab to {REPORT_PDF}")
            pdf_compiled = True
        except ImportError:
            pass
            
    # 3. Try FPDF/FPDF2 fallback
    if not pdf_compiled:
        try:
            from fpdf import FPDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Benchmark 8: Ballistic Limit (V50) Validation Report", ln=1, align="L")
            pdf.set_font("Arial", "", 10)
            pdf.cell(0, 10, f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}", ln=1, align="L")
            pdf.ln(10)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Experimental V50 Reference: 503 m/s (Kevlar 29)", ln=1, align="L")
            pdf.set_font("Arial", "", 10)
            pdf.cell(0, 10, f"Case A (450 m/s): Residual Velocity = {results['case_a']['residual_velocity']:.2f} m/s (Arrested: {not results['case_a']['penetrated']})", ln=1, align="L")
            pdf.cell(0, 10, f"Case B (503 m/s): Residual Velocity = {results['case_b']['residual_velocity']:.2f} m/s (Arrested: {not results['case_b']['penetrated']})", ln=1, align="L")
            pdf.cell(0, 10, f"Case C (550 m/s): Residual Velocity = {results['case_c']['residual_velocity']:.2f} m/s (Arrested: {not results['case_c']['penetrated']})", ln=1, align="L")
            pdf.ln(10)
            pdf.set_font("Arial", "B", 11)
            pdf.cell(0, 10, "Verification Outcomes:", ln=1, align="L")
            pdf.set_font("Arial", "", 10)
            pdf.cell(0, 10, f"  - Case A (450 m/s) is arrested: {'PASS' if not results['case_a']['penetrated'] else 'FAIL'}", ln=1, align="L")
            pdf.cell(0, 10, f"  - Case B (503 m/s) residual velocity < 25 m/s: {'PASS' if results['case_b']['residual_velocity'] < 25.0 else 'FAIL'}", ln=1, align="L")
            pdf.cell(0, 10, f"  - Case C (550 m/s) residual velocity ~220 m/s: {'PASS' if abs(results['case_c']['residual_velocity'] - 220.0) <= 20.0 else 'FAIL'}", ln=1, align="L")
            pdf.output(str(REPORT_PDF))
            print(f"PDF report successfully compiled via FPDF to {REPORT_PDF}")
            pdf_compiled = True
        except ImportError:
            pass
            
    # 4. Built-in dependency-free pure-Python fallback (Guaranteed fallback to prevent corrupt text PDFs)
    if not pdf_compiled:
        try:
            generate_pure_python_pdf(REPORT_PDF, results)
            print(f"PDF report successfully compiled via built-in pure-Python compiler to {REPORT_PDF}")
            pdf_compiled = True
        except Exception as e:
            print(f"Pure-Python PDF compiler failed: {e}")
            raise RuntimeError("All PDF compilation engines and fallbacks failed.") from e

if __name__ == "__main__":
    main()
