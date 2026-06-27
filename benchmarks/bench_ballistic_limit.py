"""Validation benchmark: Ballistic Limit (V50) Jonas-Laval Curve Fitting.

Sweeps projectile strike velocities from 100 m/s to 450 m/s against a single ply
of Kevlar 29, extracts the residual velocities, fits them to the classical Jonas-Laval
(Lambert-Jonas) equation, and validates the numerical V50 limit against experimental data.
"""

import time
from pathlib import Path

import numpy as np

# Select the fastest JIT-compiled backend (Numba) for rapid sweep execution
from kevlargrid.solver.fused import fused_leapfrog_loop as solver_loop
from kevlargrid.solver.grid import generate_rectangular_grid
from kevlargrid.solver.timestep import compute_cfl_timestep

PLOT_FILE = Path(__file__).parent / "ballistic_limit_validation.png"


def run_single_impact(v_strike: float) -> float:
    """Run a single impact simulation and return the final residual velocity (m/s)."""
    nx, ny = 31, 31
    dx = 0.01
    n_nodes = nx * ny

    material_kev29 = {
        "tensile_modulus_gpa": 71.0,
        "areal_density_kgm2": 0.47,
        "fiber_density_gcc": 1.44,
        "shear_ratio": 0.0004,
    }

    proj_mass = 0.0011
    blade_width = 0.00635
    edge_thickness = 0.00635
    k_penalty = 2e5

    # Boundary conditions: Clamped edges
    boundary_mask = np.zeros(n_nodes, dtype=bool)
    for i in range(nx):
        for j in range(ny):
            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                boundary_mask[i * ny + j] = True

    grid = generate_rectangular_grid(nx, ny, dx, material_kev29)
    positions = grid.nodes.copy()
    velocities = np.zeros_like(positions)

    # Strike from just below (Z = -0.002m) moving upwards (+Z direction)
    proj_pos = np.array([0.0, 0.0, -0.002], dtype=np.float64)
    proj_vel = np.array([0.0, 0.0, v_strike], dtype=np.float64)

    dt = compute_cfl_timestep(grid.stiffnesses, grid.masses, dx, 0.2)
    t_sim = 0.0

    # Run for up to 6 chunks (600 steps)
    for _ in range(6):
        (
            positions,
            velocities,
            grid.failed,
            proj_pos,
            proj_vel,
            *_,
        ) = solver_loop(
            positions,
            velocities,
            grid.springs,
            grid.stiffnesses,
            grid.rest_lengths,
            grid.failed,
            grid.masses,
            grid.tension_only,
            boundary_mask,
            np.zeros((n_nodes, 3)),
            proj_pos,
            proj_vel,
            proj_mass,
            blade_width,
            edge_thickness,
            1,
            n_nodes,
            0.002,
            dx,
            k_penalty,
            0.05,
            1e-7,
            0.04,
            0.024,
            1.5,
            dt,
            100,
            100,
            0.0,
            0.0,
            0.0,
            t_sim,
            1.0,
            grid.initial_spring_counts,
            grid.node_spring_offsets,
            grid.node_spring_ids,
            grid.node_spring_signs,
        )
        t_sim += 100 * dt

        # Stop early if projectile is arrested and rebounding (reversed velocity)
        if proj_vel[2] <= 0.0:
            return 0.0

        # Stop early if projectile successfully penetrated and passed the fabric plane
        if proj_pos[2] > 0.008 and proj_vel[2] > 0.0:
            break

    # Return final Z-velocity (clamp to 0 if arrested or going downwards)
    return max(0.0, float(proj_vel[2]))


def fit_jonas_laval(v_strike: np.ndarray, v_residual: np.ndarray) -> tuple[float, float]:
    """Perform a grid search to fit the Jonas-Laval (Lambert-Jonas) equation parameters."""
    best_v50 = 200.0
    best_alpha = 1.0
    min_sse = 1e20

    # Grid search for V50 in [100, 350] m/s and alpha scaling in [0.7, 1.1]
    for v50 in np.linspace(100.0, 350.0, 251):
        for alpha in np.linspace(0.7, 1.1, 81):
            pred = np.zeros_like(v_strike)
            mask = v_strike > v50
            pred[mask] = alpha * np.sqrt(v_strike[mask] ** 2 - v50 ** 2)

            sse = np.sum((v_residual - pred) ** 2)
            if sse < min_sse:
                min_sse = sse
                best_v50 = v50
                best_alpha = alpha

    return float(best_v50), float(best_alpha)


def main() -> None:
    print("Executing ballistic limit (V50) velocity sweep...")
    strike_vels = np.array([100, 150, 180, 200, 220, 240, 260, 280, 310, 340, 370, 400, 430, 460], dtype=np.float64)
    residual_vels = []

    start_time = time.perf_counter()
    for v_s in strike_vels:
        t0 = time.perf_counter()
        v_r = run_single_impact(v_s)
        residual_vels.append(v_r)
        print(f"  Strike: {v_s:3.0f} m/s | Residual: {v_r:5.1f} m/s | Duration: {time.perf_counter() - t0:.3f} s")

    residual_vels = np.array(residual_vels, dtype=np.float64)
    elapsed = time.perf_counter() - start_time
    print(f"Sweep complete in {elapsed:.3f} seconds.")

    # Fit parameters to Jonas-Laval curve
    v50_fit, alpha_fit = fit_jonas_laval(strike_vels, residual_vels)
    print("\nJonas-Laval (Lambert-Jonas) Fit Results:")
    print(f"  V50 Ballistic Limit: {v50_fit:.1f} m/s")
    print(f"  Velocity Scale (alpha): {alpha_fit:.3f}")

    # Plot results
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

        # Plot simulated points
        plt.scatter(
            strike_vels,
            residual_vels,
            color="#e74c3c",
            s=80,
            zorder=4,
            label="KevlarGrid Simulation Points",
        )

        # Plot fitted curve
        v_s_plot = np.linspace(100.0, 480.0, 500)
        v_r_plot = np.zeros_like(v_s_plot)
        mask = v_s_plot > v50_fit
        v_r_plot[mask] = alpha_fit * np.sqrt(v_s_plot[mask] ** 2 - v50_fit ** 2)
        plt.plot(
            v_s_plot,
            v_r_plot,
            color="#34495e",
            linewidth=2.5,
            zorder=3,
            label=f"Jonas-Laval Fit ($V_{{50}} = {v50_fit:.1f}$ m/s)",
        )

        # Plot experimental range band for Kevlar 29 1-ply 17-grain FSP (205-235 m/s)
        plt.axvspan(
            205.0,
            235.0,
            alpha=0.15,
            color="#2ecc71",
            zorder=1,
            label="Experimental V50 Range (205-235 m/s)",
        )
        plt.axvline(220.0, color="#2ecc71", linestyle="--", linewidth=1.5, zorder=2, label="Experimental Mean (220 m/s)")

        plt.title("VibeDynaLITE Kevlar 29 Ballistic Limit Validation (1-Ply, 17-Grain FSP)", fontsize=12, fontweight="bold")
        plt.xlabel("Strike Velocity $V_s$ (m/s)", fontsize=11)
        plt.ylabel("Residual Velocity $V_r$ (m/s)", fontsize=11)
        plt.xlim(80, 480)
        plt.ylim(-10, 480)
        plt.legend(frameon=True, facecolor="white", edgecolor="#e0e0e0", fontsize=10, loc="upper left")

        plt.tight_layout()
        plt.savefig(PLOT_FILE, dpi=300)
        plt.close()
        print(f"Validation curve plot exported to {PLOT_FILE}")
    except Exception as e:
        print(f"Failed to generate plot: {e}")


if __name__ == "__main__":
    main()
