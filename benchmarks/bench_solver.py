"""Performance benchmark suite for the KevlarGrid solver.

Benchmarks wall-clock time per step (ms) for NumPy, Numba, and JAX backends
across multiple grid sizes: 20x20, 50x50, 100x100, and 200x200.
Generates comparative performance plots.

Usage:
    python benchmarks/bench_solver.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

# Force import of our backend
from kevlargrid.solver import backend
from kevlargrid.solver.fused import fused_leapfrog_loop

GRID_SIZES = [20, 50, 100, 200]
RESULTS_FILE = Path(__file__).parent / "results.json"
PLOT_FILE = Path(__file__).parent / "performance_comparison.png"


def run_benchmark(backend_name: str, size: int, n_steps: int = 50) -> float:
    """Run simulation with specific backend and grid size, returning average time per step in milliseconds."""
    # Set backend dynamically
    backend.BACKEND = backend_name
    os.environ["KEVLARGRID_BACKEND"] = backend_name

    # Create grid geometry
    n_nodes = size * size
    positions = np.zeros((n_nodes, 3), dtype=np.float64)
    for y in range(size):
        for x in range(size):
            idx = y * size + x
            positions[idx] = [x * 0.05, y * 0.05, 0.0]

    velocities = np.zeros((n_nodes, 3), dtype=np.float64)

    # connectivity
    springs_list = []
    for y in range(size):
        for x in range(size):
            idx = y * size + x
            if x < size - 1:
                springs_list.append([idx, idx + 1])
            if y < size - 1:
                springs_list.append([idx, idx + size])

    grid_springs = np.array(springs_list, dtype=np.int32)
    n_springs = len(grid_springs)
    grid_stiffnesses = np.ones(n_springs, dtype=np.float64) * 1e5
    grid_rest_lengths = np.ones(n_springs, dtype=np.float64) * 0.05
    grid_failed = np.zeros(n_springs, dtype=bool)
    grid_masses = np.ones(n_nodes, dtype=np.float64) * 0.02
    boundary_mask = np.zeros(n_nodes, dtype=bool)
    boundary_mask[[0, size - 1, n_nodes - size, n_nodes - 1]] = True

    # Projectile
    proj_pos = np.array([size * 0.025, size * 0.025, 0.01], dtype=np.float64)
    proj_vel = np.array([0.0, 0.0, -10.0], dtype=np.float64)
    proj_mass = 0.5
    blade_width = 0.02
    edge_thickness = 0.005

    dt = 1e-5
    save_interval = 10
    k_penalty = 1e6
    damping_coeff = 0.1
    failure_strain = 0.05

    # Trigger dynamic JIT compilation (warm-up run of 10 steps, not timed)
    try:
        fused_leapfrog_loop(
            positions.copy(),
            velocities.copy(),
            grid_springs.copy(),
            grid_stiffnesses.copy(),
            grid_rest_lengths.copy(),
            grid_failed.copy(),
            grid_masses.copy(),
            boundary_mask.copy(),
            proj_pos.copy(),
            proj_vel.copy(),
            proj_mass,
            blade_width,
            edge_thickness,
            n_plies=1,
            n_nodes_per_layer=n_nodes,
            t_ply=0.002,
            k_penalty=k_penalty,
            damping_coeff=damping_coeff,
            failure_strain=failure_strain,
            dt=dt,
            n_steps=10,
            save_interval=save_interval,
            damp_dissipated_init=0.0,
            t_sim_init=0.0,
        )
    except Exception as e:
        print(f"Warm-up failed for backend {backend_name}: {e}")

    # Timed run
    start_time = time.perf_counter()
    fused_leapfrog_loop(
        positions.copy(),
        velocities.copy(),
        grid_springs.copy(),
        grid_stiffnesses.copy(),
        grid_rest_lengths.copy(),
        grid_failed.copy(),
        grid_masses.copy(),
        boundary_mask.copy(),
        proj_pos.copy(),
        proj_vel.copy(),
        proj_mass,
        blade_width,
        edge_thickness,
        n_plies=1,
        n_nodes_per_layer=n_nodes,
        t_ply=0.002,
        k_penalty=k_penalty,
        damping_coeff=damping_coeff,
        failure_strain=failure_strain,
        dt=dt,
        n_steps=n_steps,
        save_interval=save_interval,
        damp_dissipated_init=0.0,
        t_sim_init=0.0,
    )
    elapsed = time.perf_counter() - start_time
    time_per_step_ms = (elapsed / n_steps) * 1000.0
    return time_per_step_ms


def run_all() -> None:
    """Run all backend benchmarks across grid sizes and save results/plots."""
    results = {}
    backends_to_test = ["numpy"]
    if backend.HAS_NUMBA:
        backends_to_test.append("numba")
    if backend.HAS_JAX:
        backends_to_test.append("jax")

    print("Starting KevlarGrid explicit solver performance benchmark...")
    print(f"Testing backends: {backends_to_test}\n")

    for size in GRID_SIZES:
        results[size] = {}
        n_nodes = size * size
        print(f"=== Grid size: {size}x{size} ({n_nodes} nodes) ===")
        for b_name in backends_to_test:
            # Let's adjust steps for large grid sizes so it doesn't take too long
            n_steps = 100 if size < 100 else 50
            try:
                t_ms = run_benchmark(b_name, size, n_steps=n_steps)
                results[size][b_name] = t_ms
                print(f"  Backend: {b_name.upper():<6} -> {t_ms:.4f} ms/step")
            except Exception as e:
                print(f"  Backend: {b_name.upper():<6} -> FAILED: {e}")

    # Write JSON results
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    serialized_results = {
        "metadata": {
            "backends": backends_to_test,
            "sizes": GRID_SIZES,
            "timestamp": time.time(),
        },
        "data": results,
    }
    RESULTS_FILE.write_text(json.dumps(serialized_results, indent=2))
    print(f"\nJSON results saved to {RESULTS_FILE}")

    # Render professional plot S7
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

        x_vals = [size * size for size in GRID_SIZES]

        colors = {"numpy": "#e74c3c", "numba": "#2ecc71", "jax": "#3498db"}
        markers = {"numpy": "o", "numba": "s", "jax": "^"}

        for b_name in backends_to_test:
            y_vals = []
            for size in GRID_SIZES:
                val = results[size].get(b_name, None)
                if val is not None:
                    y_vals.append(val)
            if len(y_vals) == len(GRID_SIZES):
                plt.plot(
                    x_vals,
                    y_vals,
                    label=f"{b_name.upper()}",
                    color=colors.get(b_name, "#95a5a6"),
                    marker=markers.get(b_name, "d"),
                    linewidth=2.5,
                    markersize=8,
                )

        plt.title("KevlarGrid Explicit Mass-Spring Solver Performance Scaling", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Total System Degree-of-Freedom / Grid Nodes ($N_{nodes}$)", fontsize=12)
        plt.ylabel("Execution Time per Integrator Step (ms)", fontsize=12)
        plt.xscale("log")
        plt.yscale("log")
        plt.legend(frameon=True, facecolor="white", edgecolor="#e0e0e0", fontsize=11)
        plt.tight_layout()

        plt.savefig(PLOT_FILE, dpi=300)
        plt.close()
        print(f"Stunning performance comparison plot exported to {PLOT_FILE}")
    except ImportError:
        print("\nMatplotlib not available. Skipping plotting, exported raw JSON results only.")


if __name__ == "__main__":
    run_all()
