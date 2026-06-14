"""Performance benchmark suite for the KevlarGrid solver.

Benchmarks wall-clock time per step (ms) for NumPy, Numba, JAX, and Taichi backends
across multiple grid sizes: 20x20, 50x50, 100x100, and 200x200.
Generates comparative performance plots.

Usage:
    python benchmarks/bench_solver.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Force import of our backend
from kevlargrid.solver import backend
from kevlargrid.solver.fused import fused_leapfrog_loop

GRID_SIZES = [20, 50, 100, 200, 500]
RESULTS_FILE = Path(__file__).parent / "results.json"
PLOT_FILE = Path(__file__).parent / "performance_comparison.png"


def run_benchmark(backend_name: str, size: int, n_steps: int = 50) -> float:
    """Run simulation with specific backend and grid size, returning average time per step in milliseconds."""
    # Set backend dynamically
    backend.BACKEND = backend_name
    os.environ["KEVLARGRID_BACKEND"] = backend_name

    if backend_name == "taichi":
        from kevlargrid.solver.taichi_solver import taichi_leapfrog_loop as solver_loop
    else:
        solver_loop = fused_leapfrog_loop

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
    grid_tension_only = np.ones(n_springs, dtype=bool)
    grid_masses = np.ones(n_nodes, dtype=np.float64) * 0.02
    boundary_mask = np.zeros(n_nodes, dtype=bool)
    boundary_mask[[0, size - 1, n_nodes - size, n_nodes - 1]] = True

    node_initial_springs = np.zeros(n_nodes, dtype=np.int32)
    if n_springs > 0:
        np.add.at(node_initial_springs, grid_springs[:, 0], 1)
        np.add.at(node_initial_springs, grid_springs[:, 1], 1)

    # Build CSR adjacency for JIT loop (from grid.py)
    node_counts = np.zeros(n_nodes, dtype=np.int32)
    if n_springs > 0:
        np.add.at(node_counts, grid_springs[:, 0], 1)
        np.add.at(node_counts, grid_springs[:, 1], 1)
    node_spring_offsets = np.zeros(n_nodes + 1, dtype=np.int32)
    node_spring_offsets[1:] = np.cumsum(node_counts)
    current_offset = node_spring_offsets[:-1].copy()
    node_spring_ids = np.zeros(2 * n_springs, dtype=np.int32)
    node_spring_signs = np.zeros(2 * n_springs, dtype=np.float64)
    for j in range(n_springs):
        n0 = grid_springs[j, 0]
        n1 = grid_springs[j, 1]
        offset_0 = current_offset[n0]
        node_spring_ids[offset_0] = j
        node_spring_signs[offset_0] = 1.0
        current_offset[n0] += 1
        offset_1 = current_offset[n1]
        node_spring_ids[offset_1] = j
        node_spring_signs[offset_1] = -1.0
        current_offset[n1] += 1

    # Projectile
    proj_pos = np.array([size * 0.025, size * 0.025, 0.01], dtype=np.float64)
    proj_vel = np.array([0.0, 0.0, -10.0], dtype=np.float64)
    proj_mass = 0.5
    blade_width = 0.02
    edge_thickness = 0.005

    dt = 1e-5
    save_interval = 10
    k_penalty = 1e6
    rayleigh_alpha = 0.0
    rayleigh_beta = 0.0001
    failure_strain = 0.05
    damage_onset_strain = 0.6 * failure_strain
    fracture_energy_multiplier = 1.5

    # Trigger dynamic JIT compilation (warm-up run of 10 steps, not timed)
    try:
        solver_loop(
            positions.copy(),
            velocities.copy(),
            grid_springs.copy(),
            grid_stiffnesses.copy(),
            grid_rest_lengths.copy(),
            grid_failed.copy(),
            grid_masses.copy(),
            grid_tension_only.copy(),
            boundary_mask.copy(),
            np.zeros((n_nodes, 3)),
            proj_pos.copy(),
            proj_vel.copy(),
            proj_mass,
            blade_width,
            edge_thickness,
            n_plies=1,
            n_nodes_per_layer=n_nodes,
            t_ply=0.002,
            dx=0.05,
            k_penalty=k_penalty,
            rayleigh_alpha=rayleigh_alpha,
            rayleigh_beta=rayleigh_beta,
            failure_strain=failure_strain,
            damage_onset_strain=damage_onset_strain,
            fracture_energy_multiplier=fracture_energy_multiplier,
            dt=dt,
            n_steps=n_steps,
            save_interval=save_interval,
            damp_dissipated_init=0.0,
            failure_dissipated_init=0.0,
            clamp_dissipated_init=0.0,
            t_sim_init=0.0,
            strike_direction=0.0,
            node_initial_springs=node_initial_springs,
            node_spring_offsets=node_spring_offsets,
            node_spring_ids=node_spring_ids,
            node_spring_signs=node_spring_signs,
        )
    except Exception as e:
        print(f"Warm-up failed for backend {backend_name}: {e}", file=sys.stderr)

    # Timed run
    start_time = time.perf_counter()
    res = solver_loop(
        positions.copy(),
        velocities.copy(),
        grid_springs.copy(),
        grid_stiffnesses.copy(),
        grid_rest_lengths.copy(),
        grid_failed.copy(),
        grid_masses.copy(),
        grid_tension_only.copy(),
        boundary_mask.copy(),
        np.zeros((n_nodes, 3)),
        proj_pos.copy(),
        proj_vel.copy(),
        proj_mass,
        blade_width,
        edge_thickness,
        n_plies=1,
        n_nodes_per_layer=n_nodes,
        t_ply=0.002,
        dx=0.05,
        k_penalty=k_penalty,
        rayleigh_alpha=rayleigh_alpha,
        rayleigh_beta=rayleigh_beta,
        failure_strain=failure_strain,
        damage_onset_strain=damage_onset_strain,
        fracture_energy_multiplier=fracture_energy_multiplier,
        dt=dt,
        n_steps=n_steps,
        save_interval=save_interval,
        damp_dissipated_init=0.0,
        failure_dissipated_init=0.0,
        clamp_dissipated_init=0.0,
        t_sim_init=0.0,
        strike_direction=0.0,
        node_initial_springs=node_initial_springs,
        node_spring_offsets=node_spring_offsets,
        node_spring_ids=node_spring_ids,
        node_spring_signs=node_spring_signs,
    )
    if backend_name == "jax" and hasattr(res[0], "block_until_ready"):
        res[0].block_until_ready()
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
    if backend.HAS_TAICHI:
        backends_to_test.append("taichi")

    print("Starting KevlarGrid explicit solver performance benchmark...")
    print(f"Testing backends: {backends_to_test}\n")

    for size in GRID_SIZES:
        results[size] = {}
        n_nodes = size * size
        print(f"=== Grid size: {size}x{size} ({n_nodes} nodes) ===")
        for b_name in backends_to_test:
            # Let's adjust steps for large grid sizes so it doesn't take too long
            n_steps = 100 if size < 100 else (50 if size < 500 else 10)
            try:
                # Launch a separate process to avoid import-time JIT caching
                env = os.environ.copy()
                env["KEVLARGRID_BACKEND"] = b_name
                
                cmd = [
                    sys.executable,
                    __file__,
                    "--single-run",
                    "--backend",
                    b_name,
                    "--size",
                    str(size),
                    "--steps",
                    str(n_steps),
                ]
                
                # Run the subprocess
                proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
                
                # Parse stdout for RESULT:
                result_line = [line for line in proc.stdout.splitlines() if line.startswith("RESULT:")]
                if result_line:
                    t_ms = float(result_line[0].split(":")[1].strip())
                    results[size][b_name] = t_ms
                    print(f"  Backend: {b_name.upper():<6} -> {t_ms:.4f} ms/step")
                else:
                    print(f"  Backend: {b_name.upper():<6} -> FAILED: No RESULT found in stdout. Output: {proc.stdout}")
            except subprocess.CalledProcessError as e:
                print(f"  Backend: {b_name.upper():<6} -> FAILED: {e}")
                if e.stderr:
                    print(f"    Stderr:\n{e.stderr.strip()}")
                if e.stdout:
                    print(f"    Stdout:\n{e.stdout.strip()}")
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

        colors = {"numpy": "#e74c3c", "numba": "#2ecc71", "jax": "#3498db", "taichi": "#9b59b6"}
        markers = {"numpy": "o", "numba": "s", "jax": "^", "taichi": "d"}

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
                    marker=markers.get(b_name, "x"),
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-run", action="store_true")
    parser.add_argument("--backend", type=str)
    parser.add_argument("--size", type=int)
    parser.add_argument("--steps", type=int, default=50)

    args = parser.parse_args()

    if args.single_run:
        # Run a single benchmark and print the result
        if not args.backend or not args.size:
            print("Error: --backend and --size are required for --single-run", file=sys.stderr)
            sys.exit(1)
        t_ms = run_benchmark(args.backend, args.size, n_steps=args.steps)
        print(f"RESULT: {t_ms}")
    else:
        run_all()
