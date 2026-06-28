"""Performance benchmark suite for the KevlarGrid solver.

Benchmarks wall-clock time per step (ms) for Taichi CPU and GPU backends
across multiple grid sizes: 25x25, 50x50, 100x100, and 200x200, under
Mode A (single equivalent ply) and Mode B (5 discrete plies).
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

GRID_SIZES = [25, 50, 100, 200]
RESULTS_FILE = Path(__file__).parent / "results.json"
PLOT_FILE = Path(__file__).parent / "performance_comparison.png"


def run_benchmark(arch_name: str, size: int, mode: str, n_steps: int = 50) -> float:
    """Run simulation with specific arch, grid size, and mode, returning average time per step in ms."""
    if arch_name == "numba":
        from kevlargrid.solver.fused import fused_leapfrog_loop as solver_loop
    else:
        import taichi as ti

        if arch_name == "gpu":
            try:
                ti.init(arch=ti.gpu, default_fp=ti.f32)
            except Exception:
                ti.init(arch=ti.cpu, default_fp=ti.f32)
        else:
            ti.init(arch=ti.cpu, default_fp=ti.f32)

        from kevlargrid.solver.taichi_solver import taichi_leapfrog_loop as solver_loop
    from kevlargrid.solver.grid import generate_rectangular_grid

    MOCK_MATERIAL = {
        "tensile_modulus_gpa": 71.0,
        "areal_density_kgm2": 0.47,
        "fiber_density_gcc": 1.44,
        "shear_ratio": 0.0004,
    }

    n_plies = 5
    if mode == "A":
        grid = generate_rectangular_grid(
            size, size, 0.05, MOCK_MATERIAL, n_plies=n_plies, t_ply=None
        )
        t_ply = 0.002
    else:  # mode == "B"
        t_ply = 0.001
        grid = generate_rectangular_grid(
            size, size, 0.05, MOCK_MATERIAL, n_plies=n_plies, t_ply=t_ply
        )

    n_nodes = grid.n_nodes

    positions = grid.nodes
    velocities = np.zeros((n_nodes, 3), dtype=np.float64)
    boundary_mask = np.zeros(n_nodes, dtype=bool)

    # boundary conditions: clamp edges
    n_nodes_per_layer = size * size
    for ply in range(n_plies if mode == "B" else 1):
        offset = ply * n_nodes_per_layer
        for i in range(size):
            for j in range(size):
                if i == 0 or i == size - 1 or j == 0 or j == size - 1:
                    boundary_mask[offset + i * size + j] = True

    # Projectile
    proj_pos = np.array([0.0, 0.0, 0.01], dtype=np.float64)
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

    # Trigger dynamic JIT compilation (warm-up run)
    try:
        solver_loop(
            positions.copy(),
            velocities.copy(),
            grid.springs.copy(),
            grid.stiffnesses.copy(),
            grid.rest_lengths.copy(),
            grid.failed.copy(),
            grid.masses.copy(),
            grid.tension_only.copy(),
            boundary_mask.copy(),
            np.zeros((n_nodes, 3)),
            proj_pos.copy(),
            proj_vel.copy(),
            proj_mass,
            blade_width,
            edge_thickness,
            n_plies=n_plies if mode == "B" else 1,
            n_nodes_per_layer=n_nodes_per_layer,
            t_ply=t_ply,
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
            node_initial_springs=grid.initial_spring_counts,
            node_spring_offsets=grid.node_spring_offsets,
            node_spring_ids=grid.node_spring_ids,
            node_spring_signs=grid.node_spring_signs,
        )
    except Exception as e:
        print(f"Warm-up failed for arch {arch_name}: {e}", file=sys.stderr)

    # Timed run
    start_time = time.perf_counter()
    solver_loop(
        positions.copy(),
        velocities.copy(),
        grid.springs.copy(),
        grid.stiffnesses.copy(),
        grid.rest_lengths.copy(),
        grid.failed.copy(),
        grid.masses.copy(),
        grid.tension_only.copy(),
        boundary_mask.copy(),
        np.zeros((n_nodes, 3)),
        proj_pos.copy(),
        proj_vel.copy(),
        proj_mass,
        blade_width,
        edge_thickness,
        n_plies=n_plies if mode == "B" else 1,
        n_nodes_per_layer=n_nodes_per_layer,
        t_ply=t_ply,
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
        node_initial_springs=grid.initial_spring_counts,
        node_spring_offsets=grid.node_spring_offsets,
        node_spring_ids=grid.node_spring_ids,
        node_spring_signs=grid.node_spring_signs,
    )
    if arch_name != "numba":
        ti.sync()
    elapsed = time.perf_counter() - start_time
    time_per_step_ms = (elapsed / n_steps) * 1000.0
    return time_per_step_ms


def run_all() -> None:
    """Run CPU vs GPU benchmarks across grid sizes and save results/plots."""
    results = {}
    arches_to_test = ["cpu", "gpu", "numba"]
    modes = ["A", "B"]

    print("Starting KevlarGrid explicit solver performance benchmark...")
    print(f"Testing architectures: {arches_to_test}\n")

    for size in GRID_SIZES:
        results[size] = {}
        for mode in modes:
            results[size][mode] = {}
            n_nodes = size * size * (5 if mode == "B" else 1)
            print(f"=== Grid size: {size}x{size} (Mode {mode}, {n_nodes} nodes) ===")
            for arch in arches_to_test:
                # Adjust steps for large grid sizes so it doesn't take too long
                n_steps = 100 if size < 100 else (50 if size < 200 else 10)
                try:
                    # Launch a separate process to avoid import-time JIT caching
                    env = os.environ.copy()
                    cmd = [
                        sys.executable,
                        __file__,
                        "--single-run",
                        "--arch",
                        arch,
                        "--size",
                        str(size),
                        "--mode",
                        mode,
                        "--steps",
                        str(n_steps),
                    ]

                    # Run the subprocess
                    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)

                    # Parse stdout for RESULT:
                    result_line = [
                        line for line in proc.stdout.splitlines() if line.startswith("RESULT:")
                    ]
                    if result_line:
                        t_ms = float(result_line[0].split(":")[1].strip())
                        results[size][mode][arch] = t_ms
                        print(f"  Arch: {arch.upper():<6} -> {t_ms:.4f} ms/step")
                    else:
                        print(
                            f"  Arch: {arch.upper():<6} -> FAILED: No RESULT found in stdout. Output: {proc.stdout}"
                        )
                except subprocess.CalledProcessError as e:
                    print(f"  Arch: {arch.upper():<6} -> FAILED: {e}")
                    if e.stderr:
                        print(f"    Stderr:\n{e.stderr.strip()}")
                    if e.stdout:
                        print(f"    Stdout:\n{e.stdout.strip()}")
                except Exception as e:
                    print(f"  Arch: {arch.upper():<6} -> FAILED: {e}")

    # Write JSON results
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    serialized_results = {
        "metadata": {
            "arches": arches_to_test,
            "sizes": GRID_SIZES,
            "modes": modes,
            "timestamp": time.time(),
        },
        "data": results,
    }
    RESULTS_FILE.write_text(json.dumps(serialized_results, indent=2))
    print(f"\nJSON results saved to {RESULTS_FILE}")

    # Render professional plot
    try:
        import matplotlib.pyplot as plt

        _fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        plt.style.use(
            "seaborn-v0_8-whitegrid"
            if "seaborn-v0_8-whitegrid" in plt.style.available
            else "default"
        )

        x_vals = [size * size for size in GRID_SIZES]
        colors = {"cpu": "#e74c3c", "gpu": "#9b59b6", "numba": "#2ecc71"}
        markers = {"cpu": "o", "gpu": "d", "numba": "s"}

        for idx, mode in enumerate(modes):
            ax = axes[idx]
            for arch in arches_to_test:
                y_vals = []
                for size in GRID_SIZES:
                    val = results[size][mode].get(arch, None)
                    if val is not None:
                        y_vals.append(val)
                if len(y_vals) == len(GRID_SIZES):
                    label_name = "Numba" if arch == "numba" else f"Taichi {arch.upper()}"
                    ax.plot(
                        x_vals,
                        y_vals,
                        label=label_name,
                        color=colors.get(arch, "#95a5a6"),
                        marker=markers.get(arch, "x"),
                        linewidth=2.5,
                        markersize=8,
                    )
            mode_name = (
                "Mode A (Single Equivalent Sheet)" if mode == "A" else "Mode B (5 Discrete Plies)"
            )
            ax.set_title(f"{mode_name}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Nodes per Layer ($N_{nodes}$)", fontsize=11)
            if idx == 0:
                ax.set_ylabel("Execution Time per Step (ms)", fontsize=11)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.legend(frameon=True, facecolor="white", edgecolor="#e0e0e0", fontsize=10)

        plt.suptitle(
            "KevlarGrid Explicit Mass-Spring Solver Performance Scaling (5 Plies)",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout()
        plt.savefig(PLOT_FILE, dpi=300)
        plt.close()
        print(f"Stunning performance comparison plot exported to {PLOT_FILE}")
    except Exception as e:
        print(f"\nPlot generation failed: {e}. Exported raw JSON results only.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-run", action="store_true")
    parser.add_argument("--arch", type=str)
    parser.add_argument("--size", type=int)
    parser.add_argument("--mode", type=str, choices=["A", "B"], default="A")
    parser.add_argument("--steps", type=int, default=50)

    args = parser.parse_args()

    if args.single_run:
        if not args.arch or not args.size:
            print("Error: --arch and --size are required for --single-run", file=sys.stderr)
            sys.exit(1)
        t_ms = run_benchmark(args.arch, args.size, mode=args.mode, n_steps=args.steps)
        print(f"RESULT: {t_ms}")
    else:
        run_all()
