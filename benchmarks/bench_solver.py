"""Performance benchmark suite for the KevlarGrid solver.

Benchmarks wall-clock time per 1000 timesteps for grid sizes:
50x50, 100x100, 200x200, 500x500, 1000x1000.

Usage:
    python benchmarks/bench_solver.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

GRID_SIZES = [50, 100, 200, 500, 1000]
RESULTS_FILE = Path(__file__).parent / "results.json"


def benchmark_grid(size: int, n_steps: int = 1000) -> dict:
    """Benchmark solver for a given grid size."""
    # TODO: Implement when solver is ready
    return {"grid_size": size, "n_steps": n_steps, "wall_time_s": 0.0, "time_per_step_ms": 0.0}


def run_all() -> None:
    """Run benchmarks for all grid sizes and save results."""
    results = []
    for size in GRID_SIZES:
        print(f"Benchmarking {size}x{size} grid...")
        result = benchmark_grid(size)
        results.append(result)
        print(f"  {result['wall_time_s']:.3f}s total, {result['time_per_step_ms']:.3f}ms/step")

    # Ensure output folder exists
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    run_all()
