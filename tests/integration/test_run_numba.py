import sys
import time
from pathlib import Path

# Add the benchmark 8 directory to sys.path so we can import it
bench_dir = Path("/Users/bennames/Developer/VibeDynaLITE/benchmarks/benchmark_8")
sys.path.insert(0, str(bench_dir))

def test_run_numba():
    import run_benchmark_8

    print("\n--- Running Case A with NUMBA backend ---")
    sys.stdout.flush()
    t0 = time.time()
    try:
        res_numba = run_benchmark_8.run_case(450.0, "A_numba", "numba")
        t_numba = time.time() - t0
        print(f"NUMBA finished in {t_numba:.2f} seconds.")
        print("NUMBA Results:", res_numba)
        sys.stdout.flush()
    except Exception as e:
        print("NUMBA failed:", e)
        sys.stdout.flush()
        raise e
