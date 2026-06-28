import sys
import time
from pathlib import Path

# Add the benchmark 8 directory to sys.path so we can import it
bench_dir = Path("/Users/bennames/Developer/VibeDynaLITE/benchmarks/benchmark_8")
sys.path.insert(0, str(bench_dir))

def test_run_calib():
    import run_benchmark_8

    print("\n--- Running Case A with TAICHI backend ---")
    sys.stdout.flush()
    t0 = time.time()
    try:
        res_taichi = run_benchmark_8.run_case(450.0, "A_taichi", "taichi")
        t_taichi = time.time() - t0
        print(f"TAICHI finished in {t_taichi:.2f} seconds.")
        print("TAICHI Results:", res_taichi)
        sys.stdout.flush()
    except Exception as e:
        print("TAICHI failed:", e)
        sys.stdout.flush()
        raise e
