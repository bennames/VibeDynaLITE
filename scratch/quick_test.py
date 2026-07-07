import sys
from pathlib import Path
# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.benchmark_8.run_benchmark_8 import run_case

print("=== Running with Numba ===")
res_numba = run_case(450.0, "A", "numba")
print("Numba result:", res_numba)

print("\n=== Running with Taichi ===")
try:
    res_taichi = run_case(450.0, "A", "taichi")
    print("Taichi result:", res_taichi)
except Exception as e:
    print("Taichi failed:", e)
