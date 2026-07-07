import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks" / "benchmark_8"))
import run_benchmark_8

print("Starting Numba Case A...")
res = run_benchmark_8.run_case(450.0, "A_numba_test", "numba")
print("Finished Numba Case A. Result:", res)

with open(str(Path(__file__).parent.parent / "scratch" / "numba_result.txt"), "w") as f:
    json.dump(res, f, indent=4)
