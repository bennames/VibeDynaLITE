import sys
from pathlib import Path

def test_run_benchmark_8(monkeypatch):
    bench_dir = Path("/Users/bennames/Developer/VibeDynaLITE/benchmarks/benchmark_8")
    sys.path.insert(0, str(bench_dir))
    
    monkeypatch.setattr(sys, "argv", ["run_benchmark_8.py", "--backend", "taichi"])
    
    import run_benchmark_8
    run_benchmark_8.main()
