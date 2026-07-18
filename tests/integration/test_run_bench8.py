import json
import sys
from pathlib import Path

import pytest


@pytest.mark.slow
@pytest.mark.skip(reason="Temporarily deprecated for this sprint per user request")
def test_run_benchmark_8(monkeypatch):
    bench_dir = Path("/Users/bennames/Developer/VibeDynaLITE/benchmarks/benchmark_8")
    sys.path.insert(0, str(bench_dir))

    monkeypatch.setattr(sys, "argv", ["run_benchmark_8.py", "--backend", "numba"])

    import run_benchmark_8

    run_benchmark_8.main()

    # Verify generated results.json file assertions
    results_path = bench_dir / "results.json"
    assert results_path.is_file(), "results.json was not generated"
    with open(results_path) as f:
        results = json.load(f)

    for case_id in ["case_a", "case_b", "case_c"]:
        assert case_id in results, f"{case_id} missing from results"
        drift = results[case_id]["energy_drift_pct"]
        assert drift < 10.0, f"{case_id} energy drift is too high: {drift:.2f}%"
