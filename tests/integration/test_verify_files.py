import json
from pathlib import Path


def test_verify_benchmark_8_files():
    bench_dir = Path("/Users/bennames/Developer/VibeDynaLITE/benchmarks/benchmark_8")

    results_path = bench_dir / "results.json"
    plot_path = bench_dir / "validation_plot.png"
    pdf_path = bench_dir / "validation_report.pdf"

    # 1. Verify results.json
    assert results_path.is_file(), "results.json does not exist"
    with open(results_path) as f:
        data = json.load(f)
    assert "case_a" in data
    assert "case_b" in data
    assert "case_c" in data

    # 2. Verify validation_plot.png has PNG header
    assert plot_path.is_file(), "validation_plot.png does not exist"
    with open(plot_path, "rb") as f:
        header = f.read(8)
    assert header == b"\x89PNG\r\n\x1a\n", (
        f"validation_plot.png does not have valid PNG header: {header}"
    )

    # 3. Verify validation_report.pdf has PDF header
    assert pdf_path.is_file(), "validation_report.pdf does not exist"
    with open(pdf_path, "rb") as f:
        header = f.read(5)
    assert header == b"%PDF-", f"validation_report.pdf does not have valid PDF header: {header}"
