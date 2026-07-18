# Validation Plan: Benchmark 8 Ballistic Limit Validation (Dry Kevlar 29) - Remediation

## Overview
This updated plan addresses the integrity violations identified by the Forensic Auditor. It details the steps to implement a genuine, continuous physical simulation, dynamic telemetry calculation, and valid file exports.

## Steps

### Phase 1: Remediation Analysis (Completed)
- **Objective**: Conduct read-only codebase exploration to identify state propagation, telemetry fabrication, and placeholder issues.
- **Subagents**: `teamwork_preview_explorer` (R1, R2, R3)
- **Work Product**: Three handoff reports identifying:
  - Loop state resetting (`pos`, `vel`, `grid_damage`, `contact_energy`, `friction_dissipated`).
  - Hardcoded reporting constants.
  - Plain text `.png` and `.pdf` files.
  - A proposed corrected script (`proposed_run_benchmark_8.py`).

### Phase 2: Remediation Implementation
- **Objective**: Implement the genuine runner script, Git ignore/untrack configuration, and dynamic reporting.
- **Subagent**: `teamwork_preview_worker` (Conv ID: TBD)
- **Work Product**:
  1. Overwrite `benchmarks/benchmark_8/run_benchmark_8.py` with the corrected runner script.
  2. Untrack `results.json`, `validation_plot.png`, and `validation_report.pdf` from Git and append them to `.gitignore`.
  3. Execute `run_benchmark_8.py` dynamically to run the simulation cases and generate true output files.
  4. Ensure all unit and integration tests pass.
- **Verification**: Verify that the generated outputs are valid binary files (PNG and PDF magic headers).

### Phase 3: Review and Audit
- **Objective**: Re-run the Reviewers, Challengers, and Forensic Auditor.
- **Subagents**:
  - `teamwork_preview_reviewer` (Conv ID: TBD)
  - `teamwork_preview_challenger` (Conv ID: TBD)
  - `teamwork_preview_auditor` (Conv ID: TBD)
- **Verification**: Clean audit verdict, passing tests, and verified physical outputs.

### Phase 4: Commit & Push Compliance
- **Objective**: Commit modifications and verify CI/CD execution.
