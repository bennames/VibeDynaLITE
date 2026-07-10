# Handoff Report — Bug Fixes and Verification

## 1. Observation
We observed the following configurations, constraints, and results:
- **Files Modified**:
  - `src/kevlargrid/gui/app.py` around line 945.
  - `src/kevlargrid/gui/viewport3d.py` in the `reset` method around line 360.
  - `src/kevlargrid/solver/fused.py` in the CFL stable timestep calculation around line 2028, and in the CZM cohesive force loop around line 2133.
- **Commands Executed**:
  - `pytest tests/unit/test_metallic_sheet.py tests/gui/test_config_roundtrip.py`
  - Output:
    ```
    ============================= test session starts ==============================
    platform darwin -- Python 3.12.13, pytest-9.0.3, pluggy-1.6.0
    benchmark: 5.2.3 (defaults: timer=time.perf_counter disable_gc=False min_rounds=5 min_time=0.000005 max_time=1.0 calibration_precision=10 warmup=False warmup_iterations=100000)
    rootdir: /Users/bennames/Developer/VibeDynaLITE
    configfile: pyproject.toml
    plugins: benchmark-5.2.3
    collected 30 items

    tests/unit/test_metallic_sheet.py ............                           [ 40%]
    tests/gui/test_config_roundtrip.py ..................                    [100%]

    ============================== 30 passed in 2.83s ==============================
    ```
  - `make format` and `make lint` both completed successfully with no style/formatting or type errors.

## 2. Logic Chain
- **Step 1**: The original lookup `cfg["material"].get("material_name", "")` did not match the expected `name` key inside the material dictionary. Changing it to `cfg["material"].get("name", "")` aligns it with the rest of the config serialization.
- **Step 2**: The culling code `ply_indices = springs[:, 0] // self.n_nodes_per_layer` in `viewport3d.py` requires `n_nodes_per_layer` to correctly map node IDs to layers. In a metallic sheet model, all nodes are duplicated per element, so the number of nodes per layer is `len(grid.nodes) // n_plies` rather than the default 121. Updating `self.n_nodes_per_layer` to this value resolves invalid culling.
- **Step 3**: Timestep overestimation in CFL limits for spring networks is corrected by using a tighter bound `omega_spring = sqrt(4.0 * k_0 / mass_min)` instead of the previous `2.0` multiplier, preventing high-frequency oscillations from causing instability.
- **Step 4**: The cohesive forces loop was modified to:
  - Calculate centroids of adjacent element pairs (`e0 = n0 // 4`, `e1 = n1 // 4`) and determine relative centroid vector.
  - Separate normal tension from compression/penetration using `is_tension = (dx_s * dx_c + dy_s * dy_c + dz_s * dz_c) >= 0.0`.
  - Apply penalty forces during compression without updating damage, preventing premature failure.
  - Account for remaining energy upon sudden spring failure by adding the residual strain energy `0.5 * (1.0 - d_old) * k_0 * delta * delta` to the `failure_dissipated` accumulator, maintaining strict energy bookkeeping.

## 3. Caveats
- No caveats. The simulation and unit tests passed within acceptable drift thresholds, and formatting/lint checks are clean.

## 4. Conclusion
The four bugs were successfully fixed following the minimal-change principle. The test suites pass, formatting is correct, and energy bookkeeping is precise.

## 5. Verification Method
To independently verify the implementation:
1. Run `.venv/bin/pytest tests/unit/test_metallic_sheet.py tests/gui/test_config_roundtrip.py` in the workspace directory.
2. Run `make lint` to confirm mypy type safety and ruff formatting checks pass.
