# Sprint 0: Project Scaffolding & CI — Task List

**Duration:** 1 week
**Goal:** Repository structure, development environment, CI pipeline, backend abstraction.

---

## Tasks

### S0.1 — Repository Structure

- [ ] Create directory layout:
  ```
  VibeDynaLITE/
  ├── src/
  │   └── kevlargrid/
  │       ├── __init__.py
  │       ├── solver/
  │       │   ├── __init__.py
  │       │   ├── backend.py         # JAX/Numba adapter
  │       │   ├── grid.py            # Grid generation
  │       │   ├── forces.py          # Spring force computation
  │       │   ├── integrator.py      # Time integration (leapfrog)
  │       │   ├── failure.py         # Failure criteria
  │       │   ├── damping.py         # Damping models
  │       │   ├── projectile.py      # Projectile model & contact
  │       │   ├── boundary.py        # Boundary conditions
  │       │   ├── energy.py          # Energy tracking
  │       │   └── timestep.py        # CFL timestep computation
  │       ├── gui/
  │       │   ├── __init__.py
  │       │   ├── app.py             # Main DearPyGui application
  │       │   ├── config_panel.py    # Configuration input panel
  │       │   ├── controls.py        # Start/pause/stop controls
  │       │   ├── plots.py           # Real-time plot widgets
  │       │   ├── viewport3d.py      # 3D visualization viewport
  │       │   └── dashboard.py       # Pass/Fail results dashboard
  │       ├── materials/
  │       │   ├── __init__.py
  │       │   └── library.py         # Built-in material library
  │       ├── io/
  │       │   ├── __init__.py
  │       │   ├── config.py          # JSON config save/load
  │       │   ├── results.py         # HDF5 result export
  │       │   ├── csv_export.py      # CSV export
  │       │   └── report.py          # HTML/PDF report generation
  │       └── utils/
  │           ├── __init__.py
  │           └── units.py           # Unit conversion helpers
  ├── tests/
  │   ├── __init__.py
  │   ├── unit/
  │   │   ├── __init__.py
  │   │   ├── test_grid.py
  │   │   ├── test_forces.py
  │   │   ├── test_integrator.py
  │   │   ├── test_failure.py
  │   │   ├── test_damping.py
  │   │   ├── test_timestep.py
  │   │   ├── test_energy.py
  │   │   └── test_backend.py
  │   ├── integration/
  │   │   ├── __init__.py
  │   │   ├── test_wave_propagation.py
  │   │   ├── test_point_impact.py
  │   │   └── test_multiply.py
  │   ├── regression/
  │   │   ├── __init__.py
  │   │   └── golden/               # Golden test case data
  │   └── gui/
  │       ├── __init__.py
  │       └── test_config_roundtrip.py
  ├── benchmarks/
  │   ├── bench_solver.py
  │   └── results.json
  ├── configs/
  │   └── examples/
  │       ├── kevlar29_sizing.json
  │       ├── kevlar49_checkout.json
  │       └── custom_material.json
  ├── docs/
  │   ├── PRD.md
  │   ├── sprint0_tasks.md
  │   └── material_properties.md
  ├── templates/
  │   └── report.html.j2           # Jinja2 report template
  ├── .github/
  │   └── workflows/
  │       └── ci.yml
  ├── .gitignore
  ├── pyproject.toml
  ├── Makefile
  ├── README.md
  └── LICENSE
  ```

**Acceptance Criteria:**
- All directories exist with `__init__.py` where needed
- Stub files contain module-level docstrings explaining purpose
- `README.md` has project description, setup instructions, and architecture overview

---

### S0.2 — Python Project Configuration (`pyproject.toml`)

- [ ] Create `pyproject.toml` with:
  - Project metadata (name, version 0.1.0, description, Python ≥3.11)
  - Core dependencies: `numpy`, `h5py`, `jinja2`
  - GUI dependencies group: `dearpygui`
  - Compute dependencies group: `jax`, `jaxlib`, `numba`
  - Visualization dependencies group: `pyvista`, `matplotlib`
  - Export dependencies group: `weasyprint` or `pdfkit`, `ffmpeg-python`
  - Dev dependencies group: `pytest`, `pytest-benchmark`, `ruff`, `mypy`
  - Ruff configuration (line length, lint rules)
  - Mypy configuration (strict mode for `src/`)
  - Pytest configuration (test paths, markers for `slow`, `gpu`)

**Acceptance Criteria:**
- `pip install -e ".[dev,gui,compute,viz,export]"` installs all dependencies
- `pip install -e .` installs only core (numpy, h5py, jinja2) — solver works on CPU without GUI

---

### S0.3 — Development Environment Setup Script

- [ ] Create `Makefile` with targets:
  - `make setup` — creates venv, installs deps, detects hardware
  - `make test` — runs pytest (unit tests only, fast)
  - `make test-all` — runs all tests including integration and slow
  - `make lint` — runs ruff check + mypy
  - `make format` — runs ruff format
  - `make bench` — runs performance benchmarks
  - `make clean` — removes build artifacts, __pycache__, .autosave
- [ ] Create `scripts/detect_backend.py`:
  - Tries to import `jax` and checks for `metal` or `cuda` backends
  - Falls back to checking for `numba` with CUDA
  - Falls back to CPU-only (numpy)
  - Prints detected backend and capabilities
  - Exits with code 0 (GPU found) or 1 (CPU only, with warning)

**Acceptance Criteria:**
- `make setup` completes on a clean macOS machine (Apple Silicon)
- `make setup` completes on a clean Linux machine with NVIDIA GPU
- `make setup` completes on a machine with no GPU (CPU fallback, warning printed)
- `scripts/detect_backend.py` correctly identifies available hardware

---

### S0.4 — Backend Abstraction Layer (`solver/backend.py`)

- [ ] Implement `backend.py` with the following interface:
  ```python
  # Runtime backend selection
  def get_backend() -> str:
      """Returns 'jax-metal', 'jax-cuda', 'jax-cpu', 'numba', or 'numpy'"""

  # Array creation (delegates to jax.numpy or numpy)
  def zeros(shape, dtype=float): ...
  def ones(shape, dtype=float): ...
  def array(data, dtype=None): ...
  def arange(*args, **kwargs): ...

  # JIT compilation decorator
  def jit(fn): ...

  # Vectorized map
  def vmap(fn, in_axes=0, out_axes=0): ...

  # Math operations (sqrt, sum, etc.) — passthrough to backend
  def sqrt(x): ...
  def maximum(x, y): ...
  def where(condition, x, y): ...
  ```
- [ ] Write tests in `tests/unit/test_backend.py`:
  - `test_backend_detection` — detects at least 'numpy' fallback
  - `test_array_creation` — `zeros`, `ones`, `array` return correct shapes/dtypes
  - `test_jit_decorator` — JIT'd function produces same result as non-JIT'd
  - `test_vmap` — vectorized map applies function across batch dimension

**Acceptance Criteria:**
- All tests pass on CPU-only (numpy fallback)
- If JAX is installed, tests also pass with JAX backend
- `get_backend()` returns the correct string for the detected hardware

---

### S0.5 — CI Pipeline (GitHub Actions)

- [ ] Create `.github/workflows/ci.yml`:
  - Trigger: push to `main`, all PRs
  - Matrix: Python 3.11, 3.12 on ubuntu-latest
  - Steps:
    1. Checkout code
    2. Set up Python
    3. Install dependencies: `pip install -e ".[dev]"`
    4. Lint: `ruff check src/ tests/`
    5. Type check: `mypy src/`
    6. Unit tests: `pytest tests/unit/ -v`
    7. Integration tests: `pytest tests/integration/ -v` (on PR merge only)
  - Cache pip dependencies for speed

**Acceptance Criteria:**
- CI runs on every push and PR
- All linting, type checking, and unit tests pass on a fresh checkout
- CI badge added to README.md

---

### S0.6 — `.gitignore` and Initial Files

- [ ] Create `.gitignore` covering:
  - Python: `__pycache__/`, `*.pyc`, `*.egg-info/`, `dist/`, `build/`
  - Environments: `.venv/`, `venv/`, `.env`
  - IDE: `.vscode/`, `.idea/`, `*.swp`
  - OS: `.DS_Store`, `Thumbs.db`
  - Project: `.autosave/`, `benchmarks/results.json`, `*.hdf5`, `*.h5`
- [ ] Create `LICENSE` (MIT or as specified by team)
- [ ] Create initial `README.md` with:
  - Project name and one-paragraph description
  - Quick start (setup, run, test)
  - Architecture diagram (text/mermaid)
  - Link to docs/PRD.md

**Acceptance Criteria:**
- No generated files tracked by git
- README renders correctly on GitHub with all links working

---

## Definition of Done (Sprint 0)

- [ ] All tasks S0.1–S0.6 complete
- [ ] `make setup && make lint && make test` passes on clean checkout
- [ ] CI pipeline runs green on GitHub
- [ ] Repository pushed to GitHub with initial commit
- [ ] All documentation files present in `docs/`
- [ ] Team can clone, set up, and run tests within 5 minutes
