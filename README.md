# KevlarGrid Explicit Solver v2.0

**VibeDynaLITE** — A 2D/3D explicit mass-spring simulation tool for analyzing high-energy projectile impact (~9000 J) against woven Kevlar barriers.

## Overview

This tool determines and validates the minimum number of Kevlar plies required to contain a propeller blade fragment without exceeding the fabric's ultimate strain failure threshold. It features:

- **Dual-mode solver:** Fast sizing mode (Mode A) and high-fidelity multi-ply checkout mode (Mode B)
- **Built-in material library:** Kevlar 29, 49, and KM2 with sourced properties
- **Interactive GUI:** Real-time strain telemetry, energy balance plots, and 3D impact viewport
- **GPU acceleration:** Taichi Lang (Apple Silicon Metal / CUDA / Vulkan) with parallel Numba JIT fallback
- **Comprehensive output:** HDF5/CSV export, MP4/GIF animation, HTML/PDF reports

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   DearPyGui GUI                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │  Config   │ │  Plots   │ │   3D Viewport    │ │
│  │  Panel    │ │ (strain, │ │   (impact anim)  │ │
│  │          │ │  energy) │ │                  │ │
│  └──────────┘ └──────────┘ └──────────────────┘ │
└──────────────────────┬──────────────────────────┘
                       │
              ┌────────▼────────┐
              │  Solver Engine  │
              │  (backend.py)   │
              └────────┬────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │  Taichi  │ │  Numba   │ │  NumPy   │
    │ (Metal/  │ │ (Parallel│ │ (CPU     │
    │  Vulkan) │ │  JIT)    │ │ fallback)│
    └──────────┘ └──────────┘ └──────────┘
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/<your-username>/VibeDynaLITE.git
cd VibeDynaLITE

# Set up environment
make setup

# Run tests
make test

# Launch the GUI (after Sprint 4)
python -m kevlargrid
```

## Project Structure

```
VibeDynaLITE/
├── src/kevlargrid/      # Source code
│   ├── solver/          # Physics engine (backend, grid, forces, integrator, ...)
│   ├── gui/             # DearPyGui application
│   ├── materials/       # Built-in material library
│   ├── io/              # Config, results, export, reports
│   └── utils/           # Unit conversion helpers
├── tests/               # Unit, integration, regression, GUI tests
├── benchmarks/          # Performance benchmark suite
├── configs/examples/    # Example configuration files
├── docs/                # Documentation (PRD, sprint tasks, material data)
└── templates/           # Jinja2 report templates
```

## Documentation

- [Product Requirements Document (PRD)](docs/PRD.md)
- [Sprint 0 Task List](docs/sprint0_tasks.md)
- [Material Properties Reference](docs/material_properties.md)

## Tech Stack

- **Python 3.11+**
- **Taichi Lang** — GPU-accelerated mass-spring execution (Apple Silicon Metal / CUDA / Vulkan)
- **Numba** — Parallel JIT compilation CPU solver
- **DearPyGui** — Desktop GUI with real-time plots and 3D rendering
- **PyVista** — Offline 3D visualization and video export
- **HDF5 / h5py** — Large result dataset storage

## License

MIT
