#!/usr/bin/env python3
"""Hardware and compute backend detection script for KevlarGrid.

This script checks for the availability of high-performance compute libraries
(JAX, Numba) and GPU acceleration (CUDA, Metal), providing a clear summary
and choosing the optimal default backend.
"""

from __future__ import annotations

import sys


def detect_backend() -> dict[str, str | bool]:
    """Detect available compute backends and hardware accelerators.

    Returns:
        dict: Detection details including available libraries and the recommended backend.
    """
    backends = {
        "jax": False,
        "numba": False,
        "numpy": True,  # NumPy is always available
        "cuda": False,
        "metal": False,
    }

    details = []

    # Check JAX
    try:
        import jax

        backends["jax"] = True
        jax_devices = jax.devices()
        device_types = {d.device_kind for d in jax_devices}
        details.append(f"JAX: Available (Devices: {', '.join(device_types)})")

        for device in jax_devices:
            kind = device.device_kind.lower()
            if "gpu" in kind or "cuda" in kind:
                backends["cuda"] = True
            elif "metal" in kind or "mps" in kind or "tpu" in kind:
                backends["metal"] = True
    except ImportError:
        details.append("JAX: Not installed")

    # Check Numba
    try:
        import numba
        from numba import cuda

        backends["numba"] = True
        cuda_avail = cuda.is_available()
        backends["cuda"] = backends["cuda"] or cuda_avail
        numba_details = "Numba: Available"
        if cuda_avail:
            numba_details += " (CUDA GPU supported)"
        details.append(numba_details)
    except ImportError:
        details.append("Numba: Not installed")

    # Select recommended backend
    if backends["jax"]:
        recommended = "jax"
    elif backends["numba"]:
        recommended = "numba"
    else:
        recommended = "numpy"

    return {
        "recommended": recommended,
        "jax_available": backends["jax"],
        "numba_available": backends["numba"],
        "numpy_available": backends["numpy"],
        "cuda_available": backends["cuda"],
        "metal_available": backends["metal"],
        "summary": " | ".join(details),
    }


def main() -> int:
    """Run detection and print results."""
    print("==================================================")
    print(" KevlarGrid Explicit Solver - Backend Detection ")
    print("==================================================")

    result = detect_backend()

    print(f"Status:      {result['summary']}")
    print(f"CUDA GPU:    {'YES' if result['cuda_available'] else 'NO'}")
    print(f"Apple Metal: {'YES' if result['metal_available'] else 'NO'}")
    print("--------------------------------------------------")
    print(f"RECOMMENDED BACKEND: \033[1;32m{result['recommended'].upper()}\033[0m")
    print("==================================================")

    return 0


if __name__ == "__main__":
    sys.exit(main())
