"""Compute backend abstraction layer (JAX/Numba/NumPy) for KevlarGrid.

This module provides a unified API for array creation, JIT compilation,
and math operations, dynamically selecting the best available backend.
"""

# ruff: noqa: E402
from __future__ import annotations

import os
import platform
import sys

# Set Numba threading layer to 'workqueue' to prevent OpenMP crashes on macOS ARM64
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

# Caching is disabled on macOS ARM64 to prevent SIGSEGV crashes caused by Numba cache-load bugs
NUMBA_CACHE = not (sys.platform == "darwin" and platform.machine() == "arm64")

from collections.abc import Callable
from typing import Any

import numpy as np

# Detect backend recommendation
try:
    import jax
    import jax.numpy as jnp

    # Enable x64 to prevent compilation crashes on larger grids
    jax.config.update("jax_enable_x64", True)

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

try:
    import numba

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

import importlib.util

HAS_TAICHI = importlib.util.find_spec("taichi") is not None

# Select backend with override support
env_backend = os.environ.get("KEVLARGRID_BACKEND", "").lower()
if env_backend in ("jax", "numba", "numpy", "taichi"):
    BACKEND = env_backend
else:
    if HAS_TAICHI:
        BACKEND = "taichi"
    elif HAS_NUMBA:
        BACKEND = "numba"
    elif HAS_JAX:
        BACKEND = "jax"
    else:
        BACKEND = "numpy"


def get_backend_name() -> str:
    """Get the active backend name.

    Returns:
        str: 'jax', 'numba', 'numpy', or 'taichi'.
    """
    return BACKEND


def get_active_device() -> str:
    """Get description of the current active hardware device.

    Returns:
        str: Active hardware device info.
    """
    import platform

    if BACKEND == "jax" and HAS_JAX:
        try:
            device = jax.devices()[0]
            return f"JAX GPU/Metal ({device.device_kind})"
        except Exception:
            return f"JAX CPU ({platform.machine()})"
    elif BACKEND == "taichi" and HAS_TAICHI:
        return "Taichi GPU/Metal with hardware acceleration"
    elif BACKEND == "numba" and HAS_NUMBA:
        return f"CPU ({platform.machine()}) with Numba JIT"
    else:
        return f"CPU ({platform.machine()}) with NumPy fallback"


def jit(fn: Callable[..., Any] | None = None, **kwargs: Any) -> Callable[..., Any]:
    """Decorator to JIT-compile a function using the active backend.

    Args:
        fn: Function to compile (if decorated without arguments).
        kwargs: Backend-specific compilation options.

    Returns:
        Callable: JIT-compiled function or a decorator.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if BACKEND == "jax" and HAS_JAX:
            jax_kwargs = {}
            if "static_argnums" in kwargs:
                jax_kwargs["static_argnums"] = kwargs["static_argnums"]
            if "static_argnames" in kwargs:
                jax_kwargs["static_argnames"] = kwargs["static_argnames"]
            return jax.jit(func, **jax_kwargs)  # type: ignore[no-any-return]
        elif BACKEND == "numba" and HAS_NUMBA:
            numba_kwargs = {
                k: v for k, v in kwargs.items() if k not in ("static_argnums", "static_argnames")
            }
            if "parallel" not in numba_kwargs:
                numba_kwargs["parallel"] = True
            if "fastmath" not in numba_kwargs:
                numba_kwargs["fastmath"] = True
            return numba.jit(nopython=True, cache=NUMBA_CACHE, **numba_kwargs)(func)  # type: ignore[no-any-return]
        return func

    if fn is None:
        return decorator
    return decorator(fn)


def vmap(
    fn: Callable[..., Any], in_axes: int | tuple[int | None, ...] = 0, out_axes: int = 0
) -> Callable[..., Any]:
    """Vectorized map across batch dimension.

    Args:
        fn: Function to map.
        in_axes: Input axes alignment.
        out_axes: Output axes alignment.

    Returns:
        Callable: Vectorized function.
    """
    if BACKEND == "jax" and HAS_JAX:
        return jax.vmap(fn, in_axes=in_axes, out_axes=out_axes)  # type: ignore[no-any-return]

    # NumPy/Numba fallback loop
    def vectorized_fn(*args: Any, **kwargs: Any) -> Any:
        # Determine the batch dimension length
        batch_size = 0
        if isinstance(in_axes, int):
            batch_size = len(args[in_axes])
        elif isinstance(in_axes, tuple):
            for i, axis in enumerate(in_axes):
                if axis is not None:
                    batch_size = len(args[i])
                    break

        results = []
        for j in range(batch_size):
            item_args = []
            for i, arg in enumerate(args):
                axis = in_axes[i] if isinstance(in_axes, tuple) else in_axes
                if axis is not None:
                    item_args.append(arg[j])
                else:
                    item_args.append(arg)
            results.append(fn(*item_args, **kwargs))
        return np.array(results)

    return vectorized_fn


# Define JIT-compiled subroutines under Numba so they resolve at typing time.
if HAS_NUMBA:

    @numba.njit(cache=NUMBA_CACHE)
    def numba_scatter_add(
        target: np.ndarray, indices: np.ndarray, values: np.ndarray
    ) -> np.ndarray:
        for i in range(len(indices)):
            target[indices[i]] += values[i]
        return target

    @numba.njit(cache=NUMBA_CACHE, parallel=True, fastmath=True)
    def numba_stack_z(f_mag: np.ndarray) -> np.ndarray:
        res = np.zeros((len(f_mag), 3), dtype=f_mag.dtype)
        for i in numba.prange(len(f_mag)):
            res[i, 2] = f_mag[i]
        return res

    @numba.njit(cache=NUMBA_CACHE, parallel=True, fastmath=True)
    def numba_clamp_boundary(forces: np.ndarray, mask: np.ndarray) -> np.ndarray:
        for i in numba.prange(len(mask)):
            if mask[i]:
                forces[i, 0] = 0.0
                forces[i, 1] = 0.0
                forces[i, 2] = 0.0
        return forces
else:
    numba_scatter_add = None
    numba_stack_z = None
    numba_clamp_boundary = None


# Define functional wrappers for NumPy/JAX fallbacks
def py_zeros(shape: int | tuple[int, ...], dtype: Any = np.float64) -> Any:
    if BACKEND == "jax" and HAS_JAX:
        return jnp.zeros(shape, dtype=dtype)
    return np.zeros(shape, dtype=dtype)


def py_ones(shape: int | tuple[int, ...], dtype: Any = np.float64) -> Any:
    if BACKEND == "jax" and HAS_JAX:
        return jnp.ones(shape, dtype=dtype)
    return np.ones(shape, dtype=dtype)


def py_array(data: Any, dtype: Any = None) -> Any:
    if BACKEND == "jax" and HAS_JAX:
        return jnp.array(data, dtype=dtype)
    return np.array(data, dtype=dtype)


def py_sqrt(x: Any) -> Any:
    if BACKEND == "jax" and HAS_JAX:
        return jnp.sqrt(x)
    return np.sqrt(x)


def py_maximum(x: Any, y: Any) -> Any:
    if BACKEND == "jax" and HAS_JAX:
        return jnp.maximum(x, y)
    return np.maximum(x, y)


def py_minimum(x: Any, y: Any) -> Any:
    if BACKEND == "jax" and HAS_JAX:
        return jnp.minimum(x, y)
    return np.minimum(x, y)


def py_where(condition: Any, x: Any, y: Any) -> Any:
    if BACKEND == "jax" and HAS_JAX:
        return jnp.where(condition, x, y)
    return np.where(condition, x, y)


def py_sum(x: Any, axis: Any = None, keepdims: bool = False) -> Any:
    if BACKEND == "jax" and HAS_JAX:
        return jnp.sum(x, axis=axis, keepdims=keepdims)
    return np.sum(x, axis=axis, keepdims=keepdims)


def py_min(x: Any, axis: Any = None) -> Any:
    if BACKEND == "jax" and HAS_JAX:
        return jnp.min(x, axis=axis)
    return np.min(x, axis=axis)


def py_max(x: Any, axis: Any = None) -> Any:
    if BACKEND == "jax" and HAS_JAX:
        return jnp.max(x, axis=axis)
    return np.max(x, axis=axis)


def py_abs(x: Any) -> Any:
    if BACKEND == "jax" and HAS_JAX:
        return jnp.abs(x)
    return np.abs(x)


def py_scatter_add(target: Any, indices: Any, values: Any) -> Any:
    if BACKEND == "jax" and HAS_JAX:
        return target.at[indices].add(values)
    np.add.at(target, indices, values)
    return target


def py_stack_z(f_mag: Any) -> Any:
    if BACKEND == "jax" and HAS_JAX:
        return jnp.zeros((len(f_mag), 3), dtype=f_mag.dtype).at[:, 2].set(f_mag)
    res = np.zeros((len(f_mag), 3), dtype=f_mag.dtype)
    res[:, 2] = f_mag
    return res


def py_clamp_boundary(forces: Any, mask: Any) -> Any:
    if BACKEND == "jax" and HAS_JAX:
        return jnp.where(mask[:, None], 0.0, forces)
    forces[mask] = 0.0
    return forces


# Assign active variables
if BACKEND == "numba" and HAS_NUMBA:
    zeros = np.zeros  # type: ignore[assignment]
    ones = np.ones  # type: ignore[assignment]
    array = py_array  # type: ignore[assignment]
    sqrt = np.sqrt  # type: ignore[assignment]
    maximum = np.maximum  # type: ignore[assignment]
    minimum = np.minimum  # type: ignore[assignment]
    where = np.where  # type: ignore[assignment]
    sum = np.sum  # type: ignore[assignment]
    min = np.min  # type: ignore[assignment]
    max = np.max  # type: ignore[assignment]
    abs = np.abs  # type: ignore[assignment]
    scatter_add = numba_scatter_add  # type: ignore[assignment]
    stack_z = numba_stack_z  # type: ignore[assignment]
    clamp_boundary = numba_clamp_boundary  # type: ignore[assignment]
else:
    zeros = py_zeros  # type: ignore[assignment]
    ones = py_ones  # type: ignore[assignment]
    array = py_array  # type: ignore[assignment]
    sqrt = py_sqrt  # type: ignore[assignment]
    maximum = py_maximum  # type: ignore[assignment]
    minimum = py_minimum  # type: ignore[assignment]
    where = py_where  # type: ignore[assignment]
    sum = py_sum  # type: ignore[assignment]
    min = py_min  # type: ignore[assignment]
    max = py_max  # type: ignore[assignment]
    abs = py_abs  # type: ignore[assignment]
    scatter_add = py_scatter_add  # type: ignore[assignment]
    stack_z = py_stack_z  # type: ignore[assignment]
    clamp_boundary = py_clamp_boundary  # type: ignore[assignment]
