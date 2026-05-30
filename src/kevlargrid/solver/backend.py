"""Compute backend abstraction layer (JAX/Numba/NumPy) for KevlarGrid.

This module provides a unified API for array creation, JIT compilation,
and math operations, dynamically selecting the best available backend.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

# Detect backend recommendation
try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

try:
    import numba

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

import os

# Select backend with override support
env_backend = os.environ.get("KEVLARGRID_BACKEND", "").lower()
if env_backend in ("jax", "numba", "numpy"):
    BACKEND = env_backend
else:
    if HAS_NUMBA:
        BACKEND = "numba"
    elif HAS_JAX:
        BACKEND = "jax"
    else:
        BACKEND = "numpy"


def get_backend_name() -> str:
    """Get the active backend name.

    Returns:
        str: 'jax', 'numba', or 'numpy'.
    """
    return BACKEND


def zeros(shape: int | tuple[int, ...], dtype: Any = np.float64) -> Any:
    """Create array of zeros.

    Args:
        shape: Array shape.
        dtype: Data type.

    Returns:
        array: Backend array.
    """
    if BACKEND == "jax" and HAS_JAX:
        return jnp.zeros(shape, dtype=dtype)
    return np.zeros(shape, dtype=dtype)


def ones(shape: int | tuple[int, ...], dtype: Any = np.float64) -> Any:
    """Create array of ones.

    Args:
        shape: Array shape.
        dtype: Data type.

    Returns:
        array: Backend array.
    """
    if BACKEND == "jax" and HAS_JAX:
        return jnp.ones(shape, dtype=dtype)
    return np.ones(shape, dtype=dtype)


def array(data: Any, dtype: Any = None) -> Any:
    """Create backend array from data.

    Args:
        data: Input data.
        dtype: Data type.

    Returns:
        array: Backend array.
    """
    if BACKEND == "jax" and HAS_JAX:
        return jnp.array(data, dtype=dtype)
    return np.array(data, dtype=dtype)


def jit(fn: Callable[..., Any], **kwargs: Any) -> Callable[..., Any]:
    """Decorator to JIT-compile a function using the active backend.

    Args:
        fn: Function to compile.
        kwargs: Backend-specific compilation options.

    Returns:
        Callable: JIT-compiled function.
    """
    if BACKEND == "jax" and HAS_JAX:
        return jax.jit(fn, **kwargs)  # type: ignore[no-any-return]
    elif BACKEND == "numba" and HAS_NUMBA:
        return numba.jit(nopython=True, cache=True, **kwargs)(fn)  # type: ignore[no-any-return]
    return fn


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


def sqrt(x: Any) -> Any:
    """Element-wise square root.

    Args:
        x: Input array.

    Returns:
        array: Square root of x.
    """
    if BACKEND == "jax" and HAS_JAX:
        return jnp.sqrt(x)
    return np.sqrt(x)


def maximum(x: Any, y: Any) -> Any:
    """Element-wise maximum of array elements.

    Args:
        x: First input.
        y: Second input.

    Returns:
        array: Maximum of x and y.
    """
    if BACKEND == "jax" and HAS_JAX:
        return jnp.maximum(x, y)
    return np.maximum(x, y)


def where(condition: Any, x: Any, y: Any) -> Any:
    """Return elements chosen from x or y depending on condition.

    Args:
        condition: Boolean condition array.
        x: Values if True.
        y: Values if False.

    Returns:
        array: Chosen elements.
    """
    if BACKEND == "jax" and HAS_JAX:
        return jnp.where(condition, x, y)
    return np.where(condition, x, y)
