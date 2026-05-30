from __future__ import annotations

import numpy as np

from kevlargrid.solver import backend


def test_backend_detection() -> None:
    """Verify that backend detection returns a valid name."""
    name = backend.get_backend_name()
    assert name in ("jax", "numba", "numpy")


def test_array_creation_zeros() -> None:
    """Verify zeros array creation has correct shape and type."""
    arr = backend.zeros((3, 4))
    assert arr.shape == (3, 4)
    assert np.all(arr == 0.0)


def test_array_creation_ones() -> None:
    """Verify ones array creation has correct shape and type."""
    arr = backend.ones((2, 2))
    assert arr.shape == (2, 2)
    assert np.all(arr == 1.0)


def test_array_creation() -> None:
    """Verify array wrapping works correctly."""
    data = [1.0, 2.0, 3.0]
    arr = backend.array(data)
    assert arr.shape == (3,)
    assert np.allclose(arr, np.array(data))


def test_jit_decorator() -> None:
    """Verify JIT decorator passes results through correctly."""

    @backend.jit
    def add(x, y):
        return x + y

    assert add(2.0, 3.0) == 5.0


def test_vmap() -> None:
    """Verify vectorized map utility works."""

    def square(x):
        return x * x

    v_square = backend.vmap(square)
    arr = backend.array([1.0, 2.0, 3.0])
    res = v_square(arr)
    assert np.allclose(res, np.array([1.0, 4.0, 9.0]))


def test_sqrt() -> None:
    """Verify element-wise square root works."""
    arr = backend.array([4.0, 9.0, 16.0])
    res = backend.sqrt(arr)
    assert np.allclose(res, np.array([2.0, 3.0, 4.0]))


def test_maximum() -> None:
    """Verify element-wise maximum works."""
    x = backend.array([1.0, 5.0, 2.0])
    y = backend.array([3.0, 2.0, 4.0])
    res = backend.maximum(x, y)
    assert np.allclose(res, np.array([3.0, 5.0, 4.0]))


def test_where() -> None:
    """Verify element-wise conditional selection works."""
    cond = backend.array([True, False, True])
    x = backend.array([10.0, 20.0, 30.0])
    y = backend.array([1.0, 2.0, 3.0])
    res = backend.where(cond, x, y)
    assert np.allclose(res, np.array([10.0, 2.0, 30.0]))
