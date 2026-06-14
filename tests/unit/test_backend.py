from __future__ import annotations

import numpy as np

from kevlargrid.solver import backend


def test_backend_detection() -> None:
    """Verify that backend detection returns a valid name."""
    name = backend.get_backend_name()
    assert name in ("jax", "numba", "numpy", "taichi")


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
    data = np.array([1.0, 2.0, 3.0])
    arr = backend.array(data)
    assert arr.shape == (3,)
    assert np.allclose(arr, data)


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
    arr = backend.array(np.array([1.0, 2.0, 3.0]))
    res = v_square(arr)
    assert np.allclose(res, np.array([1.0, 4.0, 9.0]))


def test_sqrt() -> None:
    """Verify element-wise square root works."""
    arr = backend.array(np.array([4.0, 9.0, 16.0]))
    res = backend.sqrt(arr)
    assert np.allclose(res, np.array([2.0, 3.0, 4.0]))


def test_maximum() -> None:
    """Verify element-wise maximum works."""
    x = backend.array(np.array([1.0, 5.0, 2.0]))
    y = backend.array(np.array([3.0, 2.0, 4.0]))
    res = backend.maximum(x, y)
    assert np.allclose(res, np.array([3.0, 5.0, 4.0]))


def test_where() -> None:
    """Verify element-wise conditional selection works."""
    cond = backend.array(np.array([True, False, True]))
    x = backend.array(np.array([10.0, 20.0, 30.0]))
    y = backend.array(np.array([1.0, 2.0, 3.0]))
    res = backend.where(cond, x, y)
    assert np.allclose(res, np.array([10.0, 2.0, 30.0]))


def test_sum() -> None:
    """Verify sum wrapper works."""
    arr = backend.array(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert np.allclose(backend.sum(arr), 10.0)
    assert np.allclose(backend.sum(arr, axis=0), np.array([4.0, 6.0]))
    assert np.allclose(backend.sum(arr, axis=1), np.array([3.0, 7.0]))


def test_min_max_abs() -> None:
    """Verify min, max, and abs wrappers work."""
    arr = backend.array(np.array([-5.0, 2.0, -1.0]))
    assert np.allclose(backend.min(arr), -5.0)
    assert np.allclose(backend.max(arr), 2.0)
    assert np.allclose(backend.abs(arr), np.array([5.0, 2.0, 1.0]))


def test_scatter_add() -> None:
    """Verify scatter_add wrapper works."""
    target = backend.zeros((5, 3))
    indices = backend.array(np.array([1, 3, 1], dtype=np.int32))
    values = backend.array(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.5, 0.5, 0.5]]))
    res = backend.scatter_add(target, indices, values)
    # Row 1 should have values[0] + values[2] = [1.5, 2.5, 3.5]
    # Row 3 should have values[1] = [4.0, 5.0, 6.0]
    # Others should be [0.0, 0.0, 0.0]
    expected = np.zeros((5, 3))
    expected[1] = [1.5, 2.5, 3.5]
    expected[3] = [4.0, 5.0, 6.0]
    assert np.allclose(res, expected)


def test_stack_z() -> None:
    """Verify stack_z wrapper works."""
    f_mag = backend.array(np.array([2.5, -1.2, 0.5]))
    res = backend.stack_z(f_mag)
    expected = np.zeros((3, 3))
    expected[:, 2] = [2.5, -1.2, 0.5]
    assert np.allclose(res, expected)
