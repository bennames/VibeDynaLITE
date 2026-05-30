from __future__ import annotations

import numpy as np

from kevlargrid.solver.grid import generate_rectangular_grid

# Simple mock material
MOCK_MATERIAL = {
    "tensile_modulus_gpa": 71.0,
    "areal_density_kgm2": 0.47,
    "fiber_density_gcc": 1.44,
    "shear_ratio": 0.0004,
}


class TestRectangularGridNodeCount:
    """Tests for verifying correct node counts in rectangular grids."""

    def test_rectangular_grid_node_count(self) -> None:
        """Verify correct node count for known Nx, Ny."""
        grid = generate_rectangular_grid(5, 6, 1.0, MOCK_MATERIAL)
        assert grid.n_nodes == 30
        assert grid.nodes.shape == (30, 3)


class TestRectangularGridSpringCount:
    """Tests for verifying correct spring counts in rectangular grids."""

    def test_rectangular_grid_spring_count(self) -> None:
        """Verify spring count: 4 orthogonal + 4 diagonal per interior node."""
        nx, ny = 4, 5
        grid = generate_rectangular_grid(nx, ny, 1.0, MOCK_MATERIAL)
        # Warp: (nx - 1) * ny = 3 * 5 = 15
        # Weft: nx * (ny - 1) = 4 * 4 = 16
        # Diagonal +45: (nx - 1) * (ny - 1) = 3 * 4 = 12
        # Diagonal -45: (nx - 1) * (ny - 1) = 3 * 4 = 12
        # Total: 15 + 16 + 12 + 12 = 55
        assert grid.n_springs == 55
        assert len(grid.springs) == 55


class TestSpecificGridSizes:
    """Tests for specific grid dimensions."""

    def test_grid_5x5(self) -> None:
        """Generate a 5x5 grid and verify node/spring counts."""
        grid = generate_rectangular_grid(5, 5, 0.5, MOCK_MATERIAL)
        assert grid.n_nodes == 25
        # Springs: 4 * 5 + 5 * 4 + 4 * 4 * 2 = 20 + 20 + 32 = 72
        assert grid.n_springs == 72

    def test_grid_10x10(self) -> None:
        """Generate a 10x10 grid and verify node/spring counts."""
        grid = generate_rectangular_grid(10, 10, 0.1, MOCK_MATERIAL)
        assert grid.n_nodes == 100
        # Springs: 9 * 10 + 10 * 9 + 9 * 9 * 2 = 90 + 90 + 162 = 342
        assert grid.n_springs == 342

    def test_grid_100x100(self) -> None:
        """Generate a 100x100 grid and verify node/spring counts."""
        grid = generate_rectangular_grid(100, 100, 0.01, MOCK_MATERIAL)
        assert grid.n_nodes == 10000
        # Springs: 99 * 100 * 2 + 99 * 99 * 2 = 19800 + 19602 = 39402
        assert grid.n_springs == 39402


class TestBoundaryNodes:
    """Tests for boundary node connectivity."""

    def test_boundary_node_connections(self) -> None:
        """Boundary nodes have fewer connections than interior nodes."""
        nx, ny = 4, 4
        grid = generate_rectangular_grid(nx, ny, 1.0, MOCK_MATERIAL)

        # Count connections for each node
        node_degrees = np.zeros(grid.n_nodes, dtype=int)
        for u, v in grid.springs:
            node_degrees[u] += 1
            node_degrees[v] += 1

        # Corner nodes: index (0,0), (0,3), (3,0), (3,3) -> degree 3
        corners = [0, 3, 12, 15]
        for corner in corners:
            assert node_degrees[corner] == 3

        # Interior nodes: index (1,1), (1,2), (2,1), (2,2) -> degree 8
        interiors = [5, 6, 9, 10]
        for interior in interiors:
            assert node_degrees[interior] == 8

    def test_grid_stacking_mode_b(self) -> None:
        """Verify multi-ply stacked grid properties (Mode B)."""
        nx, ny, dx = 5, 5, 0.01
        n_plies = 3
        t_ply = 0.002  # 2mm spacing

        grid = generate_rectangular_grid(nx, ny, dx, MOCK_MATERIAL, n_plies=n_plies, t_ply=t_ply)

        # Expected counts
        expected_nodes_per_layer = nx * ny
        assert grid.n_nodes == n_plies * expected_nodes_per_layer

        # Expected springs = n_plies * springs_per_layer (72 * 3 = 216)
        assert grid.n_springs == n_plies * 72

        # Check Z coordinates of stacked nodes
        for ply in range(n_plies):
            start = ply * expected_nodes_per_layer
            end = start + expected_nodes_per_layer
            z_coords = grid.nodes[start:end, 2]
            # All nodes in this layer should have identical Z coordinate equal to ply * t_ply
            np.testing.assert_allclose(z_coords, ply * t_ply)

        # Check that spring indices map within bounds
        assert np.min(grid.springs) == 0
        assert np.max(grid.springs) == grid.n_nodes - 1
