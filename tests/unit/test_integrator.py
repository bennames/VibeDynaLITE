from __future__ import annotations

import numpy as np

from kevlargrid.solver.integrator import leapfrog_step


class TestTimeIntegration:
    """Tests for leapfrog time integration."""

    def test_constant_force_linear_acceleration(self) -> None:
        """Verify that a node under constant force accelerates linearly."""
        positions = np.array([[0.0, 0.0, 0.0]])
        # Velocity at t - dt/2 is zero
        velocities = np.array([[0.0, 0.0, 0.0]])
        # Mass = 2.0 kg, Force = 10.0 N along X -> acceleration = 5.0 m/s^2
        forces = np.array([[10.0, 0.0, 0.0]])
        masses = np.array([2.0])
        dt = 0.1

        x_new, v_new = leapfrog_step(positions, velocities, forces, masses, dt)
        # v(t + dt/2) = v(t - dt/2) + a * dt = 0 + 5 * 0.1 = 0.5 m/s
        # x(t + dt) = x(t) + v(t + dt/2) * dt = 0 + 0.5 * 0.1 = 0.05 m
        np.testing.assert_allclose(v_new[0], np.array([0.5, 0.0, 0.0]))
        np.testing.assert_allclose(x_new[0], np.array([0.05, 0.0, 0.0]))

    def test_zero_force_constant_velocity(self) -> None:
        """Verify that a node with zero external force maintains constant velocity."""
        positions = np.array([[1.0, 2.0, 3.0]])
        velocities = np.array([[2.0, 0.0, -1.0]])
        forces = np.array([[0.0, 0.0, 0.0]])
        masses = np.array([10.0])
        dt = 0.2

        x_new, v_new = leapfrog_step(positions, velocities, forces, masses, dt)
        # v(t + dt/2) = v(t - dt/2) = [2.0, 0.0, -1.0]
        # x(t + dt) = x(t) + v * dt = [1.0, 2.0, 3.0] + [0.4, 0.0, -0.2] = [1.4, 2.0, 2.8]
        np.testing.assert_allclose(v_new[0], np.array([2.0, 0.0, -1.0]))
        np.testing.assert_allclose(x_new[0], np.array([1.4, 2.0, 2.8]))

    def test_leapfrog_position_update(self) -> None:
        """Verify that the leapfrog algorithm updates position correctly."""
        positions = np.array([[0.0, 0.0, 0.0]])
        velocities = np.array([[1.0, 0.0, 0.0]])
        forces = np.array([[0.0, 0.0, 0.0]])
        masses = np.array([1.0])
        dt = 1.0

        x_new, _ = leapfrog_step(positions, velocities, forces, masses, dt)
        # x_new = 0 + 1 * 1 = 1.0
        assert x_new[0, 0] == 1.0
