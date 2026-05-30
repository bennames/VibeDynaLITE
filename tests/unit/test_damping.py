from __future__ import annotations


class TestDamping:
    """Tests for viscous and Rayleigh damping models."""

    def test_viscous_damping_opposes_velocity(self) -> None:
        """Verify that the damping force vector directly opposes the velocity vector.

        F_damp = -c * v.
        """
        pass

    def test_viscous_damping_magnitude(self) -> None:
        """Verify the magnitude of the damping force for known coefficient and velocity.

        ||F_damp|| = c * ||v||.
        """
        pass

    def test_zero_velocity_zero_damping(self) -> None:
        """Verify that zero velocity results in zero damping force.

        If a node is stationary, there is no viscous resistance.
        """
        pass
