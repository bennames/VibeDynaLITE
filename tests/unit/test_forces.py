from __future__ import annotations


class TestSpringForce:
    """Tests for spring force computation."""

    def test_single_spring_10pct_stretch(self) -> None:
        """Spring stretched 10% returns F = k * 0.1 * L_rest.

        For a spring with stiffness k and rest length L_rest,
        a 10% elongation should produce a force of k * 0.1 * L_rest.
        """
        pass

    def test_force_direction(self) -> None:
        """Force is directed along the spring axis.

        The force vector should be parallel to the vector connecting
        the two end nodes of the spring.
        """
        pass

    def test_zero_displacement_zero_force(self) -> None:
        """No displacement produces no force.

        When the spring is at its rest length, the force should be zero.
        """
        pass


class TestTensionOnly:
    """Tests for tension-only spring behavior."""

    def test_tension_only_compression(self) -> None:
        """Compression returns zero force for tension-only springs.

        When a tension-only spring is compressed (L < L_rest),
        it should produce zero force.
        """
        pass


class TestStrainComputation:
    """Tests for engineering strain calculation."""

    def test_strain_computation(self) -> None:
        """Strain = (L - L0) / L0.

        Engineering strain is computed as the change in length
        divided by the original rest length.
        """
        pass
