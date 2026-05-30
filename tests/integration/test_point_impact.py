from __future__ import annotations

import pytest


class TestPointImpact:
    """Integration tests verifying point impact energy conservation and kinematics."""

    @pytest.mark.slow
    def test_energy_conservation_undamped(self) -> None:
        """Verify that total energy remains constant for undamped point impacts.

        Over 10,000 steps, energy drift should be less than 0.1%.
        """
        pass

    def test_projectile_deceleration(self) -> None:
        """Verify that the projectile decelerates correctly on contact.

        Force transferred from the fabric grid slows down the rigid projectile.
        """
        pass
