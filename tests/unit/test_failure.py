from __future__ import annotations

import numpy as np

from kevlargrid.solver.failure import check_failures
from kevlargrid.solver.forces import compute_spring_forces


class TestFailureCriteria:
    """Tests for spring failure criteria."""

    def test_spring_fails_at_threshold(self) -> None:
        """Verify that a spring fails when its strain exceeds the failure strain."""
        strains = np.array([0.04])
        failed = np.array([False])
        epsilon_fail = 0.036

        res = check_failures(strains, failed, epsilon_fail)
        assert res[0]
        assert failed[0]

    def test_spring_below_threshold_survives(self) -> None:
        """Verify that a spring survives when its strain is below failure strain."""
        strains = np.array([0.03])
        failed = np.array([False])
        epsilon_fail = 0.036

        res = check_failures(strains, failed, epsilon_fail)
        assert not res[0]
        assert not failed[0]

    def test_failed_spring_stays_failed(self) -> None:
        """Verify that a failed spring remains failed in subsequent steps."""
        strains = np.array([0.01])
        failed = np.array([True])
        epsilon_fail = 0.036

        res = check_failures(strains, failed, epsilon_fail)
        assert res[0]
        assert failed[0]

    def test_failed_spring_zero_force(self) -> None:
        """Verify that a failed spring produces zero force."""
        positions = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]])
        springs = np.array([[0, 1]], dtype=np.int32)
        stiffnesses = np.array([100.0])
        rest_lengths = np.array([1.0])
        failed = np.array([True])  # Pre-failed

        forces = compute_spring_forces(positions, springs, stiffnesses, rest_lengths, failed)
        np.testing.assert_allclose(forces, np.zeros_like(positions))
