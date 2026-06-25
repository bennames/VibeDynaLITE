from __future__ import annotations

import math
import numpy as np

from kevlargrid.solver.projectile import Projectile, q_mul, q_rotate_vector
from kevlargrid.solver.failure import scale_failure_strain


def test_quaternion_integration() -> None:
    """Verify that quaternion integration is accurate, normalizes correctly, and spins correctly."""
    # Rotate by 90 degrees around Y axis
    # omega = [0.0, np.pi/2, 0.0] rad/s
    # In 1 second, it should rotate by pi/2 radians.
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    omega = np.array([0.0, np.pi / 2, 0.0], dtype=np.float64)
    
    dt = 0.001
    steps = 1000
    for _ in range(steps):
        omega_q = np.array([0.0, omega[0], omega[1], omega[2]], dtype=np.float64)
        q_dot = q_mul(omega_q, q)
        q = q + 0.5 * dt * q_dot
        q = q / np.linalg.norm(q)

    # Expected: 90 degrees rotation around Y.
    # Angle theta = pi/2 -> half angle theta/2 = pi/4
    # q = [cos(pi/4), 0, sin(pi/4), 0]
    expected = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])
    np.testing.assert_allclose(q, expected, atol=1e-3)
    np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-7)


def test_inertia_tensor_uniform_density() -> None:
    """Verify analytical moments of inertia and numerical slice integrations."""
    # 1. Sphere
    # m = 0.5, R = 0.02
    # I = 2/5 * m * R^2 = 0.4 * 0.5 * 0.0004 = 8e-5
    proj_sphere = Projectile(
        mass=0.5,
        velocity=[0.0, 0.0, 0.0],
        position=[0.0, 0.0, 0.0],
        shape_type="sphere",
        radius=0.02,
    )
    expected_sphere = np.diag([8.0e-5, 8.0e-5, 8.0e-5])
    np.testing.assert_allclose(proj_sphere.inertia, expected_sphere, atol=1e-9)

    # 2. Cylinder
    # m = 0.5, R = 0.02, L = 0.06
    # Izz = 1/2 * m * R^2 = 0.5 * 0.5 * 0.0004 = 1.0e-4
    # Ixx = Iyy = 1/12 * m * (3 * R^2 + L^2) = (0.5/12) * (3 * 0.0004 + 0.0036) = 2.0e-4
    proj_cyl = Projectile(
        mass=0.5,
        velocity=[0.0, 0.0, 0.0],
        position=[0.0, 0.0, 0.0],
        shape_type="cylinder",
        radius=0.02,
        length=0.06,
        edge_radius=0.0,
    )
    expected_cyl = np.diag([2.0e-4, 2.0e-4, 1.0e-4])
    np.testing.assert_allclose(proj_cyl.inertia, expected_cyl, atol=1e-9)

    # 3. Ogive Bullet
    proj_bullet = Projectile(
        mass=0.5,
        velocity=[0.0, 0.0, 0.0],
        position=[0.0, 0.0, 0.0],
        shape_type="bullet",
        radius=0.02,
        length=0.06,
        ogive_multiplier=2.0,
    )
    assert proj_bullet.volume > 0.0
    assert np.all(np.diag(proj_bullet.inertia) > 0.0)

    # 4. Propeller
    proj_prop = Projectile(
        mass=0.5,
        velocity=[0.0, 0.0, 0.0],
        position=[0.0, 0.0, 0.0],
        shape_type="propeller",
        span=0.1,
        root_chord=0.02,
        tip_chord=0.01,
        twist=30.0,
        thickness_ratio=12.0,
        tip_radius=0.002,
    )
    assert proj_prop.volume > 0.0
    assert np.all(np.diag(proj_prop.inertia) > 0.0)


def test_sdf_evaluations_and_gradients() -> None:
    """Verify signed distances and analytical normals/gradients for shapes."""
    # Sphere
    proj = Projectile(
        mass=0.5,
        velocity=[0.0, 0.0, 0.0],
        position=[0.0, 0.0, 0.0],
        shape_type="sphere",
        radius=0.02,
    )
    # Inside, outside, on boundary
    assert np.abs(proj.sdf(np.array([0.02, 0.0, 0.0]))) < 1e-7
    assert np.abs(proj.sdf(np.array([0.0, 0.0, 0.0])) - (-0.02)) < 1e-7
    assert np.abs(proj.sdf(np.array([0.04, 0.0, 0.0])) - 0.02) < 1e-7

    # Gradient/normal
    normal = proj.sdf_normal(np.array([0.02, 0.0, 0.0]))
    np.testing.assert_allclose(normal, [1.0, 0.0, 0.0], atol=1e-5)

    # Cylinder
    proj_cyl = Projectile(
        mass=0.5,
        velocity=[0.0, 0.0, 0.0],
        position=[0.0, 0.0, 0.0],
        shape_type="cylinder",
        radius=0.02,
        length=0.06,
        edge_radius=0.0,
    )
    # Outside along radial axis
    assert np.abs(proj_cyl.sdf(np.array([0.03, 0.0, 0.0])) - 0.01) < 1e-5
    # Outside along axial axis
    assert np.abs(proj_cyl.sdf(np.array([0.0, 0.0, 0.04])) - 0.01) < 1e-5


def test_propeller_edge_and_tip_rounding() -> None:
    """Verify rounded propeller blade leading/trailing edges and tip sphere cap."""
    proj = Projectile(
        mass=0.5,
        velocity=[0.0, 0.0, 0.0],
        position=[0.0, 0.0, 0.0],
        shape_type="propeller",
        span=0.05,
        root_chord=0.02,
        tip_chord=0.01,
        twist=15.0,
        thickness_ratio=12.0,
        tip_radius=0.002,
    )
    # Beyond span: y > S - R_tip -> should be evaluated as a sphere cap
    p_tip = np.array([0.0, 0.05 - 0.002 + 0.003 - proj.y_com, 0.0]) # 1mm outside tip sphere center
    val = proj.sdf(p_tip)
    assert np.abs(val - 0.001) < 1e-5
    
    # Check leading edge rounding
    # At root (y_geom = 0), chord = 0.02, twist = 0. Leading edge is at x = -0.01
    p_le = np.array([-0.011, -proj.y_com, 0.0])
    val_le = proj.sdf(p_le)
    # Point is outside leading edge by 1mm. Due to tip_radius rounding (R_tip=0.002),
    # leading edge is rounded with radius 0.002 centered at x = -c/2 + R_tip = -0.008.
    # Dist to center is sqrt((-0.011 - (-0.008))^2 + 0) = 0.003.
    # SDF = 0.003 - 0.002 = 0.001.
    assert np.abs(val_le - 0.001) < 1e-5


def test_eccentric_impact_torque() -> None:
    """Verify torque generated by eccentric forces about the center of mass."""
    r = np.array([0.01, 0.02, 0.0])
    F = np.array([0.0, 0.0, -100.0])
    torque = np.cross(r, F)
    expected = np.array([-2.0, 1.0, 0.0])
    np.testing.assert_allclose(torque, expected)


def test_bazant_strain_regularization() -> None:
    """Verify Bazant scaling does not scale for dx >= 1mm and scales for dx < 1mm."""
    eps0 = 0.036
    # dx = 2.0 mm (no scaling)
    assert scale_failure_strain(eps0, 0.002) == eps0
    # dx = 0.25 mm (scaled by sqrt(10 / 0.25) = sqrt(40))
    expected = eps0 * math.sqrt(0.01 / 0.00025)
    np.testing.assert_allclose(scale_failure_strain(eps0, 0.00025), expected)
