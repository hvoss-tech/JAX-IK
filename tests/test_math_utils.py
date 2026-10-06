import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_ik.helper import get_node_transform, quaternion_to_matrix
from jax_ik.ik import (
    matrix_to_euler_xyz,
    resample_frames,
    tf_euler_to_matrix,
    tf_matrix_to_euler,
    tf_rotation_matrix_from_axis_angle,
)
from jax_ik.objectives import _safe_arccos, _safe_norm, get_config


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


RNG_ANGLES = np.random.default_rng(0).uniform(-1.4, 1.4, (20, 3)).astype(np.float32)


# --- tf_euler_to_matrix ---------------------------------------------------


@pytest.mark.parametrize("angles", RNG_ANGLES)
def test_euler_to_matrix_is_rz_ry_rx(angles):
    M = np.asarray(tf_euler_to_matrix(jnp.asarray(angles)))
    expected = _rz(angles[2]) @ _ry(angles[1]) @ _rx(angles[0])
    assert M.shape == (4, 4)
    assert np.allclose(M[:3, :3], expected, atol=1e-5)
    assert np.allclose(M[3], [0, 0, 0, 1])
    assert np.allclose(M[:3, 3], 0)


def test_euler_to_matrix_zero_is_identity():
    assert np.allclose(tf_euler_to_matrix(jnp.zeros(3)), np.eye(4))


@pytest.mark.parametrize("angles", RNG_ANGLES[:5])
def test_euler_to_matrix_is_proper_rotation(angles):
    R = np.asarray(tf_euler_to_matrix(jnp.asarray(angles)))[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-5)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-5)


# --- tf_matrix_to_euler / matrix_to_euler_xyz ------------------------------


@pytest.mark.parametrize("angles", RNG_ANGLES)
def test_tf_matrix_to_euler_roundtrip(angles):
    back = np.asarray(tf_matrix_to_euler(tf_euler_to_matrix(jnp.asarray(angles))))
    assert np.allclose(back, angles, atol=1e-4)


@pytest.mark.parametrize("angles", RNG_ANGLES)
def test_matrix_to_euler_xyz_roundtrip_4x4_and_3x3(angles):
    M = np.asarray(tf_euler_to_matrix(jnp.asarray(angles)))
    assert np.allclose(matrix_to_euler_xyz(M), angles, atol=1e-4)
    assert np.allclose(matrix_to_euler_xyz(M[:3, :3]), angles, atol=1e-4)


@pytest.mark.parametrize("pitch", [np.pi / 2, -np.pi / 2])
def test_matrix_to_euler_xyz_gimbal_lock_gives_same_rotation(pitch):
    M = np.asarray(tf_euler_to_matrix(jnp.asarray([0.3, pitch, 0.2], jnp.float32)))
    angles = matrix_to_euler_xyz(M)
    assert angles[2] == 0.0  # singular branch pins z
    back = np.asarray(tf_euler_to_matrix(jnp.asarray(angles, jnp.float32)))
    assert np.allclose(back, M, atol=1e-5)


# --- tf_rotation_matrix_from_axis_angle -----------------------------------


@pytest.mark.parametrize(
    "axis, ref",
    [([1.0, 0.0, 0.0], _rx), ([0.0, 1.0, 0.0], _ry), ([0.0, 0.0, 1.0], _rz)],
)
@pytest.mark.parametrize("angle", [-2.0, -0.3, 0.0, 0.7, np.pi])
def test_axis_angle_matches_principal_rotations(axis, ref, angle):
    M = np.asarray(tf_rotation_matrix_from_axis_angle(jnp.asarray(axis), angle))
    assert np.allclose(M[:3, :3], ref(angle), atol=1e-5)
    assert np.allclose(M[:3, 3], 0) and np.allclose(M[3], [0, 0, 0, 1])


def test_axis_angle_keeps_axis_fixed_and_negative_axis_reverses():
    axis = np.array([1.0, 2.0, -0.5])
    axis /= np.linalg.norm(axis)
    M = np.asarray(tf_rotation_matrix_from_axis_angle(jnp.asarray(axis), 0.9))
    assert np.allclose(M[:3, :3] @ axis, axis, atol=1e-5)
    M_neg = np.asarray(tf_rotation_matrix_from_axis_angle(jnp.asarray(-axis), -0.9))
    assert np.allclose(M, M_neg, atol=1e-5)


def test_axis_angle_is_differentiable():
    f = lambda a: tf_rotation_matrix_from_axis_angle(jnp.array([0.0, 0.0, 1.0]), a)[0, 1]
    # d/da (-sin a) = -cos a
    assert float(jax.grad(f)(0.4)) == pytest.approx(-np.cos(0.4), abs=1e-5)


# --- resample_frames -------------------------------------------------------


def test_resample_same_length_returns_copy():
    data = np.arange(6, dtype=np.float32).reshape(3, 2)
    out = resample_frames(data, 3)
    assert np.array_equal(out, data)
    out[0, 0] = 99
    assert data[0, 0] == 0


def test_resample_upsample_is_linear_and_keeps_endpoints():
    data = np.array([[0.0, 10.0], [1.0, 20.0]], dtype=np.float32)
    out = resample_frames(data, 5)
    assert out.shape == (5, 2) and out.dtype == np.float32
    assert np.allclose(out[:, 0], [0, 0.25, 0.5, 0.75, 1.0])
    assert np.allclose(out[:, 1], [10, 12.5, 15, 17.5, 20])


def test_resample_downsample_keeps_endpoints():
    data = np.linspace(0, 1, 11)[:, None]
    out = resample_frames(data, 3)
    assert np.allclose(out[:, 0], [0.0, 0.5, 1.0])


def test_resample_edge_lengths():
    single = np.array([[1.0, 2.0]])
    assert np.allclose(resample_frames(single, 3), [[1, 2]] * 3)
    assert np.allclose(resample_frames(np.array([[1.0], [2.0], [3.0]]), 1), [[1.0]])
    assert resample_frames(np.array([[1.0], [2.0]]), 0).shape == (0, 1)


# --- quaternion / glTF node transforms -------------------------------------


def test_quaternion_identity_and_90_degrees_about_z():
    assert np.allclose(quaternion_to_matrix(np.array([0, 0, 0, 1.0])), np.eye(4))
    s = np.sqrt(0.5)
    M = quaternion_to_matrix(np.array([0, 0, s, s]))
    assert np.allclose(M[:3, :3], _rz(np.pi / 2), atol=1e-6)


def test_quaternion_random_unit_is_rotation():
    rng = np.random.default_rng(3)
    for _ in range(10):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        R = quaternion_to_matrix(q)[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-5)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-5)
        # q and -q are the same rotation
        assert np.allclose(R, quaternion_to_matrix(-q)[:3, :3], atol=1e-6)


class _Node:
    def __init__(self, matrix=None, translation=None, rotation=None, scale=None):
        self.matrix = matrix
        self.translation = translation
        self.rotation = rotation
        self.scale = scale


def test_node_transform_defaults_to_identity():
    assert np.allclose(get_node_transform(_Node()), np.eye(4))


def test_node_transform_trs_order():
    s = np.sqrt(0.5)
    node = _Node(translation=[1, 2, 3], rotation=[0, 0, s, s], scale=[2, 2, 2])
    M = get_node_transform(node)
    # T @ R @ S applied to (1,0,0): scaled to 2, rotated to +y, translated
    p = M @ np.array([1.0, 0, 0, 1])
    assert np.allclose(p[:3], [1, 4, 3], atol=1e-5)


def test_node_transform_matrix_is_column_major():
    M = np.eye(4)
    M[:3, 3] = [5, 6, 7]
    node = _Node(matrix=list(M.T.ravel()))  # glTF stores column-major
    assert np.allclose(get_node_transform(node), M)


def test_node_transform_all_zero_matrix_falls_back_to_trs():
    node = _Node(matrix=[0.0] * 16, translation=[1, 0, 0])
    assert np.allclose(get_node_transform(node)[:3, 3], [1, 0, 0])


# --- safe math in objectives ----------------------------------------------


def test_safe_norm_value_and_finite_grad_at_zero():
    assert float(_safe_norm(jnp.array([3.0, 4.0]))) == pytest.approx(5.0, abs=1e-5)
    g = jax.grad(lambda x: _safe_norm(x))(jnp.zeros(3))
    assert np.all(np.isfinite(g))


def test_safe_norm_axis():
    out = _safe_norm(jnp.array([[3.0, 4.0], [0.0, 0.0]]), axis=1)
    assert np.allclose(out, [5.0, 1e-6], atol=1e-5)


@pytest.mark.parametrize("x", [-1.0, 1.0, -2.0, 2.0])
def test_safe_arccos_clips_and_has_finite_grad(x):
    assert np.isfinite(float(_safe_arccos(jnp.float32(x))))
    assert np.isfinite(float(jax.grad(_safe_arccos)(jnp.float32(x))))


def test_safe_arccos_matches_arccos_inside_domain():
    xs = jnp.linspace(-0.9, 0.9, 7)
    assert np.allclose(_safe_arccos(xs), np.arccos(np.asarray(xs)), atol=1e-5)


def test_get_config():
    v = jnp.arange(3.0)
    assert get_config(v) is v
    traj = jnp.arange(6.0).reshape(2, 3)
    assert np.allclose(get_config(traj), [3, 4, 5])
