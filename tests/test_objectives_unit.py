"""Exact-value tests for every objective, using conftest.StubFK whose bone
"b" sits at angles[:3] (translation) with its tail 1 unit along local +y.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.tree_util import tree_flatten, tree_unflatten

from jax_ik.objectives import (
    BoneDirectionObjective,
    BoneRelativeLookObj,
    BoneZeroRotationObj,
    CombinedDerivativeObj,
    DerivativeObj,
    DistanceObjTraj,
    EndEffectorOrientationObj,
    EqualDistanceObj,
    InitPoseObj,
    SDFCollisionPenaltyObj,
    SDFSelfCollisionPenaltyObj,
    SphereCollisionPenaltyObjTraj,
)


def f32(x):
    return jnp.asarray(x, jnp.float32)


def _sdf(value, size=5, spacing=0.5, origin=-1.0):
    return {
        "grid": jnp.full((size, size, size), value, jnp.float32),
        "origin": jnp.full(3, origin, jnp.float32),
        "spacing": jnp.float32(spacing),
    }


# --- DistanceObjTraj -------------------------------------------------------


def test_distance_zero_at_target(stub_fk):
    obj = DistanceObjTraj("b", [1.0, 2.0, 3.0], use_head=True)
    assert float(obj(f32([1, 2, 3]), stub_fk)) == pytest.approx(0.0)


def test_distance_is_weighted_mean_squared_error(stub_fk):
    obj = DistanceObjTraj("b", [0.0, 0.0, 0.0], use_head=True, weight=2.0)
    # diff (1, 2, 2): mean of squares = 9 / 3
    assert float(obj(f32([1, 2, 2]), stub_fk)) == pytest.approx(2.0 * 3.0)


def test_distance_head_vs_tail(stub_fk):
    target = [0.0, 1.0, 0.0]
    head = DistanceObjTraj("b", target, use_head=True)
    tail = DistanceObjTraj("b", target, use_head=False)
    cfg = f32([0, 0, 0])
    assert float(tail(cfg, stub_fk)) == pytest.approx(0.0)
    assert float(head(cfg, stub_fk)) == pytest.approx(1.0 / 3.0)


def test_distance_single_target_uses_last_frame(stub_fk):
    obj = DistanceObjTraj("b", [5.0, 0.0, 0.0], use_head=True)
    traj = f32([[0, 0, 0], [1, 0, 0], [5, 0, 0]])
    assert float(obj(traj, stub_fk)) == pytest.approx(0.0)


def test_distance_multiple_targets_spread_over_frames(stub_fk):
    # 2 targets on a 5-frame trajectory compare against frames 2 and 4
    obj = DistanceObjTraj("b", [[2.0, 0, 0], [4.0, 0, 0]], use_head=True)
    traj = f32([[i, 0, 0] for i in range(5)])
    assert float(obj(traj, stub_fk)) == pytest.approx(0.0)
    shifted = traj.at[2, 0].set(3.0)
    # one target off by 1 in x: mean over 2 points x 3 coords
    assert float(obj(shifted, stub_fk)) == pytest.approx(1.0 / 6.0)


def test_distance_more_targets_than_frames(stub_fk):
    obj = DistanceObjTraj("b", [[0, 0, 0]] * 4, use_head=True)
    assert float(obj(f32([[0, 0, 0], [0, 0, 0]]), stub_fk)) == pytest.approx(0.0)


@pytest.mark.parametrize("bad", [np.zeros((2, 2)), np.zeros(4), np.zeros((1, 2, 3))])
def test_distance_rejects_bad_target_shape(bad):
    with pytest.raises(ValueError):
        DistanceObjTraj("b", bad)


def test_distance_referenced_bones():
    assert DistanceObjTraj("b", [0, 0, 0]).referenced_bones() == ("b",)


# --- BoneRelativeLookObj --------------------------------------------------


def test_look_tail_without_modifications_is_aligned(stub_fk):
    obj = BoneRelativeLookObj("b", use_head=False, modifications=[])
    assert float(obj(f32([0, 0, 0]), stub_fk)) == pytest.approx(0.0, abs=1e-5)


def test_look_head_without_modifications_is_degenerate_but_finite(stub_fk):
    # Target == head gives a zero target vector; the safe norm keeps it finite
    obj = BoneRelativeLookObj("b", use_head=True, modifications=[])
    val = float(obj(f32([0, 0, 0]), stub_fk))
    assert val == pytest.approx((np.pi / 2) ** 2, abs=1e-3)


def test_look_perpendicular_modification_gives_45_degrees(stub_fk):
    # Tail at (0,1,0) pushed +1 in x -> target direction (1,1,0)
    obj = BoneRelativeLookObj("b", use_head=False, modifications=[(0, 1.0)], weight=3.0)
    assert float(obj(f32([0, 0, 0]), stub_fk)) == pytest.approx(
        3.0 * (np.pi / 4) ** 2, abs=1e-4
    )


def test_look_update_params(stub_fk):
    obj = BoneRelativeLookObj("b", use_head=False, modifications=[(0, 1.0)])
    obj.update_params({"modifications": [], "weight": 5.0, "unknown": 1})
    assert obj.mod_idx.size == 0
    assert float(obj.weight) == 5.0
    assert float(obj(f32([0, 0, 0]), stub_fk)) == pytest.approx(0.0, abs=1e-4)
    obj.update_params({"modifications": None})
    assert obj.mod_idx.size == 0


def test_look_uses_last_frame_only(stub_fk):
    assert BoneRelativeLookObj.LAST_FRAME_ONLY


# --- EndEffectorOrientationObj ---------------------------------------------


def test_orientation_zero_when_matching(stub_fk):
    obj = EndEffectorOrientationObj("b", np.eye(4))
    assert float(obj(f32([0, 0, 0, 0, 0, 0]), stub_fk)) == pytest.approx(0.0, abs=1e-4)


@pytest.mark.parametrize("angle", [0.3, np.pi / 2, 2.5])
def test_orientation_is_squared_geodesic_angle(stub_fk, angle):
    obj = EndEffectorOrientationObj("b", np.eye(4), weight=2.0)
    val = float(obj(f32([0, 0, 0, 0, 0, angle]), stub_fk))
    assert val == pytest.approx(2.0 * angle**2, abs=1e-3)


def test_orientation_accepts_3x3_target(stub_fk):
    obj = EndEffectorOrientationObj("b", np.eye(3))
    assert obj.target_R.shape == (3, 3)


def test_orientation_pytree_roundtrip_keeps_bone_name():
    obj = EndEffectorOrientationObj("b", np.eye(4), weight=0.5)
    leaves, treedef = tree_flatten(obj)
    back = tree_unflatten(treedef, leaves)
    assert back.bone_name == "b"
    assert float(back.weight) == 0.5


# --- DerivativeObj / CombinedDerivativeObj ----------------------------------


@pytest.mark.parametrize("order", [0, 4, -1])
def test_derivative_rejects_bad_order(order):
    with pytest.raises(ValueError):
        DerivativeObj(order, 1.0)
    with pytest.raises(ValueError):
        CombinedDerivativeObj(order)


def test_derivative_single_pose_is_zero():
    for order in (1, 2, 3):
        assert float(DerivativeObj(order, 1.0)(f32([1, 2, 3]))) == 0.0


def test_derivative_linear_trajectory():
    traj = f32([[0.0], [2.0], [4.0], [6.0]])
    assert float(DerivativeObj(1, 1.0)(traj)) == pytest.approx(4.0)
    assert float(DerivativeObj(2, 1.0)(traj)) == pytest.approx(0.0)
    assert float(DerivativeObj(3, 1.0)(traj)) == pytest.approx(0.0)


def test_derivative_quadratic_and_cubic():
    t = np.arange(5.0)
    assert float(DerivativeObj(2, 1.0)(f32(t[:, None] ** 2))) == pytest.approx(4.0)
    assert float(DerivativeObj(3, 0.5)(f32(t[:, None] ** 3))) == pytest.approx(0.5 * 36.0)


def test_derivative_too_short_for_order_is_zero():
    assert float(DerivativeObj(2, 1.0)(f32([[0.0], [1.0]]))) == 0.0
    assert float(DerivativeObj(3, 1.0)(f32([[0.0], [1.0], [5.0]]))) == 0.0


def test_derivative_next_frames_extend_trajectory():
    obj = DerivativeObj(1, 1.0, next_frames=np.array([[3.0]]))
    # diffs 1, 2 -> mean of squares 2.5
    assert float(obj(f32([[0.0], [1.0]]))) == pytest.approx(2.5)


def test_derivative_update_params_only_changes_weight():
    obj = DerivativeObj(2, 1.0)
    obj.update_params({"weight": 3.0, "order": 1})
    assert float(obj.weight) == 3.0 and obj.order == 2


def test_combined_equals_sum_of_orders():
    traj = f32(np.random.default_rng(0).normal(size=(6, 4)))
    combined = CombinedDerivativeObj(3, weights=[1.0, 2.0, 3.0])
    expected = sum(float(DerivativeObj(o, w)(traj)) for o, w in [(1, 1), (2, 2), (3, 3)])
    assert float(combined(traj)) == pytest.approx(expected, rel=1e-5)


def test_combined_single_weight_and_lengths():
    traj = f32([[0.0], [1.0], [3.0]])
    obj = CombinedDerivativeObj(3, weight=1.0)
    assert np.allclose(obj.weights, [1, 1, 1])
    # jerk needs 4 frames and is skipped
    assert float(obj(traj)) == pytest.approx(2.5 + 1.0)
    assert float(obj(f32([1.0]))) == 0.0
    assert float(obj(f32([[1.0]]))) == 0.0


def test_combined_rejects_wrong_weights_length():
    with pytest.raises(ValueError):
        CombinedDerivativeObj(2, weights=[1.0])


def test_combined_update_params():
    obj = CombinedDerivativeObj(2, weight=1.0)
    obj.update_params({"weight": 4.0})
    assert np.allclose(obj.weights, [4, 4])
    obj.update_params({"weights": [1.0, 2.0], "weight": 9.0})
    assert np.allclose(obj.weights, [1, 2])


# --- InitPoseObj -----------------------------------------------------------


def test_init_pose_first_last_full():
    init = np.zeros(2)
    traj = f32([[1.0, 1.0], [0.0, 0.0], [2.0, 2.0]])
    assert float(InitPoseObj(init)(traj)) == pytest.approx(1.0)
    assert float(InitPoseObj(init, last_position=True)(traj)) == pytest.approx(4.0)
    assert float(InitPoseObj(init, full_trajectory=True)(traj)) == pytest.approx(5.0 / 3.0)


def test_init_pose_mask_and_weight():
    obj = InitPoseObj(np.zeros(2), mask=np.array([1.0, 0.0]), weight=2.0)
    assert float(obj(f32([3.0, 100.0]))) == pytest.approx(2.0 * 9.0 / 2.0)


def test_init_pose_single_pose():
    assert float(InitPoseObj(np.ones(3))(f32([1, 1, 1]))) == pytest.approx(0.0)


# --- EqualDistanceObj -------------------------------------------------------


def test_equal_distance_zero_for_even_spacing():
    traj = f32([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    assert float(EqualDistanceObj()(traj)) == pytest.approx(0.0, abs=1e-6)


def test_equal_distance_penalizes_uneven_spacing():
    traj = f32([[0.0], [1.0], [4.0]])  # steps 1 and 3 around mean 2
    assert float(EqualDistanceObj(weight=2.0)(traj)) == pytest.approx(2.0, abs=1e-4)


def test_equal_distance_single_pose_and_static_traj():
    assert float(EqualDistanceObj()(f32([1.0, 2.0]))) == 0.0
    still = f32(np.zeros((4, 3)))
    assert float(EqualDistanceObj()(still)) == pytest.approx(0.0, abs=1e-8)
    g = jax.grad(lambda x: EqualDistanceObj()(x))(still)
    assert np.all(np.isfinite(g))


# --- SphereCollisionPenaltyObjTraj -----------------------------------------


def test_sphere_far_away_is_zero(stub_fk):
    obj = SphereCollisionPenaltyObjTraj({"center": [10, 10, 10], "radius": 0.5})
    assert float(obj(f32([1, 0, 0]), stub_fk)) == 0.0


def test_sphere_on_segment_penalized(stub_fk):
    # Segment root(0,0,0) -> b(2,0,0) passes through the sphere's center
    obj = SphereCollisionPenaltyObjTraj(
        {"center": [1, 0, 0], "radius": 0.1},
        min_clearance=0.0,
        segment_radius=0.0,
        weight=2.0,
    )
    # distance 0 -> penetration 0.1
    assert float(obj(f32([2, 0, 0]), stub_fk)) == pytest.approx(2.0 * 0.01, rel=1e-3)


def test_sphere_clearance_and_segment_radius_extend_reach(stub_fk):
    base = {"center": [1, 0.3, 0], "radius": 0.1}
    near = SphereCollisionPenaltyObjTraj(base, min_clearance=0.0, segment_radius=0.0)
    padded = SphereCollisionPenaltyObjTraj(base, min_clearance=0.15, segment_radius=0.1)
    cfg = f32([2, 0, 0])
    assert float(near(cfg, stub_fk)) == 0.0
    assert float(padded(cfg, stub_fk)) == pytest.approx(0.05**2, rel=1e-3)


def test_sphere_trajectory_is_mean_of_frames(stub_fk):
    obj = SphereCollisionPenaltyObjTraj(
        {"center": [1, 0, 0], "radius": 0.1}, min_clearance=0.0, segment_radius=0.0
    )
    traj = f32([[2, 0, 0], [0, 5, 0]])
    assert float(obj(traj, stub_fk)) == pytest.approx(0.01 / 2, rel=1e-3)


def test_sphere_needs_full_skeleton_and_update_params():
    obj = SphereCollisionPenaltyObjTraj({"center": [0, 0, 0], "radius": 1.0})
    assert obj.referenced_bones() is None
    obj.update_params({"weight": 7.0})
    assert float(obj.weight) == 7.0


# --- BoneDirectionObjective -------------------------------------------------


def test_direction_aligned_and_opposite(stub_fk):
    cfg = f32([0, 0, 0])  # bone points +y (tail - head)
    aligned = BoneDirectionObjective("b", use_head=False, directions=[[0, 1, 0]])
    opposite = BoneDirectionObjective("b", use_head=False, directions=[[0, -1, 0]])
    assert float(aligned(cfg, stub_fk)) == pytest.approx(0.0, abs=1e-5)
    # angle pi normalized by pi^2
    assert float(opposite(cfg, stub_fk)) == pytest.approx(1.0, abs=1e-3)


def test_direction_use_head_flips_bone_vector(stub_fk):
    obj = BoneDirectionObjective("b", use_head=True, directions=[[0, -1, 0]])
    assert float(obj(f32([0, 0, 0]), stub_fk)) == pytest.approx(0.0, abs=1e-5)


def test_direction_multiple_directions_are_summed(stub_fk):
    obj = BoneDirectionObjective("b", use_head=False, directions=[[0, 1, 0], [1, 0, 0]])
    # Combined direction (1,1,0): 45 degrees from +y
    assert float(obj(f32([0, 0, 0]), stub_fk)) == pytest.approx(0.25**2, abs=1e-4)


def test_direction_default_and_trajectory_mean(stub_fk):
    obj = BoneDirectionObjective("b", use_head=False, weight=2.0)
    assert np.allclose(obj.directions, [[0, 1, 0]])
    traj = f32([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, np.pi]])
    # frame 2 is rotated 180 degrees about z
    assert float(obj(traj, stub_fk)) == pytest.approx(2.0 * 0.5, abs=1e-3)


# --- BoneZeroRotationObj ---------------------------------------------------


def test_zero_rotation_value_mask_and_trajectory():
    assert float(BoneZeroRotationObj(weight=2.0)(f32([1.0, 2.0]))) == pytest.approx(5.0)
    masked = BoneZeroRotationObj(mask=np.array([0.0, 1.0]))
    assert float(masked(f32([10.0, 2.0]))) == pytest.approx(2.0)
    traj = f32([[1.0, 1.0], [3.0, 3.0]])
    assert float(BoneZeroRotationObj()(traj)) == pytest.approx(5.0)


# --- SDFCollisionPenaltyObj ------------------------------------------------


def test_sdf_collision_inside_and_outside(stub_fk):
    inside = SDFCollisionPenaltyObj("b", _sdf(-0.2), num_samples=4, weight=2.0)
    outside = SDFCollisionPenaltyObj("b", _sdf(0.3), num_samples=4)
    cfg = f32([0, 0, 0])
    assert float(inside(cfg, stub_fk)) == pytest.approx(2.0 * 0.04, rel=1e-4)
    assert float(outside(cfg, stub_fk)) == 0.0


def test_sdf_collision_outside_grid_counts_as_free(stub_fk):
    obj = SDFCollisionPenaltyObj("b", _sdf(-1.0))
    assert float(obj(f32([50, 50, 50]), stub_fk)) == 0.0


def test_sdf_collision_trajectory_and_update(stub_fk):
    obj = SDFCollisionPenaltyObj("b", _sdf(-0.2), num_samples=3)
    traj = f32([[0, 0, 0], [50, 50, 50]])
    assert float(obj(traj, stub_fk)) == pytest.approx(0.04 / 2, rel=1e-4)
    obj.update_params({"weight": 0.0})
    assert float(obj(traj, stub_fk)) == 0.0
    assert obj.referenced_bones() == ("b",)


# --- SDFSelfCollisionPenaltyObj --------------------------------------------


def test_self_collision_without_sdf_is_zero(stub_fk):
    obj = SDFSelfCollisionPenaltyObj(["b"])
    assert float(obj(f32([0, 0, 0]), stub_fk)) == 0.0
    assert float(obj(f32(np.zeros((3, 3))), stub_fk)) == 0.0


def test_self_collision_with_sdf_and_rigid_mesh(stub_fk):
    # Mesh data without skinning makes inverse skinning a no-op
    stub_fk.sdf = _sdf(-0.1)
    stub_fk.mesh_data = {"vertices": jnp.zeros((1, 4))}
    obj = SDFSelfCollisionPenaltyObj(["b"], num_samples_per_bone=3, min_dist=0.05)
    # penetration = min_dist - (-0.1)
    assert float(obj(f32([0, 0, 0]), stub_fk)) == pytest.approx(0.15**2, rel=1e-4)
    obj.update_params({"min_dist": 0.0, "weight": 2.0})
    assert float(obj(f32([0, 0, 0]), stub_fk)) == pytest.approx(2.0 * 0.01, rel=1e-4)


def test_self_collision_needs_full_skeleton():
    assert SDFSelfCollisionPenaltyObj(["b"]).referenced_bones() is None


# --- Shared behaviour: pytrees, params, gradients ---------------------------


def _all_objectives():
    return [
        DistanceObjTraj("b", [1.0, 0.5, 0.0], use_head=True),
        BoneRelativeLookObj("b", use_head=False, modifications=[(0, 0.5)]),
        EndEffectorOrientationObj("b", np.eye(4)),
        DerivativeObj(2, 1.0),
        CombinedDerivativeObj(3, weight=0.5),
        InitPoseObj(np.zeros(6), last_position=True),
        EqualDistanceObj(),
        SphereCollisionPenaltyObjTraj({"center": [0.5, 0, 0], "radius": 0.2}),
        BoneDirectionObjective("b", directions=[[1, 0, 0]]),
        BoneZeroRotationObj(),
        SDFCollisionPenaltyObj("b", _sdf(-0.1)),
        SDFSelfCollisionPenaltyObj(["b"]),
    ]


@pytest.mark.parametrize("obj", _all_objectives(), ids=lambda o: type(o).__name__)
def test_objective_pytree_roundtrip_gives_same_value(obj, stub_fk):
    traj = f32(np.random.default_rng(1).uniform(-0.5, 0.5, (4, 6)))
    leaves, treedef = tree_flatten(obj)
    back = tree_unflatten(treedef, leaves)
    assert float(back(traj, stub_fk)) == pytest.approx(float(obj(traj, stub_fk)), rel=1e-5)


@pytest.mark.parametrize("obj", _all_objectives(), ids=lambda o: type(o).__name__)
def test_objective_can_be_passed_through_jit(obj, stub_fk):
    traj = f32(np.random.default_rng(2).uniform(-0.5, 0.5, (4, 6)))
    jitted = jax.jit(lambda o, x: o(x, stub_fk))
    assert float(jitted(obj, traj)) == pytest.approx(float(obj(traj, stub_fk)), rel=1e-4)


@pytest.mark.parametrize("obj", _all_objectives(), ids=lambda o: type(o).__name__)
def test_objective_gradient_is_finite_at_zero_pose(obj, stub_fk):
    # All-equal frames hit the zero-norm / arccos(+-1) corner cases
    traj = jnp.zeros((4, 6), jnp.float32)
    g = jax.grad(lambda x: obj(x, stub_fk))(traj)
    assert np.all(np.isfinite(np.asarray(g)))


@pytest.mark.parametrize("obj", _all_objectives(), ids=lambda o: type(o).__name__)
def test_objective_params_roundtrip(obj, stub_fk):
    traj = f32(np.random.default_rng(3).uniform(-0.5, 0.5, (4, 6)))
    before = float(obj(traj, stub_fk))
    obj.update_params(obj.get_params())
    assert float(obj(traj, stub_fk)) == pytest.approx(before, rel=1e-5)


def test_weight_update_through_params_scales_loss(stub_fk):
    obj = DistanceObjTraj("b", [1.0, 0.0, 0.0], use_head=True, weight=1.0)
    cfg = f32([0, 0, 0])
    base = float(obj(cfg, stub_fk))
    obj.update_params({"weight": 3.0, "not_an_attribute": 1})
    assert float(obj(cfg, stub_fk)) == pytest.approx(3.0 * base)
    assert not hasattr(obj, "not_an_attribute")


def test_get_params_is_json_friendly():
    import json

    params = DistanceObjTraj("b", [[1.0, 2.0, 3.0]], use_head=True).get_params()
    assert params["target_points"] == [[1.0, 2.0, 3.0]]
    assert params["use_head"] is True
    json.dumps(params)


def test_int_and_bool_fields_are_static():
    obj = DerivativeObj(2, 1.0)
    leaves, treedef = tree_flatten(obj)
    # order (int) must not become a traced leaf
    assert all(not isinstance(leaf, int) for leaf in leaves)
    assert tree_unflatten(treedef, leaves).order == 2
