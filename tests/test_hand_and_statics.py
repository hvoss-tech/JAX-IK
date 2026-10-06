import numpy as np
import pytest

from jax_ik.hand_specification import HandSpecification, getBoneFillArray, getPoseMask
from jax_ik.objectives import BoneDirectionObjective, InitPoseObj, SDFSelfCollisionPenaltyObj
from jax_ik.smplx_statics import (
    bounds_dict_to_list,
    controlled_bones_both,
    controlled_bones_left,
    controlled_bones_right,
    get_flat_pose,
    get_pointing_pose,
    get_shaping_pose,
    left_arm_bounds_dict,
    left_bounds,
    mirror_limits,
    right_arm_bounds_dict,
)

# --- pose helpers -------------------------------------------------------------


def test_pose_mask_and_fill():
    pose = {"a": np.array([1.0, 2.0, 3.0]), "c": np.array([4.0, 5.0, 6.0])}
    bones = ["a", "b", "c"]
    assert np.array_equal(getPoseMask(pose, bones), [1, 1, 1, 0, 0, 0, 1, 1, 1])
    assert np.array_equal(getBoneFillArray(pose, bones), [1, 2, 3, 0, 0, 0, 4, 5, 6])


def test_pose_helpers_empty_inputs():
    assert getPoseMask({}, []).shape == (0,)
    assert np.array_equal(getBoneFillArray({}, ["a"]), [0, 0, 0])


@pytest.mark.parametrize("pose_fn", [get_pointing_pose, get_shaping_pose, get_flat_pose])
def test_hand_poses_are_mirrored(pose_fn):
    pose = pose_fn()
    lefts = [k for k in pose if k.startswith("left_")]
    assert len(lefts) == 15
    for k in lefts:
        right = pose[k.replace("left_", "right_")]
        assert np.allclose(right, pose[k] * [1, -1, -1])
        assert pose[k].shape == (3,)
        assert np.all(np.abs(pose[k]) <= np.pi)


# --- bounds -------------------------------------------------------------------


def test_mirror_limits_flips_y_and_z():
    out = mirror_limits({"left_x": ([-10, -20, -30], [1, 2, 3]), "spine": ([0] * 3, [1] * 3)})
    assert set(out) == {"right_x"}  # only left_ bones are mirrored
    assert out["right_x"] == ([-10, -2, -3], [1, 20, 30])


def test_mirrored_bounds_are_ordered():
    for lower, upper in right_arm_bounds_dict.values():
        assert all(lo <= hi for lo, hi in zip(lower, upper))
    assert len(right_arm_bounds_dict) == len(left_arm_bounds_dict)


def test_bounds_dict_to_list():
    flat, bones = bounds_dict_to_list(left_arm_bounds_dict, "left")
    assert len(bones) == 19 and len(flat) == 19 * 3
    assert bones[:4] == ["left_collar", "left_shoulder", "left_elbow", "left_wrist"]
    lower, upper = left_arm_bounds_dict["left_shoulder"]
    assert flat[3:6] == list(zip(lower, upper))
    assert flat == left_bounds and bones == controlled_bones_left


def test_bounds_dict_to_list_missing_bone_raises():
    with pytest.raises(KeyError):
        bounds_dict_to_list({}, "left")


def test_both_hands_list_has_no_duplicates():
    assert len(set(controlled_bones_both)) == len(controlled_bones_both)
    assert controlled_bones_both == controlled_bones_left + controlled_bones_right


# --- HandSpecification ----------------------------------------------------------

ARM = ["left_collar", "left_shoulder", "left_elbow", "left_wrist", "left_index1"]


def test_default_spec_only_has_elbow_direction():
    objs = HandSpecification().get_objectives(controlled_bones=ARM)
    assert len(objs) == 1
    assert isinstance(objs[0], BoneDirectionObjective)
    assert objs[0].bone_name == "left_elbow"
    assert np.allclose(objs[0].directions, [[1, 0, 0]])


def test_right_hand_mirrors_x():
    objs = HandSpecification(arm_flat=True).get_objectives(left_hand=False, controlled_bones=ARM)
    assert [o.bone_name for o in objs] == ["right_elbow", "right_shoulder"]
    assert np.allclose(objs[1].directions, [[-1, 0, 0]])


FLAGS = [
    "look_forward",
    "look_45_up",
    "look_45_down",
    "look_up",
    "look_down",
    "look_45_x_downwards",
    "look_45_x_upwards",
    "look_x_inward",
    "look_to_body",
    "arm_down",
    "arm_45_down",
    "arm_flat",
]


@pytest.mark.parametrize("flag", FLAGS)
def test_each_direction_flag_adds_weighted_objectives(flag):
    objs = HandSpecification(**{flag: True}).get_objectives(controlled_bones=ARM, weight=0.7)
    extra = objs[1:]
    assert len(extra) == (2 if flag == "look_forward" else 1)
    for o in extra:
        assert isinstance(o, BoneDirectionObjective)
        assert float(o.weight) == pytest.approx(0.7)


@pytest.mark.parametrize("flag", ["is_pointing", "is_shaping", "is_flat"])
def test_hand_pose_flags_add_masked_init_pose(flag):
    objs = HandSpecification(**{flag: True}).get_objectives(
        controlled_bones=ARM, full_trajectory=True, last_position=False, weight=2.0
    )
    pose = objs[-1]
    assert isinstance(pose, InitPoseObj)
    assert pose.full_trajectory and not pose.last_position
    assert float(pose.weight) == 2.0
    # Only the finger bone (index1) is constrained
    assert np.array_equal(pose.mask, [0] * 12 + [1] * 3)


def test_self_collision_flag():
    objs = HandSpecification(penalize_self_collision=True).get_objectives(
        controlled_bones=ARM, weight=2.0
    )
    sdf = [o for o in objs if isinstance(o, SDFSelfCollisionPenaltyObj)]
    assert len(sdf) == 1
    assert sdf[0].bone_names == ("left_wrist", "left_elbow")
    assert float(sdf[0].weight) == pytest.approx(0.2)


def test_all_flags_together():
    spec = HandSpecification(is_pointing=True, penalize_self_collision=True, **{f: True for f in FLAGS})
    objs = spec.get_objectives(controlled_bones=ARM)
    # elbow + collision + pose + 13 direction objectives
    assert len(objs) == 1 + 1 + 1 + 13
