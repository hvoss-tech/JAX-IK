import os

import numpy as np
import pytest

from jax_ik.helper import load_skeleton_from_urdf
from jax_ik.ik import FKSolver, InverseKinematicsSolver
from jax_ik.objectives import DistanceObjTraj

PANDA_URDF = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "models", "panda", "urdf", "panda.urdf"
    )
)
FINGER_MAX = 0.04  # panda_finger_joint upper limit in meters

pytestmark = pytest.mark.skipif(
    not os.path.exists(PANDA_URDF), reason="Panda URDF not available"
)


def test_prismatic_limits_stay_in_meters():
    skeleton, limits = load_skeleton_from_urdf(PANDA_URDF)

    assert limits["panda_leftfinger"] == pytest.approx((0.0, FINGER_MAX))
    assert limits["panda_rightfinger"] == pytest.approx((0.0, FINGER_MAX))
    assert skeleton["panda_leftfinger"]["joint_type"] == "prismatic"
    assert np.allclose(skeleton["panda_leftfinger"]["joint_axis"], [0.0, 1.0, 0.0])

    # Revolute limits are still converted to degrees
    lo, hi = limits["panda_link1"]
    assert lo == pytest.approx(-166.0, abs=0.1)
    assert hi == pytest.approx(166.0, abs=0.1)


def test_prismatic_joint_translates_along_axis():
    bones = ["panda_link1", "panda_leftfinger"]
    fk_solver = FKSolver(PANDA_URDF, controlled_bones=bones, do_compute_sdf=False)
    finger_slot = [fk_solver.bone_names[i] for i in fk_solver.controlled_indices].index(
        "panda_leftfinger"
    )
    finger_idx = fk_solver.bone_names.index("panda_leftfinger")

    zero = np.zeros(len(bones) * 3, dtype=np.float32)
    opened = zero.copy()
    opened[finger_slot * 3] = 0.03
    # The unused second/third parameters must not move a prismatic joint
    opened[finger_slot * 3 + 1] = 0.5
    opened[finger_slot * 3 + 2] = -0.5

    fk0 = np.asarray(fk_solver.compute_fk_from_angles(zero))[finger_idx]
    fk1 = np.asarray(fk_solver.compute_fk_from_angles(opened))[finger_idx]

    # Orientation unchanged, position shifted 3cm along the world joint axis
    assert np.allclose(fk0[:3, :3], fk1[:3, :3], atol=1e-6)
    world_axis = fk0[:3, :3] @ np.array([0.0, 1.0, 0.0])
    assert np.allclose(fk1[:3, 3] - fk0[:3, 3], 0.03 * world_axis, atol=1e-6)


def test_solver_uses_metric_bounds_for_prismatic_joints():
    bones = ["panda_link1", "panda_leftfinger"]
    solver = InverseKinematicsSolver(
        PANDA_URDF, controlled_bones=bones, compute_sdf=False
    )
    lower = np.asarray(solver.lower_bounds)
    upper = np.asarray(solver.upper_bounds)

    # Revolute joint bounds are converted to radians
    assert upper[2] == pytest.approx(np.radians(166.0), abs=1e-3)
    # Prismatic bounds stay in meters, only the first parameter can move
    assert lower[3:6] == pytest.approx([0.0, 0.0, 0.0])
    assert upper[3:6] == pytest.approx([FINGER_MAX, 0.0, 0.0])


def test_solve_opens_finger_and_respects_limit():
    bones = ["panda_leftfinger"]
    solver = InverseKinematicsSolver(
        PANDA_URDF, controlled_bones=bones, threshold=1e-8, compute_sdf=False
    )
    fk_solver = solver.fk_solver
    finger_idx = fk_solver.bone_names.index("panda_leftfinger")
    fk0 = np.asarray(fk_solver.compute_fk_from_angles(np.zeros(3)))[finger_idx]
    world_axis = fk0[:3, :3] @ np.array([0.0, 1.0, 0.0])

    def solve_to(opening):
        target = fk0[:3, 3] + opening * world_axis
        result, _, _ = solver.solve(
            initial_rotations=np.zeros(3, dtype=np.float32),
            learning_rate=0.01,
            mandatory_objective_functions=[
                DistanceObjTraj(
                    bone_name="panda_leftfinger",
                    target_points=[target],
                    use_head=True,
                )
            ],
            patience=50,
            verbose=False,
        )
        return np.asarray(result[-1])

    # Reachable opening inside the 0-4cm range
    final = solve_to(0.02)
    assert final[0] == pytest.approx(0.02, abs=2e-3)

    # Target beyond the limit gets clamped at the real 4cm limit
    final = solve_to(0.1)
    assert final[0] == pytest.approx(FINGER_MAX, abs=1e-3)
    assert final[0] <= FINGER_MAX + 1e-6


def _global(fk_solver, angles, bone):
    fk = np.asarray(fk_solver.compute_fk_from_angles(np.asarray(angles, np.float32)))
    return fk[fk_solver.bone_names.index(bone)]


def test_angle_vector_follows_controlled_bones_order():
    # Deliberately listed against the skeleton's own (root-first) order
    bones = ["panda_link2", "panda_link1"]
    fk_solver = FKSolver(PANDA_URDF, controlled_bones=bones, do_compute_sdf=False)
    assert [fk_solver.bone_names[i] for i in fk_solver.controlled_indices] == bones

    zero = np.zeros(6)
    moved = zero.copy()
    moved[2] = 0.5  # first slot -> panda_link2
    # panda_link1 sits above panda_link2, so it must stay put
    assert np.allclose(
        _global(fk_solver, zero, "panda_link1"),
        _global(fk_solver, moved, "panda_link1"),
        atol=1e-6,
    )
    assert not np.allclose(
        _global(fk_solver, zero, "panda_link3"),
        _global(fk_solver, moved, "panda_link3"),
        atol=1e-4,
    )


def test_solver_bounds_follow_controlled_bones_order():
    bones = ["panda_leftfinger", "panda_link1"]
    solver = InverseKinematicsSolver(
        PANDA_URDF, controlled_bones=bones, compute_sdf=False
    )
    upper = np.asarray(solver.upper_bounds)
    assert upper[0:3] == pytest.approx([FINGER_MAX, 0.0, 0.0])
    assert upper[5] == pytest.approx(np.radians(166.0), abs=1e-3)

    # And the first slot really drives the finger in FK
    fk_solver = solver.fk_solver
    opened = np.zeros(6)
    opened[0] = 0.03
    fk0 = _global(fk_solver, np.zeros(6), "panda_leftfinger")
    fk1 = _global(fk_solver, opened, "panda_leftfinger")
    assert np.linalg.norm(fk1[:3, 3] - fk0[:3, 3]) == pytest.approx(0.03, abs=1e-6)


def test_prismatic_mimic_finger_follows_master():
    skeleton, _ = load_skeleton_from_urdf(PANDA_URDF)
    assert skeleton["panda_rightfinger"]["mimic"] == {
        "master": "panda_leftfinger",
        "multiplier": 1.0,
        "offset": 0.0,
    }

    # Right finger is listed too, its own parameters must be ignored
    bones = ["panda_leftfinger", "panda_rightfinger"]
    fk_solver = FKSolver(PANDA_URDF, controlled_bones=bones, do_compute_sdf=False)
    zero = np.zeros(6)
    opened = np.array([0.03, 0.0, 0.0, 0.01, 0.2, -0.2])

    left0 = _global(fk_solver, zero, "panda_leftfinger")
    left1 = _global(fk_solver, opened, "panda_leftfinger")
    right0 = _global(fk_solver, zero, "panda_rightfinger")
    right1 = _global(fk_solver, opened, "panda_rightfinger")

    right_axis = right0[:3, :3] @ np.array([0.0, -1.0, 0.0])
    assert np.allclose(right1[:3, :3], right0[:3, :3], atol=1e-6)
    assert np.allclose(right1[:3, 3] - right0[:3, 3], 0.03 * right_axis, atol=1e-6)
    # Fingers open symmetrically, in opposite directions
    assert np.allclose(
        left1[:3, 3] - left0[:3, 3], -(right1[:3, 3] - right0[:3, 3]), atol=1e-6
    )

    # Mimic bones get zero bounds, the solver must not move them on their own
    solver = InverseKinematicsSolver(
        PANDA_URDF, controlled_bones=bones, compute_sdf=False
    )
    assert np.asarray(solver.upper_bounds)[3:6] == pytest.approx([0.0, 0.0, 0.0])
    assert np.asarray(solver.lower_bounds)[3:6] == pytest.approx([0.0, 0.0, 0.0])


def test_mimic_finger_follows_without_being_controlled():
    fk_solver = FKSolver(
        PANDA_URDF, controlled_bones=["panda_leftfinger"], do_compute_sdf=False
    )
    right0 = _global(fk_solver, np.zeros(3), "panda_rightfinger")
    right1 = _global(fk_solver, [0.02, 0.0, 0.0], "panda_rightfinger")
    right_axis = right0[:3, :3] @ np.array([0.0, -1.0, 0.0])
    assert np.allclose(right1[:3, 3] - right0[:3, 3], 0.02 * right_axis, atol=1e-6)


REVOLUTE_MIMIC_URDF = """<?xml version="1.0"?>
<robot name="mimic_test">
  <link name="base"/>
  <link name="master"/>
  <link name="slave"/>
  <joint name="master_joint" type="revolute">
    <parent link="base"/>
    <child link="master"/>
    <axis xyz="0 0 -1"/>
    <limit lower="-1.0" upper="1.0" effort="1" velocity="1"/>
  </joint>
  <joint name="slave_joint" type="revolute">
    <parent link="base"/>
    <child link="slave"/>
    <origin xyz="1 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-3.0" upper="3.0" effort="1" velocity="1"/>
    <mimic joint="master_joint" multiplier="-2.0" offset="0.1"/>
  </joint>
</robot>
"""


def test_revolute_mimic_uses_master_joint_value(tmp_path):
    urdf = tmp_path / "mimic.urdf"
    urdf.write_text(REVOLUTE_MIMIC_URDF)
    fk_solver = FKSolver(str(urdf), controlled_bones=["master"], do_compute_sdf=False)

    # Euler z of 0.3 about +z is a joint value of -0.3 about the -z axis
    master = _global(fk_solver, [0.0, 0.0, 0.3], "master")
    slave = _global(fk_solver, [0.0, 0.0, 0.3], "slave")
    slave0 = _global(fk_solver, [0.0, 0.0, 0.0], "slave")
    base = _global(fk_solver, [0.0, 0.0, 0.0], "base")

    def local_angle_about_x(g):
        R = base[:3, :3].T @ g[:3, :3]
        return np.arctan2(R[2, 1], R[1, 1])

    def local_angle_about_z(g):
        R = base[:3, :3].T @ g[:3, :3]
        return np.arctan2(R[1, 0], R[0, 0])

    assert local_angle_about_z(master) == pytest.approx(0.3, abs=1e-5)
    # slave = -2 * (-0.3) + 0.1 about +x
    assert local_angle_about_x(slave) == pytest.approx(0.7, abs=1e-5)
    # Offset applies even at the master's zero position
    assert local_angle_about_x(slave0) == pytest.approx(0.1, abs=1e-5)


NEGATIVE_AXIS_URDF = """<?xml version="1.0"?>
<robot name="negative_axis_test">
  <link name="base"/>
  <link name="arm"/>
  <joint name="arm_joint" type="revolute">
    <parent link="base"/>
    <child link="arm"/>
    <axis xyz="0 0 -1"/>
    <limit lower="-0.2" upper="1.0" effort="1" velocity="1"/>
  </joint>
</robot>
"""


def test_negative_axis_revolute_bounds_are_mirrored(tmp_path):
    urdf = tmp_path / "negative_axis.urdf"
    urdf.write_text(NEGATIVE_AXIS_URDF)
    solver = InverseKinematicsSolver(
        str(urdf), controlled_bones=["arm"], compute_sdf=False
    )

    # Euler z rotates about +z, so joint values [-0.2, 1.0] about -z
    # correspond to Euler z in [-1.0, 0.2]
    assert float(solver.lower_bounds[2]) == pytest.approx(-1.0, abs=1e-5)
    assert float(solver.upper_bounds[2]) == pytest.approx(0.2, abs=1e-5)

    # The upper joint limit really is reachable at the new lower Euler bound
    fk_solver = solver.fk_solver
    base = _global(fk_solver, np.zeros(3), "base")
    arm = _global(fk_solver, [0.0, 0.0, float(solver.lower_bounds[2])], "arm")
    R = base[:3, :3].T @ arm[:3, :3]
    joint_value = -np.arctan2(R[1, 0], R[0, 0])  # angle about -z
    assert joint_value == pytest.approx(1.0, abs=1e-5)
