import jax
import jax.numpy as jnp
import numpy as np
import pytest
import urchin
from conftest import PANDA_ARM

from jax_ik.helper import load_skeleton_from_urdf
from jax_ik.ik import JOINT_EULER, JOINT_PRISMATIC, JOINT_REVOLUTE, FKSolver

TINY_BONES = ["l1", "l2", "slider"]


def _urchin_reference(urdf, bone_names, joint_values):
    """Our FK in the library's frame: coord transform @ urchin's link FK."""
    robot = urchin.URDF.load(urdf, lazy_load_meshes=True)
    skeleton, _ = load_skeleton_from_urdf(urdf)
    root = [n for n, b in skeleton.items() if b["parent"] is None][0]
    C = skeleton[root]["local_transform"]
    link_fk = robot.link_fk(cfg=joint_values)
    by_name = {link.name: C @ T for link, T in link_fk.items()}
    return np.stack([by_name[n] for n in bone_names])


# --- loading and structure --------------------------------------------------


def test_bones_are_ordered_parents_first(panda_urdf):
    fk = FKSolver(panda_urdf, controlled_bones=PANDA_ARM, do_compute_sdf=False)
    for i, p in enumerate(fk.parent_list):
        assert p < i
    assert fk.parent_list[0] == -1
    # Level bounds cover every bone exactly once, in order
    covered = [i for start, end in fk.level_bounds for i in range(start, end)]
    assert covered == list(range(len(fk.bone_names)))


def test_bind_fk_is_fk_of_zero_pose(panda_urdf):
    fk = FKSolver(panda_urdf, controlled_bones=PANDA_ARM, do_compute_sdf=False)
    zero = fk.compute_fk_from_angles(np.zeros(len(PANDA_ARM) * 3))
    assert np.allclose(fk.bind_fk, zero)
    assert fk.bind_fk.shape == (len(fk.bone_names), 4, 4)


def test_fk_matches_urchin_for_random_poses(panda_urdf):
    bones = PANDA_ARM + ["panda_leftfinger"]
    fk = FKSolver(panda_urdf, controlled_bones=bones, do_compute_sdf=False)
    robot = urchin.URDF.load(panda_urdf, lazy_load_meshes=True)
    joint_of = {j.child: j for j in robot.joints}
    rng = np.random.default_rng(0)
    for _ in range(5):
        values = {}
        angles = np.zeros(len(bones) * 3, np.float32)
        for k, bone in enumerate(bones):
            joint = joint_of[bone]
            values[joint.name] = rng.uniform(joint.limit.lower, joint.limit.upper)
            angles[3 * k] = values[joint.name]
        ours = np.asarray(fk.compute_fk_from_angles(angles))
        ref = _urchin_reference(panda_urdf, fk.bone_names, values)
        assert np.allclose(ours, ref, atol=1e-4)


def test_fk_matches_urchin_on_tiny_chain(tiny_urdf):
    fk = FKSolver(tiny_urdf, controlled_bones=TINY_BONES, do_compute_sdf=False)
    angles = np.array([0.4, 0, 0, -0.8, 0, 0, 0.15, 0, 0], np.float32)
    ref = _urchin_reference(tiny_urdf, fk.bone_names, {"j1": 0.4, "j2": -0.8, "j3": 0.15})
    assert np.allclose(fk.compute_fk_from_angles(angles), ref, atol=1e-5)


def test_joint_modes_from_urdf_types(tiny_urdf):
    fk = FKSolver(tiny_urdf, controlled_bones=TINY_BONES, do_compute_sdf=False)
    assert list(np.asarray(fk.joint_mode)) == [
        JOINT_REVOLUTE,
        JOINT_REVOLUTE,
        JOINT_PRISMATIC,
    ]
    assert np.allclose(fk.joint_axes, [[0, 0, 1], [0, 1, 0], [1, 0, 0]])


def test_gltf_bones_keep_euler_mode(model_path):
    fk = FKSolver(model_path, controlled_bones=["left_elbow"], do_compute_sdf=False)
    assert fk.joint_mode is None and fk.joint_axes is None
    assert JOINT_EULER == 0


def test_no_controlled_bones(tiny_urdf):
    fk = FKSolver(tiny_urdf, controlled_bones=None, do_compute_sdf=False)
    assert fk.controlled_bones == [] and fk.controlled_indices == []
    assert fk.compute_fk_from_angles(np.zeros(0)).shape == (len(fk.bone_names), 4, 4)


def test_fixed_joint_bone_stays_rigid(tiny_urdf):
    fk = FKSolver(tiny_urdf, controlled_bones=TINY_BONES, do_compute_sdf=False)
    out = np.asarray(fk.compute_fk_from_angles(np.array([0.4, 0, 0, -0.8, 0, 0, 0.1, 0, 0])))
    slider, tip = fk.bone_names.index("slider"), fk.bone_names.index("tip")
    rel = np.linalg.inv(out[slider]) @ out[tip]
    assert np.allclose(rel[:3, :3], np.eye(3), atol=1e-6)
    assert np.allclose(rel[:3, 3], [0, 0, 0.1], atol=1e-6)


def test_fk_is_jittable_and_differentiable(panda_urdf):
    fk = FKSolver(panda_urdf, controlled_bones=PANDA_ARM, do_compute_sdf=False)
    idx = fk.bone_names.index("panda_hand_tcp")
    f = jax.jit(lambda a: fk.compute_fk_from_angles(a)[idx, :3, 3].sum())
    g = jax.grad(f)(jnp.zeros(len(PANDA_ARM) * 3))
    assert np.all(np.isfinite(g))
    # Unused (off-axis) parameters have no gradient
    assert np.allclose(np.asarray(g).reshape(-1, 3)[:, 1:], 0)


# --- head/tail --------------------------------------------------------------


def test_head_tail(tiny_urdf):
    fk = FKSolver(tiny_urdf, controlled_bones=TINY_BONES, do_compute_sdf=False)
    head, tail = fk.get_bone_head_tail_from_fk(fk.bind_fk, "l1")
    assert np.allclose(head, np.asarray(fk.bind_fk)[fk.bone_names.index("l1"), :3, 3])
    # Tail is bone_length along local +y
    assert np.linalg.norm(tail - head) == pytest.approx(fk.skeleton["l1"]["bone_length"])


def test_head_tail_unknown_bone_raises(tiny_urdf):
    fk = FKSolver(tiny_urdf, controlled_bones=[], do_compute_sdf=False)
    with pytest.raises(ValueError):
        fk.get_bone_head_tail_from_fk(fk.bind_fk, "missing")


# --- pruning ----------------------------------------------------------------


def test_bones_of_interest_prunes_but_keeps_fk(panda_urdf):
    full = FKSolver(panda_urdf, controlled_bones=["panda_link1", "panda_link2"], do_compute_sdf=False)
    pruned = FKSolver(
        panda_urdf,
        controlled_bones=["panda_link1", "panda_link2"],
        bones_of_interest=["panda_link4"],
        do_compute_sdf=False,
    )
    assert set(pruned.bone_names) == {f"panda_link{i}" for i in range(5)}
    angles = np.array([0.3, 0, 0, -0.5, 0, 0], np.float32)
    a, b = full.compute_fk_from_angles(angles), pruned.compute_fk_from_angles(angles)
    for name in pruned.bone_names:
        assert np.allclose(a[full.bone_names.index(name)], b[pruned.bone_names.index(name)], atol=1e-6)


def test_pruned_view_shares_skeleton_and_drops_mesh(panda_urdf):
    base = FKSolver(panda_urdf, controlled_bones=["panda_link1"], do_compute_sdf=False)
    view = FKSolver._pruned_view(base, ["panda_link1"], ["panda_link3"])
    assert view.skeleton is base.skeleton
    assert view.mesh_data is None and view.sdf is None
    assert "panda_hand" not in view.bone_names


# --- errors -----------------------------------------------------------------


def test_unsupported_file_type_raises(tmp_path):
    path = tmp_path / "model.obj"
    path.write_text("")
    with pytest.raises(ValueError, match="Unsupported"):
        FKSolver(str(path), controlled_bones=[])


def test_missing_file_raises():
    with pytest.raises(Exception):
        FKSolver("/nonexistent/robot.urdf", controlled_bones=[])
