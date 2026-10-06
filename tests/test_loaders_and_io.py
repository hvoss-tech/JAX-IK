import json

import jax.numpy as jnp
import numpy as np
import pytest
import trimesh
from conftest import PANDA_ARM

from jax_ik.helper import (
    compute_sdf,
    deform_mesh,
    inverse_skin_points,
    load_mesh_data_from_gltf,
    load_mesh_data_from_urdf,
    load_skeleton_from_gltf,
    load_skeleton_from_urdf,
)
from jax_ik.ik import (
    FKSolver,
    compute_objective_breakdown,
    export_all_frames,
    export_frames,
)
from jax_ik.objectives import BoneZeroRotationObj, DistanceObjTraj


def _write_urdf(tmp_path, body, name="robot.urdf"):
    path = tmp_path / name
    path.write_text(f'<?xml version="1.0"?>\n<robot name="r">{body}</robot>')
    return str(path)


# --- URDF skeleton ----------------------------------------------------------


def test_urdf_skeleton_structure(tiny_urdf):
    skeleton, limits = load_skeleton_from_urdf(tiny_urdf)
    assert set(skeleton) == {"base", "l1", "l2", "slider", "tip"}
    assert skeleton["base"]["parent"] is None
    assert skeleton["l2"]["parent"] == "l1" and "l2" in skeleton["l1"]["children"]
    assert skeleton["l1"]["joint_type"] == "revolute"
    assert skeleton["tip"]["joint_type"] == "fixed"
    assert limits["tip"] == (0, 0)
    assert limits["slider"] == pytest.approx((0.0, 0.2))
    assert limits["l2"] == pytest.approx((np.degrees(-2.0), np.degrees(0.5)))
    # Bone length is the distance to the child joint
    assert skeleton["l1"]["bone_length"] == pytest.approx(0.5)


def test_urdf_root_gets_y_up_transform(tiny_urdf):
    skeleton, _ = load_skeleton_from_urdf(tiny_urdf)
    C = skeleton["base"]["local_transform"]
    # URDF +z (up) becomes +y, and the robot is moved down by 1
    assert np.allclose(C[:3, :3] @ [0, 0, 1], [0, 1, 0], atol=1e-6)
    assert np.allclose(C[:3, 3], [0, -1, 0])


def test_urdf_limits_clamped_to_180_degrees(tmp_path):
    path = _write_urdf(
        tmp_path,
        '<link name="a"/><link name="b"/>'
        '<joint name="j" type="revolute"><parent link="a"/><child link="b"/>'
        '<axis xyz="0 0 1"/><limit lower="-10" upper="10" effort="1" velocity="1"/></joint>',
    )
    assert load_skeleton_from_urdf(path)[1]["b"] == (-180, 180)


def test_urdf_joint_axis_is_normalized(tmp_path):
    path = _write_urdf(
        tmp_path,
        '<link name="a"/><link name="b"/>'
        '<joint name="j" type="revolute"><parent link="a"/><child link="b"/>'
        '<axis xyz="0 3 4"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>',
    )
    skeleton, _ = load_skeleton_from_urdf(path)
    assert np.allclose(skeleton["b"]["joint_axis"], [0, 0.6, 0.8])


def test_urdf_mimic_master_resolves_to_link(tmp_path):
    # URDF names the master joint, the skeleton needs the master's link
    path = _write_urdf(
        tmp_path,
        '<link name="a"/><link name="b"/><link name="c"/>'
        '<joint name="j" type="revolute"><parent link="a"/><child link="b"/>'
        '<axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>'
        '<joint name="k" type="revolute"><parent link="b"/><child link="c"/>'
        '<axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/>'
        '<mimic joint="j" multiplier="2"/></joint>',
    )
    skeleton, _ = load_skeleton_from_urdf(path)
    assert skeleton["c"]["mimic"] == {"master": "b", "multiplier": 2.0, "offset": 0.0}


# --- URDF meshes ------------------------------------------------------------


def test_urdf_mesh_data_panda(panda_urdf):
    fk = FKSolver(panda_urdf, controlled_bones=PANDA_ARM, do_compute_sdf=False)
    md = load_mesh_data_from_urdf(panda_urdf, fk)
    V = md["vertices"].shape[0]
    assert md["vertices"].shape == (V, 4) and np.allclose(md["vertices"][:, 3], 1)
    assert md["vertex_assignment"].shape == (V,)
    assert int(md["vertex_assignment"].min()) >= 0
    assert int(md["vertex_assignment"].max()) < len(fk.bone_names)
    assert np.asarray(md["faces"]).max() < V and md["is_urdf"]
    # Every arm link owns some vertices
    owned = {fk.bone_names[i] for i in np.unique(np.asarray(md["vertex_assignment"]))}
    assert set(PANDA_ARM) <= owned


def test_urdf_mesh_follows_its_link(panda_urdf):
    fk = FKSolver(panda_urdf, controlled_bones=PANDA_ARM, do_compute_sdf=False)
    md = load_mesh_data_from_urdf(panda_urdf, fk)
    angles = np.zeros(len(PANDA_ARM) * 3, np.float32)
    angles[0] = 0.7
    posed = np.asarray(deform_mesh(angles, fk, md))
    rest = np.asarray(md["vertices"][:, :3])
    assign = np.asarray(md["vertex_assignment"])
    base = assign == fk.bone_names.index("panda_link0")
    hand = assign == fk.bone_names.index("panda_hand")
    assert np.allclose(posed[base], rest[base], atol=1e-5)
    assert not np.allclose(posed[hand], rest[hand], atol=1e-3)


def test_urdf_without_meshes_gets_boxes(tiny_urdf):
    fk = FKSolver(tiny_urdf, controlled_bones=["l1"], do_compute_sdf=False)
    md = load_mesh_data_from_urdf(tiny_urdf, fk)
    # One 8-vertex box per link
    assert md["vertices"].shape[0] == 8 * len(fk.bone_names)
    assert np.allclose(deform_mesh(np.zeros(3), fk, md), md["vertices"][:, :3], atol=1e-6)


# --- glTF ---------------------------------------------------------------------


def test_gltf_skeleton_smplx(model_path):
    skeleton = load_skeleton_from_gltf(model_path)
    assert skeleton["pelvis"]["parent"] is None
    for name, bone in skeleton.items():
        assert bone["local_transform"].shape == (4, 4)
        assert bone["bone_length"] > 0
        for child in bone["children"]:
            assert skeleton[child]["parent"] == name


def test_gltf_without_pelvis_raises(tmp_path):
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.creation.box(), node_name="root", geom_name="box")
    path = tmp_path / "nopelvis.glb"
    scene.export(str(path))
    with pytest.raises(ValueError, match="pelvis"):
        load_skeleton_from_gltf(str(path))


def test_gltf_skinned_mesh_data(model_path):
    fk = FKSolver(model_path, controlled_bones=["left_elbow"], do_compute_sdf=False)
    md = load_mesh_data_from_gltf(model_path, fk)
    V = md["vertices"].shape[0]
    w = np.asarray(md["skin_weights"])
    j = np.asarray(md["skin_joints"])
    assert w.shape == j.shape == (V, 4)
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-5)
    assert j.min() >= 0 and j.max() < len(fk.bone_names)
    assert np.asarray(md["faces"]).max() < V


def test_gltf_skinned_rest_pose_is_unchanged(model_path):
    fk = FKSolver(model_path, controlled_bones=["left_elbow"], do_compute_sdf=False)
    md = load_mesh_data_from_gltf(model_path, fk)
    assert np.allclose(deform_mesh(np.zeros(3), fk, md), md["vertices"][:, :3], atol=1e-5)


# --- deform / inverse skinning ----------------------------------------------


class _OneBoneFK:
    def __init__(self, offset):
        self.bind_fk = jnp.eye(4)[None]
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = offset
        self._T = jnp.asarray(T)[None]

    def compute_fk_from_angles(self, angles):
        return self._T


def _lbs_mesh():
    verts = np.array([[0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1]], np.float32)
    return {
        "vertices": jnp.asarray(verts),
        "skin_joints": jnp.zeros((3, 4), jnp.int32),
        "skin_weights": jnp.asarray(np.tile([1.0, 0, 0, 0], (3, 1)), jnp.float32),
        "faces": np.array([[0, 1, 2]]),
    }


def test_deform_mesh_lbs_and_rigid_translate():
    fk = _OneBoneFK([0.0, 0.0, 2.0])
    lbs = _lbs_mesh()
    rigid = {"vertices": lbs["vertices"], "vertex_assignment": jnp.zeros(3, jnp.int32)}
    expected = np.asarray(lbs["vertices"][:, :3]) + [0, 0, 2]
    assert np.allclose(deform_mesh(None, fk, lbs), expected)
    assert np.allclose(deform_mesh(None, fk, rigid), expected)


def test_deform_mesh_rejects_unknown_mesh_data():
    with pytest.raises(ValueError):
        deform_mesh(None, _OneBoneFK([0, 0, 0]), {"vertices": jnp.zeros((1, 4))})


def test_inverse_skin_points_undoes_skinning():
    fk = _OneBoneFK([0.0, 0.0, 2.0])
    md = _lbs_mesh()
    world = jnp.asarray([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]])
    local = inverse_skin_points(world, fk, md, fk._T)
    assert np.allclose(local, [[0, 0, 0], [1, 0, 0]], atol=1e-6)


def test_inverse_skin_points_without_skin_is_identity():
    pts = jnp.ones((2, 3))
    fk = _OneBoneFK([0, 0, 1])
    assert np.allclose(inverse_skin_points(pts, fk, None, fk._T), pts)
    assert np.allclose(inverse_skin_points(pts, fk, {"vertices": jnp.zeros((1, 4))}, fk._T), pts)


# --- SDF ----------------------------------------------------------------------


def test_compute_sdf_box():
    box = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    sdf = compute_sdf(box, resolution=12)
    assert sdf["grid"].shape == (12, 12, 12) and sdf["grid"].dtype == jnp.float32
    assert float(sdf["spacing"]) == pytest.approx(1.1 / 11, rel=1e-5)
    assert np.allclose(sdf["origin"], [-0.55] * 3, atol=1e-5)
    grid = np.asarray(sdf["grid"])
    # Inside is negative, the grid corners are outside
    assert grid[5:7, 5:7, 5:7].max() < 0
    assert grid[0, 0, 0] > 0


def test_compute_sdf_rejects_non_mesh():
    with pytest.raises(TypeError):
        compute_sdf(np.zeros((3, 3)))


# --- export / breakdown -------------------------------------------------------


def test_export_frames_single_pose(tmp_path):
    out = tmp_path / "f.json"
    export_frames(np.zeros(6), np.arange(6.0), ["a", "b"], str(out))
    frames = json.loads(out.read_text())
    assert frames == [{"a": [0, 0, 0], "b": [0, 0, 0]}, {"a": [0, 1, 2], "b": [3, 4, 5]}]


def test_export_frames_trajectory(tmp_path):
    out = tmp_path / "f.json"
    traj = np.arange(9.0).reshape(3, 3)
    export_frames(np.zeros(3), traj, ["a"], str(out))
    assert json.loads(out.read_text()) == [{"a": [0, 1, 2]}, {"a": [3, 4, 5]}, {"a": [6, 7, 8]}]


def test_export_frames_bone_count_mismatch(tmp_path):
    with pytest.raises(ValueError):
        export_frames(np.zeros(6), np.zeros(6), ["only_one"], str(tmp_path / "f.json"))


def test_export_all_frames_concatenates(tmp_path):
    out = tmp_path / "all.json"
    export_all_frames(
        [(np.zeros(3), np.ones(3)), (np.zeros(3), np.full((2, 3), 2.0))], ["a"], str(out)
    )
    frames = json.loads(out.read_text())
    assert [f["a"][0] for f in frames] == [0, 1, 2, 2]
    with pytest.raises(ValueError):
        export_all_frames([(np.zeros(6), np.zeros(6))], ["a"], str(out))


def test_objective_breakdown(tiny_urdf):
    fk = FKSolver(tiny_urdf, controlled_bones=["l1"], do_compute_sdf=False)
    X = np.array([[0.2, 0.0, 0.0], [0.4, 0.0, 0.0]], np.float32)
    dist = DistanceObjTraj("tip", [0, 0, 0], True)
    out = compute_objective_breakdown(X, [("zero", BoneZeroRotationObj()), ("dist", dist)], fk)
    assert set(out) == {"zero", "dist"}
    assert all(isinstance(v, float) for v in out.values())
    assert out["zero"] == pytest.approx((0.04 + 0.16) / 6, rel=1e-5)
    assert out["dist"] == pytest.approx(float(dist(jnp.asarray(X), fk)), rel=1e-6)
    assert compute_objective_breakdown(X, [], fk) == {}
