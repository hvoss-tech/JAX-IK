"""Regression tests for bugs found while writing the test suite."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import trimesh
from jax.tree_util import register_pytree_node_class

from jax_ik.helper import deform_mesh, load_mesh_data_from_gltf, load_skeleton_from_urdf
from jax_ik.ik import FKSolver, InverseKinematicsSolver, export_all_frames, export_frames
from jax_ik.objectives import (
    DistanceObjTraj,
    ObjectiveFunction,
    SDFCollisionPenaltyObj,
    SDFSelfCollisionPenaltyObj,
)

TINY_BONES = ["l1", "l2", "slider"]


# Unknown controlled bones used to be dropped silently, leaving the angle
# vector shorter than the bounds


def test_unknown_controlled_bone_raises(tiny_urdf):
    with pytest.raises(ValueError, match="nope"):
        FKSolver(tiny_urdf, controlled_bones=["l1", "nope"], do_compute_sdf=False)
    with pytest.raises(ValueError, match="nope"):
        InverseKinematicsSolver(tiny_urdf, controlled_bones=["nope"], compute_sdf=False)


def test_unknown_bone_of_interest_raises(tiny_urdf):
    with pytest.raises(ValueError, match="ghost"):
        FKSolver(tiny_urdf, controlled_bones=["l1"], bones_of_interest=["ghost"], do_compute_sdf=False)


# GLB/GLTF models without bounds failed with "'NoneType' object is not iterable"


def test_gltf_without_bounds_raises_clear_error(model_path):
    with pytest.raises(ValueError, match="bounds are required"):
        InverseKinematicsSolver(model_path, controlled_bones=["left_elbow"], compute_sdf=False)


# Wrong-length bounds failed later with a cryptic reshape error


@pytest.mark.parametrize("n", [3, 8, 10])
def test_wrong_number_of_bounds_raises(tiny_urdf, n):
    with pytest.raises(ValueError, match="Expected 9 bounds"):
        InverseKinematicsSolver(
            tiny_urdf, controlled_bones=TINY_BONES, bounds=[(-1, 1)] * n, compute_sdf=False
        )


def test_no_controlled_bones_raises(tiny_urdf):
    with pytest.raises(ValueError, match="controlled bone"):
        InverseKinematicsSolver(tiny_urdf, controlled_bones=[], compute_sdf=False)


# Duplicate controlled bones made the bounds longer than the angle vector


def test_duplicate_controlled_bones_are_merged(tiny_urdf):
    solver = InverseKinematicsSolver(
        tiny_urdf, controlled_bones=["l1", "l2", "l1"], compute_sdf=False
    )
    assert solver.controlled_bones == ["l1", "l2"]
    assert solver.lower_bounds.shape == (6,)
    result, loss, _ = solver.solve(
        mandatory_objective_functions=[DistanceObjTraj("tip", [0.2, 0, 0.9], True)],
        verbose=False,
    )
    assert result.shape == (2, 6) and np.isfinite(loss)


# export_frames/export_all_frames crashed on plain Python lists


def test_export_accepts_lists(tmp_path):
    out = tmp_path / "f.json"
    export_frames([0.0, 0.0, 0.0], [1.0, 2.0, 3.0], ["a"], str(out))
    assert json.loads(out.read_text())[1] == {"a": [1, 2, 3]}
    export_all_frames([([0.0] * 3, [[1.0, 2.0, 3.0]])], ["a"], str(out))
    assert json.loads(out.read_text()) == [{"a": [1, 2, 3]}]


# SDF objectives used cval=inf, so linear interpolation at or beyond the grid
# edge computed inf * 0 = nan, which poisoned the solver's joint angles


class _PointFK:
    """Bone "b" from angles[:3] to angles[:3] + (0, 1, 0)."""

    bone_names = ["b"]

    def compute_fk_from_angles(self, cfg):
        return jnp.eye(4).at[:3, 3].set(cfg[:3])[None]

    def get_bone_head_tail_from_fk(self, fk, name):
        return fk[0, :3, 3], fk[0, :3, 3] + jnp.array([0.0, 1.0, 0.0])


def _grid(value):
    # Covers [-1, 1]^3, the bone from the origin ends exactly on its edge
    return {
        "grid": jnp.full((5, 5, 5), value, jnp.float32),
        "origin": jnp.full(3, -1.0, jnp.float32),
        "spacing": jnp.float32(0.5),
    }


@pytest.mark.parametrize("cfg", [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [30.0, 30.0, 30.0]])
def test_sdf_collision_finite_at_and_beyond_grid_edge(cfg):
    obj = SDFCollisionPenaltyObj("b", _grid(-0.2), num_samples=5)
    x = jnp.asarray(cfg, jnp.float32)
    assert np.isfinite(float(obj(x, _PointFK())))
    g = jax.grad(lambda c: obj(c, _PointFK()))(x)
    assert np.all(np.isfinite(np.asarray(g)))


def test_sdf_outside_grid_is_free_space():
    obj = SDFCollisionPenaltyObj("b", _grid(-0.2), num_samples=3)
    assert float(obj(jnp.full(3, 30.0), _PointFK())) == 0.0


def test_sdf_self_collision_finite_at_grid_edge():
    fk = _PointFK()
    fk.sdf = _grid(-0.1)
    fk.mesh_data = {"vertices": jnp.zeros((1, 4))}
    obj = SDFSelfCollisionPenaltyObj(["b"], num_samples_per_bone=5)
    x = jnp.zeros(3)
    assert np.isfinite(float(obj(x, fk)))
    assert np.all(np.isfinite(np.asarray(jax.grad(lambda c: obj(c, fk))(x))))


def test_solver_with_partially_covered_sdf_stays_finite(panda_urdf):
    solver = InverseKinematicsSolver(
        panda_urdf, controlled_bones=["panda_link1", "panda_link2"], num_steps=50, compute_sdf=False
    )
    fk = solver.fk_solver
    p = np.asarray(fk.compute_fk_from_angles(np.zeros(6)))[fk.bone_names.index("panda_link3"), :3, 3]
    sdf = {
        "grid": jnp.full((4, 4, 4), 0.5, jnp.float32),
        "origin": jnp.asarray(p - 0.1, jnp.float32),
        "spacing": jnp.float32(0.05),
    }
    result, loss, _ = solver.solve(
        mandatory_objective_functions=[DistanceObjTraj("panda_link3", [p + 0.05], True)],
        optional_objective_functions=[SDFCollisionPenaltyObj("panda_link3", sdf)],
        verbose=False,
    )
    assert np.all(np.isfinite(result)) and loss < 1e6


# DistanceObjTraj only checked the last dimension of target_points


def test_distance_rejects_3d_target_array():
    with pytest.raises(ValueError):
        DistanceObjTraj("b", np.zeros((1, 2, 3)))


# A nan gradient from any objective turned every joint angle into nan


@register_pytree_node_class
class _SqrtObj(ObjectiveFunction):
    def __init__(self):
        self.weight = np.float32(1.0)

    def referenced_bones(self):
        return ()

    def __call__(self, X, fk_solver=None):
        return jnp.sqrt(X[-1, 0] - 0.2) * self.weight  # nan below 0.2


def test_nan_gradient_does_not_corrupt_solution(tiny_urdf):
    solver = InverseKinematicsSolver(
        tiny_urdf, controlled_bones=["l1", "l2"], threshold=1e-7, num_steps=300, compute_sdf=False
    )
    target = [[0.3, 0.0, 0.8]]
    reach = [DistanceObjTraj("tip", target, True)]
    with_nan, loss, _ = solver.solve(
        mandatory_objective_functions=reach,
        optional_objective_functions=[_SqrtObj()],
        learning_rate=0.05,
        verbose=False,
    )
    _, clean_loss, _ = solver.solve(mandatory_objective_functions=reach, learning_rate=0.05, verbose=False)
    assert np.all(np.isfinite(with_nan))
    # The other objectives still drive the solve
    assert loss < 2 * clean_loss + 1e-6


# Continuous joints got +-30 degrees without a <limit> tag but +-90 with one


def test_continuous_joint_limits_without_limit_tag(tmp_path):
    path = tmp_path / "c.urdf"
    path.write_text(
        '<?xml version="1.0"?><robot name="c"><link name="a"/><link name="b"/><link name="c"/>'
        '<joint name="j" type="continuous"><parent link="a"/><child link="b"/><axis xyz="0 0 1"/></joint>'
        '<joint name="k" type="continuous"><parent link="b"/><child link="c"/><axis xyz="0 0 1"/>'
        '<limit effort="1" velocity="1"/></joint></robot>'
    )
    _, limits = load_skeleton_from_urdf(str(path))
    assert limits["b"] == (-90, 90) == limits["c"]


# glTF files without a skin crashed: decimation needed the optional
# fast_simplification package and passed the face count as `percent`, and
# scene.graph.get's (transform, geometry) tuple was used as a matrix


def test_unskinned_gltf_loads_with_rigid_assignment(tmp_path):
    scene = trimesh.Scene()
    box = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
    box.apply_translation([0.0, 0.5, 0.0])
    scene.add_geometry(box, node_name="pelvis", geom_name="box")
    path = str(tmp_path / "box.glb")
    scene.export(path)

    fk = FKSolver(path, controlled_bones=["pelvis"], do_compute_sdf=False)
    md = load_mesh_data_from_gltf(path, fk)
    assert "vertex_assignment" in md
    verts = np.asarray(md["vertices"][:, :3])
    assert verts.shape[1] == 3 and len(verts) > 0
    assert np.allclose(verts.mean(axis=0), [0.0, 0.5, 0.0], atol=0.05)
    assert np.allclose(deform_mesh(np.zeros(3), fk, md), verts, atol=1e-6)
