import os

import numpy as np
import pytest

from jax_ik.helper import deform_mesh
from jax_ik.ik import FKSolver, InverseKinematicsSolver
from jax_ik.visualizer import (
    IKController,
    MeshPoser,
    default_controlled_bones,
    load_mesh_data,
)

PANDA_URDF = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "models", "panda", "urdf", "panda.urdf"
    )
)
PANDA_ARM = [f"panda_link{i}" for i in range(1, 8)]

pytestmark = pytest.mark.skipif(
    not os.path.exists(PANDA_URDF), reason="Panda URDF not available"
)


@pytest.fixture(scope="module")
def panda_solver():
    return InverseKinematicsSolver(
        PANDA_URDF,
        controlled_bones=PANDA_ARM,
        threshold=1e-6,
        num_steps=100,
        compute_sdf=False,
    )


@pytest.fixture(scope="module")
def panda_mesh(panda_solver):
    return load_mesh_data(panda_solver.fk_solver)


def test_default_controlled_bones_follow_chain():
    probe = FKSolver(PANDA_URDF, controlled_bones=[], do_compute_sdf=False)
    assert default_controlled_bones(probe.skeleton, "panda_hand_tcp") == PANDA_ARM
    # Mimic finger is driven by its master, so it is never picked
    assert default_controlled_bones(probe.skeleton, "panda_rightfinger") == PANDA_ARM
    assert default_controlled_bones(probe.skeleton, "panda_leftfinger") == (
        PANDA_ARM + ["panda_leftfinger"]
    )
    with pytest.raises(ValueError):
        default_controlled_bones(probe.skeleton, "no_such_bone")


def test_mesh_poser_matches_deform_mesh(panda_solver, panda_mesh):
    fk_solver = panda_solver.fk_solver
    poser = MeshPoser(fk_solver, panda_mesh)
    angles = np.linspace(-0.5, 0.5, len(PANDA_ARM) * 3).astype(np.float32)

    verts, heads = poser.pose(angles)
    expected = np.asarray(deform_mesh(angles, fk_solver, panda_mesh))
    assert verts.shape == expected.shape
    assert np.allclose(verts, expected, atol=1e-5)

    fk = np.asarray(fk_solver.compute_fk_from_angles(angles))
    assert np.allclose(heads, fk[:, :3, 3], atol=1e-6)

    lines = poser.skeleton_lines(heads).reshape(-1, 3)
    # One segment per non-root bone, each from parent to child
    assert len(lines) == len(fk_solver.bone_names) - 1
    assert np.all(lines[:, 0] == 2)


def test_mesh_poser_lbs_matches_deform_mesh(model_path):
    fk_solver = FKSolver(
        model_path, controlled_bones=["left_shoulder", "left_elbow"], do_compute_sdf=False
    )
    mesh_data = load_mesh_data(fk_solver)
    angles = np.array([0.3, -0.2, 0.5, 0.0, 0.4, -0.3], dtype=np.float32)
    verts, _ = MeshPoser(fk_solver, mesh_data).pose(angles)
    expected = np.asarray(deform_mesh(angles, fk_solver, mesh_data))
    assert np.allclose(verts, expected, atol=1e-4)


def test_ik_controller_follows_moving_target(panda_solver):
    mid = (
        np.asarray(panda_solver.lower_bounds) + np.asarray(panda_solver.upper_bounds)
    ) / 2
    controller = IKController(panda_solver, ["panda_hand_tcp"], initial_angles=mid)
    start = controller.end_effector_positions()[0]

    # Move the target in small increments, like a dragged sphere
    for t in np.linspace(0.0, 1.0, 10):
        target = start + t * np.array([0.1, -0.15, 0.1], dtype=np.float32)
        controller.step([target])
    for _ in range(5):
        controller.step([target])

    reached = controller.end_effector_positions()[0]
    assert np.linalg.norm(reached - target) < 5e-3

    # Holding the target still must not make the redundant arm wander
    prev = controller.angles
    for _ in range(10):
        controller.step([target])
        assert np.abs(controller.angles - prev).max() < 0.05
        prev = controller.angles
    assert np.all(controller.angles >= np.asarray(panda_solver.lower_bounds) - 1e-6)
    assert np.all(controller.angles <= np.asarray(panda_solver.upper_bounds) + 1e-6)


def test_ik_controller_starts_inside_bounds(panda_solver):
    # All-zero is outside the Panda's 4th joint range
    controller = IKController(panda_solver, ["panda_hand_tcp"])
    assert np.all(controller.angles >= np.asarray(panda_solver.lower_bounds))
    assert np.all(controller.angles <= np.asarray(panda_solver.upper_bounds))


def test_visualizer_updates_mesh_offscreen(panda_solver, panda_mesh):
    pv = pytest.importorskip("pyvista")
    pv.OFF_SCREEN = True
    from jax_ik.visualizer import RealtimeVisualizer

    try:
        viz = RealtimeVisualizer(panda_solver.fk_solver, mesh_data=panda_mesh)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"No offscreen rendering available: {e}")

    before = np.array(viz.mesh.points)
    angles = np.zeros(len(PANDA_ARM) * 3, dtype=np.float32)
    angles[0] = 0.8  # rotate the first joint
    viz.set_pose(angles)
    after = np.array(viz.mesh.points)
    assert not np.allclose(before, after)
    assert np.allclose(viz.skeleton.points, viz.poser.pose(angles)[1])

    # Streaming mode
    angles[0] = -0.8
    viz.update(angles)
    assert np.allclose(viz.mesh.points, viz.poser.pose(angles)[0])

    controller = IKController(panda_solver, ["panda_hand_tcp"])
    viz.run_interactive(controller)  # offscreen show returns right away
    viz.plotter.close()
