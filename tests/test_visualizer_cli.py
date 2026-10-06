import numpy as np
import pytest
from conftest import PANDA_ARM

from jax_ik.ik import FKSolver, InverseKinematicsSolver
from jax_ik.visualizer import (
    MeshPoser,
    TrajectoryPlanner,
    _smplx_preset,
    default_controlled_bones,
    load_mesh_data,
    main,
)


@pytest.fixture()
def offscreen():
    pv = pytest.importorskip("pyvista")
    old = pv.OFF_SCREEN
    pv.OFF_SCREEN = True
    yield pv
    pv.OFF_SCREEN = old


def test_default_controlled_bones_tiny_chain(tiny_urdf):
    fk = FKSolver(tiny_urdf, controlled_bones=[], do_compute_sdf=False)
    # Fixed joints are skipped, root first
    assert default_controlled_bones(fk.skeleton, "tip") == ["l1", "l2", "slider"]
    assert default_controlled_bones(fk.skeleton, "l1") == ["l1"]
    assert default_controlled_bones(fk.skeleton, "base") == []


def test_smplx_preset(model_path):
    fk = FKSolver(model_path, controlled_bones=[], do_compute_sdf=False)
    for hand in ("left", "right"):
        bones, bounds, ee = _smplx_preset(fk, hand)
        assert bones == [f"{hand}_{b}" for b in ("collar", "shoulder", "elbow", "wrist")]
        assert len(bounds) == 12 and ee == f"{hand}_wrist"
        assert all(lo <= hi for lo, hi in bounds)


def test_skeleton_lines_single_bone():
    class _FK:
        bind_fk = np.eye(4)[None]
        parent_list = [-1]
        controlled_indices = []

        def compute_fk_from_angles(self, a):
            return self.bind_fk

    md = {"vertices": np.zeros((1, 4)), "vertex_assignment": np.zeros(1, int), "faces": np.zeros((0, 3))}
    poser = MeshPoser(_FK(), md)
    assert poser.skeleton_lines(np.zeros((1, 3))).size == 0


def test_trajectory_planner_edge_cases(tiny_urdf):
    solver = InverseKinematicsSolver(
        tiny_urdf, controlled_bones=["l1", "l2", "slider"], num_steps=300, compute_sdf=False
    )
    planner = TrajectoryPlanner(solver, ["tip"], ik_points=4, seed=0)
    # Before any plan the trajectory is just the current pose
    assert planner.trajectory.shape == (1, 9)
    assert np.allclose(planner.sample(0.7), planner.pose)
    # Progress outside [0, 1] is clamped
    traj = planner.plan(planner.random_targets())
    assert np.allclose(planner.sample(-1.0), traj[0])
    assert np.allclose(planner.sample(5.0), traj[-1])
    # Random targets come from legal poses, so they are reachable
    lo, hi = np.asarray(solver.lower_bounds), np.asarray(solver.upper_bounds)
    assert np.all(traj >= lo - 1e-6) and np.all(traj <= hi + 1e-6)


def test_trajectory_planner_multiple_end_effectors(tiny_urdf):
    solver = InverseKinematicsSolver(
        tiny_urdf, controlled_bones=["l1", "l2", "slider"], num_steps=300, compute_sdf=False
    )
    planner = TrajectoryPlanner(solver, ["l2", "tip"], ik_points=3, seed=1)
    targets = planner.random_targets()
    assert len(targets) == 2
    planner.plan(targets)
    paths = planner.end_effector_paths()
    assert len(paths) == 2 and all(p.shape == (4, 3) for p in paths)


def test_main_list_bones(tiny_urdf, capsys):
    main([tiny_urdf, "--list_bones"])
    out = capsys.readouterr().out
    assert "l1 revolute" in out and "slider prismatic" in out and "tip fixed" in out


def test_main_requires_end_effector(tiny_urdf):
    with pytest.raises(SystemExit):
        main([tiny_urdf])


def test_main_gltf_requires_controlled_bones(tmp_path):
    import trimesh

    scene = trimesh.Scene()
    scene.add_geometry(trimesh.creation.box(), node_name="pelvis", geom_name="box")
    path = tmp_path / "box.glb"
    scene.export(str(path))
    with pytest.raises(SystemExit):
        main([str(path), "--end_effector", "pelvis"])


def test_main_runs_both_modes_offscreen(tiny_urdf, offscreen, capsys):
    main([tiny_urdf, "--end_effector", "tip"])
    main([tiny_urdf, "--end_effector", "tip", "--random_targets", "1", "--ik_points", "3"])
    out = capsys.readouterr().out
    assert "Controlled bones: ['l1', 'l2', 'slider']" in out


def test_visualizer_skeleton_only(panda_urdf, offscreen):
    from jax_ik.visualizer import RealtimeVisualizer

    fk = FKSolver(panda_urdf, controlled_bones=PANDA_ARM, do_compute_sdf=False)
    viz = RealtimeVisualizer(fk, mesh_data=load_mesh_data(fk), show_mesh=False)
    assert viz.mesh is None and viz.skeleton is not None
    viz.set_pose(np.full(len(PANDA_ARM) * 3, 0.2, np.float32))
    viz.set_status("hello")
    viz.plotter.close()
