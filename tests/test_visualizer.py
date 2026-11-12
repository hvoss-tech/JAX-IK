import os
import time

import numpy as np
import pytest

from jax_ik.helper import deform_mesh
from jax_ik.ik import FKSolver, InverseKinematicsSolver
from jax_ik.visualizer import (
    IKController,
    MeshPoser,
    PlanPipeline,
    ReachMap,
    TrajectoryPlanner,
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


@pytest.fixture(scope="module")
def panda_traj_solver():
    return InverseKinematicsSolver(
        PANDA_URDF,
        controlled_bones=PANDA_ARM,
        threshold=1e-6,
        num_steps=1000,
        compute_sdf=False,
    )


def _mid(solver):
    return (np.asarray(solver.lower_bounds) + np.asarray(solver.upper_bounds)) / 2


def test_trajectory_planner_reaches_random_targets(panda_traj_solver):
    planner = TrajectoryPlanner(
        panda_traj_solver, ["panda_hand_tcp"], initial_angles=_mid(panda_traj_solver), seed=0
    )
    dists = []
    for _ in range(5):
        start = planner.pose.copy()
        targets = planner.random_targets()
        traj = planner.plan(targets)
        assert traj.shape == (planner.ik_points + 1, len(PANDA_ARM) * 3)
        # Frame 0 is the pose we start from
        assert np.allclose(traj[0], start)
        dists.append(np.linalg.norm(planner.end_effector_paths()[0][-1] - targets[0]))

        # Playback ends on the last frame, which seeds the next plan
        assert np.allclose(planner.sample(0.0), traj[0])
        # Halfway through 11 frames is exactly frame 5
        assert np.allclose(planner.sample(0.5), traj[5])
        assert np.allclose(planner.sample(1.0), traj[-1])
        assert np.allclose(planner.pose, traj[-1])
    assert np.median(dists) < 5e-3


def test_trajectory_smoothing_reduces_jumps(panda_traj_solver):
    jumps = {}
    for smoothing in (0.0, 0.05):
        planner = TrajectoryPlanner(
            panda_traj_solver,
            ["panda_hand_tcp"],
            smoothing=smoothing,
            initial_angles=_mid(panda_traj_solver),
            seed=1,
        )
        worst = 0.0
        for _ in range(4):
            traj = planner.plan(planner.random_targets())
            planner.sample(1.0)
            worst = max(worst, np.abs(np.diff(traj, axis=0)).max())
        jumps[smoothing] = worst
    assert jumps[0.05] < jumps[0.0]


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


def test_look_down_sets_elevation_and_distance(panda_solver, panda_mesh):
    pv = pytest.importorskip("pyvista")
    pv.OFF_SCREEN = True
    from jax_ik.visualizer import RealtimeVisualizer

    try:
        viz = RealtimeVisualizer(panda_solver.fk_solver, mesh_data=panda_mesh)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"No offscreen rendering available: {e}")
    cam = viz.plotter.camera
    focal = np.array(cam.focal_point)
    before = np.linalg.norm(np.array(cam.position) - focal)
    viz.look_down(elevation=45.0, zoom_out=1.5)
    offset = np.array(cam.position) - focal
    up = np.array(cam.up) / np.linalg.norm(cam.up)
    assert np.allclose(cam.focal_point, focal)
    assert np.isclose(np.linalg.norm(offset), 1.5 * before)
    elevation = np.degrees(np.arcsin(offset @ up / np.linalg.norm(offset)))
    assert np.isclose(elevation, 45.0, atol=1e-3)
    viz.plotter.close()


def test_orbit_around_moves_focal_point_onto_bone_axis(panda_solver, panda_mesh):
    pv = pytest.importorskip("pyvista")
    pv.OFF_SCREEN = True
    from jax_ik.visualizer import RealtimeVisualizer

    try:
        viz = RealtimeVisualizer(panda_solver.fk_solver, mesh_data=panda_mesh)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"No offscreen rendering available: {e}")
    cam = viz.plotter.camera
    up = np.array(cam.up) / np.linalg.norm(cam.up)
    before = np.array(cam.position) - np.array(cam.focal_point)
    height = np.array(cam.focal_point) @ up
    viz.orbit_around("panda_link0")
    base = viz.poser.pose(viz._angles)[1][viz.fk_solver.bone_names.index("panda_link0")]
    focal = np.array(cam.focal_point)
    # On the vertical axis through the base, at the same height
    off_axis = (focal - base) - ((focal - base) @ up) * up
    assert np.linalg.norm(off_axis) < 1e-6
    assert np.isclose(focal @ up, height)
    # Same view direction and distance
    assert np.allclose(np.array(cam.position) - focal, before)
    # Turning the camera keeps it on a circle around that axis
    radius = np.linalg.norm(before - (before @ up) * up)
    cam.Azimuth(70)
    offset = np.array(cam.position) - focal
    assert np.isclose(np.linalg.norm(offset - (offset @ up) * up), radius)
    assert np.isclose(offset @ up, before @ up)
    viz.plotter.close()


def test_trajectory_mode_offscreen(panda_traj_solver, panda_mesh):
    pv = pytest.importorskip("pyvista")
    pv.OFF_SCREEN = True
    from jax_ik.visualizer import RealtimeVisualizer

    try:
        viz = RealtimeVisualizer(panda_traj_solver.fk_solver, mesh_data=panda_mesh)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"No offscreen rendering available: {e}")
    planner = TrajectoryPlanner(
        panda_traj_solver, ["panda_hand_tcp"], initial_angles=_mid(panda_traj_solver)
    )
    viz.run_trajectory(planner, interval=5.0, playback=2.0)
    # The first target was planned and the pose set to its start
    assert len(planner.trajectory) == planner.ik_points + 1
    assert np.allclose(viz.mesh.points, viz.poser.pose(planner.pose)[0], atol=1e-5)
    viz.plotter.close()


@pytest.fixture(scope="module")
def panda_reach(panda_traj_solver):
    return ReachMap(panda_traj_solver, ["panda_hand_tcp"], samples=2048, seed=0)


def test_reach_map_positions_come_from_fk(panda_traj_solver, panda_reach):
    planner = TrajectoryPlanner(panda_traj_solver, ["panda_hand_tcp"])
    for i in (0, 100, 2047):
        assert np.allclose(
            panda_reach.positions[i, 0],
            planner.end_effector_positions(panda_reach.poses[i])[0],
            atol=1e-5,
        )
    assert np.all(panda_reach.poses >= np.asarray(panda_traj_solver.lower_bounds))
    assert np.all(panda_reach.poses <= np.asarray(panda_traj_solver.upper_bounds))
    # No collision model: every sample counts
    assert panda_reach.free().all()


def test_reach_map_picks_far_reachable_targets(panda_reach):
    rng = np.random.default_rng(0)
    current = [panda_reach.positions[0, 0]]
    dist = np.linalg.norm(panda_reach.positions[:, 0] - current[0], axis=1)
    for _ in range(20):
        target = panda_reach.pick(current, rng, far=0.3)[0]
        # Among the furthest 30%, give or take the interpolation
        assert np.linalg.norm(target - current[0]) >= np.quantile(dist, 0.7) - 1.5 * panda_reach.cell
        # Next to a sample, so reachable
        near = np.linalg.norm(panda_reach.positions[:, 0] - target, axis=1).min()
        assert near <= 1.5 * panda_reach.cell


def test_planner_targets_measured_from_start(panda_traj_solver, panda_reach):
    planner = TrajectoryPlanner(
        panda_traj_solver,
        ["panda_hand_tcp"],
        initial_angles=_mid(panda_traj_solver),
        seed=0,
        reach_map=panda_reach,
        far=0.2,
    )
    start = panda_reach.poses[5]
    origin = planner.end_effector_positions(start)[0]
    dist = np.linalg.norm(panda_reach.positions[:, 0] - origin, axis=1)
    for _ in range(10):
        target = planner.random_targets(start=start)[0]
        assert np.linalg.norm(target - origin) >= np.quantile(dist, 0.8) - 1.5 * panda_reach.cell


def test_planner_guess_blends_to_a_map_pose(panda_traj_solver, panda_reach):
    planner = TrajectoryPlanner(
        panda_traj_solver,
        ["panda_hand_tcp"],
        initial_angles=_mid(panda_traj_solver),
        seed=0,
        reach_map=panda_reach,
    )
    start = planner.pose.copy()
    targets = planner.random_targets()
    guess = planner.initial_guess(start, targets)
    assert guess.shape == (planner.ik_points + 1, len(start))
    assert np.allclose(guess[0], start)
    assert np.allclose(guess[-1], panda_reach.poses_near(targets, 1)[0])
    traj = planner.plan(targets)
    assert np.allclose(traj[0], start)
    assert np.linalg.norm(planner.end_effector_paths()[0][-1] - targets[0]) < 5e-3


def test_planner_solves_between_end_poses(panda_traj_solver, panda_reach):
    planner = TrajectoryPlanner(
        panda_traj_solver,
        ["panda_hand_tcp"],
        initial_angles=_mid(panda_traj_solver),
        seed=0,
        reach_map=panda_reach,
    )
    start = planner.pose.copy()
    targets = planner.random_targets()
    end, miss, _ = planner.solve_end_pose(start, targets)
    assert miss < 5e-3
    assert np.linalg.norm(planner.end_effector_positions(end)[0] - targets[0]) == pytest.approx(miss)
    guess = planner.path_guess(start, end)
    # Nothing to collide with: the straight blend
    assert np.allclose(guess, start + np.linspace(0, 1, len(guess))[:, None] * (end - start))
    traj, _, _, _ = planner.solve_between(guess)
    assert planner.trajectory_clear(traj)
    assert traj.shape == (planner.ik_points + 1, len(start))
    assert np.allclose(traj[0], start)
    assert np.allclose(traj[-1], end)


def test_plan_pipeline_chains_parallel_plans(panda_traj_solver, panda_reach):
    planner = TrajectoryPlanner(
        panda_traj_solver,
        ["panda_hand_tcp"],
        initial_angles=_mid(panda_traj_solver),
        seed=0,
        reach_map=panda_reach,
    )
    pipeline = PlanPipeline(planner, buffer=4, candidates=2, max_miss=5e-3)
    try:
        # Fill the buffer before anything is played
        deadline = time.perf_counter() + 120
        while len(pipeline._ahead) < 4 and time.perf_counter() < deadline:
            pipeline.poll()
            time.sleep(0.01)
        assert len(pipeline._ahead) == 4
        start = planner.pose.copy()
        for _ in range(4):
            while not pipeline.ready():
                pipeline.poll()
                time.sleep(0.01)
            targets, (traj, _, _, _) = pipeline.pop()
            # Each plan starts exactly where the one before it ends
            assert np.allclose(traj[0], start)
            assert np.linalg.norm(planner.end_effector_paths(traj)[0][-1] - targets[0]) < 5e-3
            start = traj[-1]
        # Planning carries on from the last one played
        pipeline.poll()
        assert pipeline._ending
        pipeline.restart(start)
        assert not pipeline._ahead and not pipeline._ending
    finally:
        pipeline.shutdown()


class _StubPlanner:
    """Poses are single numbers, each target one further than the start;
    the trajectories with the given call numbers come out colliding."""

    ik_points = 2

    def __init__(self, colliding):
        self.trajectory = np.zeros((1, 1), np.float32)
        self._rng = np.random.default_rng(0)
        self.colliding = set(colliding)
        self.bad = []
        self.calls = 0

    def random_targets(self, start):
        return [start + 1]

    def solve_end_pose(self, start, targets):
        return np.asarray(targets[0], np.float32), 0.0, 1.0

    def path_guess(self, start, end, rng=None):
        return np.stack([start, (start + end) / 2, end])

    def solve_between(self, guess):
        return guess.copy(), 0.0, 1, 1.0

    def trajectory_clear(self, traj):
        self.calls += 1
        if self.calls in self.colliding:
            self.bad.append(traj)
            return False
        return True


def _drain(pipeline, count):
    plans = []
    deadline = time.perf_counter() + 30
    while len(plans) < count and time.perf_counter() < deadline:
        pipeline.poll()
        if pipeline.ready():
            plans.append(pipeline.pop()[1][0])
        time.sleep(0.001)
    return plans


def test_plan_pipeline_replans_colliding_trajectories():
    planner = _StubPlanner(colliding={2, 3, 5})
    # One attempt each: every collision drops the plans after it
    pipeline = PlanPipeline(planner, buffer=3, candidates=2, attempts=1)
    try:
        plans = _drain(pipeline, 6)
    finally:
        pipeline.shutdown()
    assert len(plans) == 6
    assert len(planner.bad) == 3
    # Nothing colliding was played, and every plan picks up where the one
    # before it ends
    assert not any(any(p is b for b in planner.bad) for p in plans)
    assert np.allclose([p[0, 0] for p in plans], np.arange(6))
    assert np.allclose([p[-1, 0] for p in plans], np.arange(1, 7))


def test_plan_pipeline_plays_colliding_plans_after_max_retries():
    planner = _StubPlanner(colliding=range(1, 1000))
    pipeline = PlanPipeline(planner, buffer=2, max_retries=3, attempts=1)
    try:
        plans = _drain(pipeline, 1)
    finally:
        pipeline.shutdown()
    assert len(plans) == 1
    assert len(planner.bad) >= 4


def test_plan_pipeline_retries_a_colliding_trajectory_in_place():
    planner = _StubPlanner(colliding={1, 2})
    pipeline = PlanPipeline(planner, buffer=4, attempts=3)
    try:
        deadline = time.perf_counter() + 30
        while pipeline.planned < 4 and time.perf_counter() < deadline:
            pipeline.poll()
            time.sleep(0.001)
        # The two colliding solves were redone in place: their end poses
        # stayed, so nothing after them was dropped
        assert pipeline.planned == 4
        assert planner.calls == 6
        assert sum(slot["attempts"] for slot in pipeline._ahead) == 6
        plans = _drain(pipeline, 4)
    finally:
        pipeline.shutdown()
    assert np.allclose([p[0, 0] for p in plans], np.arange(4))
