"""Panda arm trajectories with soft and hard self-collision handling.

Pass requirement: no link pair of the Panda may ever collide, checked on
every trajectory frame and on densely interpolated poses between frames
(the motion the arm actually makes when a trajectory is played back).
Clearances are conservative lower bounds on the true mesh distance, so a
non-negative clearance means the real meshes do not touch.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import PANDA_ARM

from jax_ik.collision import CollisionModel, SelfCollisionObj
from jax_ik.ik import InverseKinematicsSolver
from jax_ik.objectives import CombinedDerivativeObj, DistanceObjTraj

MARGIN = 0.005  # hard mode buffer, meters
IK_POINTS = 10
N_TRAJECTORIES = 6
N_PROVOKING = 6
DENSE = 15  # interpolated poses checked per trajectory segment


@pytest.fixture(scope="module")
def setup(panda_urdf):
    solver = InverseKinematicsSolver(
        panda_urdf,
        controlled_bones=PANDA_ARM,
        threshold=1e-6,
        num_steps=300,
        compute_sdf=False,
    )
    fk = solver.fk_solver
    model = CollisionModel.cached(fk, lower=solver.lower_bounds, upper=solver.upper_bounds)
    clear = jax.jit(jax.vmap(lambda q: jnp.min(model.clearances(q, fk))))
    lo, hi = np.asarray(solver.lower_bounds), np.asarray(solver.upper_bounds)
    start = ((lo + hi) / 2).astype(np.float32)
    assert float(clear(start[None])[0]) >= MARGIN  # a collision-free start
    return solver, model, clear, start


def _ee(solver, q):
    fk = solver.fk_solver
    out = fk.compute_fk_from_angles(jnp.asarray(q, jnp.float32))
    return np.asarray(fk.get_bone_head_tail_from_fk(out, "panda_hand_tcp")[0])


def _dense(traj):
    """Every frame plus DENSE interpolated poses per segment."""
    t = np.linspace(0.0, 1.0, DENSE + 1)[:-1, None]
    segments = [traj[i] * (1 - t) + traj[i + 1] * t for i in range(len(traj) - 1)]
    return np.concatenate(segments + [traj[-1:]]).astype(np.float32)


def _reachable_targets(solver, n, seed):
    """Hand positions of random poses inside the joint limits."""
    rng = np.random.default_rng(seed)
    lo, hi = np.asarray(solver.lower_bounds), np.asarray(solver.upper_bounds)
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    return [_ee(solver, np.clip(mid + rng.uniform(-0.6, 0.6, mid.shape) * half, lo, hi)) for _ in range(n)]


def _provoking_targets(solver, clear, n, seed):
    """Hand positions of random poses that collide with themselves: the
    straightforward way to reach these runs the arm into itself."""
    rng = np.random.default_rng(seed)
    lo, hi = np.asarray(solver.lower_bounds), np.asarray(solver.upper_bounds)
    out = []
    while len(out) < n:
        q = rng.uniform(lo, hi).astype(np.float32)
        if float(clear(q[None])[0]) < -0.01:
            out.append(_ee(solver, q))
    return out


def _objective(model, mode):
    if mode == "off":
        return []
    return [SelfCollisionObj(model, margin=MARGIN, hard=(mode == "hard"))]


def _run_trajectories(setup, mode, targets):
    solver, model, clear, start = setup
    pose, worst, misses = start, np.inf, []
    for target in targets:
        traj, _, _ = solver.solve(
            initial_rotations=pose,
            learning_rate=0.05,
            mandatory_objective_functions=[DistanceObjTraj("panda_hand_tcp", [target], True)],
            optional_objective_functions=[CombinedDerivativeObj(3, weight=0.05)]
            + _objective(model, mode),
            ik_points=IK_POINTS,
            patience=200,
            verbose=False,
        )
        traj = np.asarray(traj)
        worst = min(worst, float(np.min(clear(jnp.asarray(_dense(traj))))))
        misses.append(np.linalg.norm(_ee(solver, traj[-1]) - target))
        pose = traj[-1]  # the next trajectory continues from here
    return worst, np.asarray(misses)


@pytest.mark.parametrize("mode", ["soft", "hard"])
def test_trajectories_never_collide(setup, mode):
    solver, model, clear, _ = setup
    targets = _reachable_targets(solver, N_TRAJECTORIES, seed=11)
    worst, misses = _run_trajectories(setup, mode, targets)
    assert worst >= 0.0, f"{mode}: links collided (clearance {worst * 1000:.1f} mm)"
    if mode == "hard":
        assert worst >= MARGIN - 1e-5
    # Collision handling must not stop the arm from doing its job
    assert np.median(misses) < 0.02, misses


@pytest.mark.parametrize("mode", ["soft", "hard"])
def test_provoking_trajectories_never_collide(setup, mode):
    solver, model, clear, _ = setup
    targets = _provoking_targets(solver, clear, N_PROVOKING, seed=3)
    worst, _ = _run_trajectories(setup, mode, targets)
    assert worst >= 0.0, f"{mode}: links collided (clearance {worst * 1000:.1f} mm)"
    if mode == "hard":
        assert worst >= MARGIN - 1e-5


def test_provoking_targets_do_collide_without_handling(setup):
    # Makes sure the tests above actually exercise collisions
    solver, model, clear, _ = setup
    targets = _provoking_targets(solver, clear, N_PROVOKING, seed=3)
    worst, _ = _run_trajectories(setup, "off", targets)
    assert worst < 0.0


@pytest.mark.parametrize("mode", ["soft", "hard"])
def test_single_pose_solves_never_collide(setup, mode):
    solver, model, clear, start = setup
    targets = _provoking_targets(solver, clear, N_PROVOKING, seed=7)
    for target in targets:
        result, _, _ = solver.solve(
            initial_rotations=start,
            learning_rate=0.05,
            mandatory_objective_functions=[DistanceObjTraj("panda_hand_tcp", [target], True)],
            optional_objective_functions=_objective(model, mode),
            patience=50,
            verbose=False,
        )
        worst = float(np.min(clear(jnp.asarray(_dense(np.asarray(result))))))
        assert worst >= 0.0, f"{mode}: links collided (clearance {worst * 1000:.1f} mm)"
        if mode == "hard":
            assert worst >= MARGIN - 1e-5
