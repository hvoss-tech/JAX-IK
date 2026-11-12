import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import PANDA_ARM

from jax_ik import ik
from jax_ik.ik import InverseKinematicsSolver, _matmul4, solve_ik
from jax_ik.objectives import (
    BoneZeroRotationObj,
    CombinedDerivativeObj,
    DistanceObjTraj,
    InitPoseObj,
    SphereCollisionPenaltyObjTraj,
)

TINY_BONES = ["l1", "l2", "slider"]


@pytest.fixture(scope="module")
def tiny_solver(tiny_urdf):
    return InverseKinematicsSolver(
        tiny_urdf,
        controlled_bones=TINY_BONES,
        threshold=1e-7,
        num_steps=400,
        compute_sdf=False,
    )


def _tip(solver, angles):
    fk = solver.fk_solver
    out = fk.compute_fk_from_angles(np.asarray(angles, np.float32))
    return np.asarray(fk.get_bone_head_tail_from_fk(out, "tip")[0])


def _reach(target):
    return [DistanceObjTraj("tip", [target], use_head=True)]


# --- bounds -----------------------------------------------------------------


def test_urdf_limits_become_bounds(tiny_solver):
    lower = np.asarray(tiny_solver.lower_bounds).reshape(-1, 3)
    upper = np.asarray(tiny_solver.upper_bounds).reshape(-1, 3)
    assert np.allclose(lower[:, 0], [-1.5, -2.0, 0.0], atol=1e-5)
    assert np.allclose(upper[:, 0], [1.5, 0.5, 0.2], atol=1e-5)
    assert np.all(lower[:, 1:] == 0) and np.all(upper[:, 1:] == 0)


def test_user_bounds_are_degrees_except_prismatic(tiny_urdf):
    bounds = [(-90, 90), (0, 0), (0, 0), (-45, 10), (0, 0), (0, 0), (0.0, 0.1), (0, 0), (0, 0)]
    solver = InverseKinematicsSolver(
        tiny_urdf, controlled_bones=TINY_BONES, bounds=bounds, compute_sdf=False
    )
    assert np.allclose(np.asarray(solver.lower_bounds)[[0, 3, 6]], [-np.pi / 2, -np.pi / 4, 0.0], atol=1e-5)
    assert np.allclose(np.asarray(solver.upper_bounds)[[0, 3, 6]], [np.pi / 2, np.radians(10), 0.1], atol=1e-5)


def test_gltf_bounds_are_converted_to_radians(model_path):
    solver = InverseKinematicsSolver(
        model_path,
        controlled_bones=["left_elbow"],
        bounds=[(-90, 90), (-45, 45), (0, 180)],
        compute_sdf=False,
    )
    assert np.allclose(solver.upper_bounds, np.radians([90, 45, 180]), atol=1e-5)


# --- solve ------------------------------------------------------------------


def test_reaches_reachable_target(tiny_solver):
    goal = np.array([0.6, 0, 0, -0.9, 0, 0, 0.1, 0, 0], np.float32)
    target = _tip(tiny_solver, goal)
    result, loss, steps = tiny_solver.solve(
        mandatory_objective_functions=_reach(target), learning_rate=0.1, verbose=False
    )
    assert result.shape == (2, 9)
    assert np.linalg.norm(_tip(tiny_solver, result[-1]) - target) < 2e-3
    assert loss < 1e-6 and steps < tiny_solver.num_steps


def test_unreachable_target_stays_in_bounds(tiny_solver):
    result, loss, _ = tiny_solver.solve(
        mandatory_objective_functions=_reach([10.0, 10.0, 10.0]),
        learning_rate=0.1,
        verbose=False,
    )
    assert np.all(np.isfinite(result)) and np.isfinite(loss) and loss > 1.0
    assert np.all(result >= np.asarray(tiny_solver.lower_bounds) - 1e-6)
    assert np.all(result <= np.asarray(tiny_solver.upper_bounds) + 1e-6)


def test_already_at_target_stops_immediately(tiny_solver):
    start = np.zeros(9, np.float32)
    _, loss, steps = tiny_solver.solve(
        initial_rotations=start,
        mandatory_objective_functions=_reach(_tip(tiny_solver, start)),
        verbose=False,
    )
    assert steps == 1 and loss < 1e-7


def test_num_steps_caps_iterations(tiny_urdf):
    solver = InverseKinematicsSolver(
        tiny_urdf, controlled_bones=TINY_BONES, threshold=0.0, num_steps=7, compute_sdf=False
    )
    _, _, steps = solver.solve(
        mandatory_objective_functions=_reach([0.3, 0.1, 0.5]), patience=1000, verbose=False
    )
    assert steps == 7


def test_patience_stops_without_improvement(tiny_urdf):
    solver = InverseKinematicsSolver(
        tiny_urdf, controlled_bones=TINY_BONES, threshold=0.0, num_steps=500, compute_sdf=False
    )
    # Zero-weight objective: loss never improves after the first step
    _, _, steps = solver.solve(
        mandatory_objective_functions=[DistanceObjTraj("tip", [1, 1, 1], weight=0.0)],
        patience=5,
        verbose=False,
    )
    assert steps <= 7


def test_initial_rotations_none_and_ik_points(tiny_solver):
    result, _, _ = tiny_solver.solve(
        mandatory_objective_functions=_reach([0.2, 0, 0.9]), ik_points=4, verbose=False
    )
    assert result.shape == (5, 9)
    assert np.allclose(result[0], 0)  # frame 0 is the fixed start
    # ik_points < 1 is treated as 1
    result, _, _ = tiny_solver.solve(
        mandatory_objective_functions=_reach([0.2, 0, 0.9]), ik_points=0, verbose=False
    )
    assert result.shape == (2, 9)


def test_trajectory_initial_keeps_prefix(tiny_solver):
    prefix = np.random.default_rng(0).uniform(-0.3, 0.1, (3, 9)).astype(np.float32)
    prefix = np.clip(prefix, tiny_solver.lower_bounds, tiny_solver.upper_bounds)
    result, _, _ = tiny_solver.solve(
        initial_rotations=prefix,
        mandatory_objective_functions=_reach([0.2, 0, 0.9]),
        ik_points=2,
        verbose=False,
    )
    assert result.shape == (5, 9)
    assert np.allclose(result[:3], prefix)


def test_solve_guess_keeps_prefix_frames(tiny_solver):
    guess = np.zeros((4, 9), np.float32)
    guess[0, 0] = 0.4
    result, _, _ = tiny_solver.solve_guess(
        guess,
        learning_rate=0.05,
        mandatory_objective_functions=_reach([0.3, 0.0, 0.8]),
        prefix_len=2,
    )
    assert np.allclose(result[:2], guess[:2])
    assert not np.allclose(result[2:], guess[2:])


def test_solve_guess_keeps_suffix_frames(tiny_solver):
    guess = np.zeros((5, 9), np.float32)
    guess[0, 0] = 0.4
    guess[-1, 3] = -0.6
    result, _, _ = tiny_solver.solve_guess(
        guess,
        learning_rate=0.05,
        optional_objective_functions=[CombinedDerivativeObj(max_order=2, weight=1.0)],
        suffix_len=1,
    )
    assert np.allclose(result[0], guess[0])
    assert np.allclose(result[-1], guess[-1])
    # Smoothing moves the frames in between towards the fixed ends
    assert not np.allclose(result[1:-1], guess[1:-1])
    with pytest.raises(ValueError):
        tiny_solver.solve_guess(guess[:2], prefix_len=1, suffix_len=1)


def test_optional_only_solve(tiny_solver):
    start = np.full(9, 0.1, np.float32)
    start = np.clip(start, tiny_solver.lower_bounds, tiny_solver.upper_bounds)
    result, loss, _ = tiny_solver.solve(
        initial_rotations=start,
        optional_objective_functions=[BoneZeroRotationObj()],
        learning_rate=0.05,
        verbose=False,
    )
    assert np.abs(result[-1]).max() < np.abs(start).max()
    assert np.isfinite(loss)


def test_smoothing_objective_in_trajectory(tiny_solver):
    target = _tip(tiny_solver, [0.8, 0, 0, -1.0, 0, 0, 0.1, 0, 0])
    rough, _, _ = tiny_solver.solve(
        mandatory_objective_functions=_reach(target), ik_points=8, learning_rate=0.05, verbose=False
    )
    smooth, _, _ = tiny_solver.solve(
        mandatory_objective_functions=_reach(target),
        optional_objective_functions=[CombinedDerivativeObj(2, weight=1.0)],
        ik_points=8,
        learning_rate=0.05,
        verbose=False,
    )
    assert np.abs(np.diff(smooth, n=2, axis=0)).sum() < np.abs(np.diff(rough, n=2, axis=0)).sum()


def test_too_many_objectives_raise(tiny_solver):
    with pytest.raises(ValueError, match="mandatory"):
        tiny_solver.solve(mandatory_objective_functions=[BoneZeroRotationObj()] * 11, verbose=False)
    with pytest.raises(ValueError, match="optional"):
        tiny_solver.solve(optional_objective_functions=[BoneZeroRotationObj()] * 11, verbose=False)


def test_solve_ik_mask_freezes_frames(tiny_solver):
    init = np.zeros((3, 9), np.float32)
    steps, best, loss, _ = solve_ik(
        init_rot=init,
        lower_bounds=tiny_solver.lower_bounds,
        upper_bounds=tiny_solver.upper_bounds,
        mandatory_obj_fns=_reach([0.3, 0.0, 0.8]),
        optional_obj_fns=[],
        fksolver=tiny_solver.fk_solver,
        threshold=1e-7,
        num_steps=50,
        learning_rate=0.05,
        mask=np.array([True, False, True]),
    )
    best = np.asarray(best)
    assert np.allclose(best[1], 0)
    assert not np.allclose(best[2], 0)


def test_solve_ik_default_mask_only_frees_last_frame(tiny_solver):
    init = np.zeros((3, 9), np.float32)
    _, best, _, _ = solve_ik(
        init,
        tiny_solver.lower_bounds,
        tiny_solver.upper_bounds,
        _reach([0.3, 0.0, 0.8]),
        [],
        tiny_solver.fk_solver,
        num_steps=30,
        learning_rate=0.05,
    )
    best = np.asarray(best)
    assert np.allclose(best[:2], 0) and not np.allclose(best[2], 0)


def test_solver_is_compiled_without_library_fusions(panda_urdf, monkeypatch):
    # Tiny reductions and matmuls handed to YNNPACK cost far more per step
    # than the arithmetic, so the solver loop is compiled without them
    from jax_ik.collision import CollisionModel, SelfCollisionObj

    solver = InverseKinematicsSolver(
        panda_urdf, controlled_bones=PANDA_ARM, num_steps=5, compute_sdf=False
    )
    model = CollisionModel.cached(
        solver.fk_solver, lower=solver.lower_bounds, upper=solver.upper_bounds
    )
    calls = []
    core = ik._solve_ik_core

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return core(*args, **kwargs)

    monkeypatch.setattr(ik, "_solve_ik_core", spy)
    solver.solve(
        mandatory_objective_functions=[DistanceObjTraj("panda_hand_tcp", [[0.3, -0.5, 0.3]], True)],
        optional_objective_functions=[SelfCollisionObj(model)],
        ik_points=3,
        verbose=False,
    )
    args, kwargs = calls[0]
    assert "ynn_fusion" not in core.lower(*args, **kwargs).compile().as_text()


# --- FK ------------------------------------------------------------------------


def test_matmul4_matches_matmul():
    rng = np.random.default_rng(0)
    a = jnp.asarray(rng.normal(size=(5, 3, 4, 4)).astype(np.float32))
    b = jnp.asarray(rng.normal(size=(3, 4, 4)).astype(np.float32))
    assert np.allclose(_matmul4(a, b), a @ b, atol=1e-5)
    # Gradients too, as the solver differentiates through FK
    ga = jax.grad(lambda a, b: jnp.sum(jnp.sin(_matmul4(a, b))), argnums=(0, 1))(a, b)
    gb = jax.grad(lambda a, b: jnp.sum(jnp.sin(a @ b)), argnums=(0, 1))(a, b)
    for x, y in zip(ga, gb):
        assert np.allclose(x, y, atol=1e-4)


def test_vmapped_fk_matches_per_pose_fk(panda_urdf):
    solver = InverseKinematicsSolver(panda_urdf, controlled_bones=PANDA_ARM, compute_sdf=False)
    fk = solver.fk_solver
    lo, hi = np.asarray(solver.lower_bounds), np.asarray(solver.upper_bounds)
    Q = jnp.asarray(np.random.default_rng(2).uniform(lo, hi, (6, lo.size)).astype(np.float32))
    batched = np.asarray(jax.vmap(fk.compute_fk_from_angles)(Q))
    for q, T in zip(Q, batched):
        assert np.allclose(T, np.asarray(fk.compute_fk_from_angles(q)), atol=1e-5)


# --- FK pruning cache --------------------------------------------------------


def test_fk_solver_for_prunes_and_caches(panda_urdf):
    solver = InverseKinematicsSolver(
        panda_urdf, controlled_bones=PANDA_ARM[:3], compute_sdf=False
    )
    objs = [DistanceObjTraj("panda_link4", [0, 0, 0], True)]
    pruned = solver._fk_solver_for(objs, [InitPoseObj(np.zeros(9))])
    assert pruned is not solver.fk_solver
    assert "panda_hand" not in pruned.bone_names and "panda_link4" in pruned.bone_names
    # Same bone set (different target) reuses the cached view
    again = solver._fk_solver_for([DistanceObjTraj("panda_link4", [1, 1, 1], True)], [])
    assert again is pruned


def test_fk_solver_for_full_skeleton_objectives(panda_urdf):
    solver = InverseKinematicsSolver(panda_urdf, controlled_bones=PANDA_ARM[:3], compute_sdf=False)
    sphere = SphereCollisionPenaltyObjTraj({"center": [0, 0, 0], "radius": 0.1})
    assert solver._fk_solver_for([], [sphere]) is solver.fk_solver
