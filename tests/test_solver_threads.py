"""solve() from several threads at once must give each caller its own answer.

solve_ik fills module-level objective pools before dispatching; without a
lock, two threads could swap targets between filling and dispatching.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import jax_ik.ik as ik
from jax_ik.objectives import DistanceObjTraj


def _solve(solver, start, target):
    result, _, _ = solver.solve(
        initial_rotations=start,
        learning_rate=0.15,
        mandatory_objective_functions=[
            DistanceObjTraj(bone_name="left_wrist", target_points=[target], use_head=True)
        ],
        ik_points=1,
        patience=30,
        verbose=False,
    )
    return result


def test_concurrent_solves_keep_their_targets(solver, zero_angles, monkeypatch):
    rng = np.random.default_rng(0)
    targets = [rng.uniform(-0.1, 0.1, 3).astype(np.float32) for _ in range(12)]
    expected = [_solve(solver, zero_angles, t) for t in targets]

    # Widen the window between filling the pools and dispatching, so other
    # threads would get to overwrite the pools in between
    core = ik._solve_ik_core

    def slow_core(*args, **kwargs):
        time.sleep(0.01)
        return core(*args, **kwargs)

    monkeypatch.setattr(ik, "_solve_ik_core", slow_core)

    with ThreadPoolExecutor(max_workers=4) as pool:
        got = list(pool.map(lambda t: _solve(solver, zero_angles, t), targets * 3))

    for i, result in enumerate(got):
        np.testing.assert_allclose(result, expected[i % len(targets)], atol=1e-5)
