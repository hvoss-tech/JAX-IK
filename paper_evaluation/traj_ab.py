"""Paired A/B timing of two collision.py variants on the trajectory benchmark.

Both files are loaded as separate modules in one process and alternate solve
by solve on the same jobs, so load from other processes hits both alike.
Prints the median per-job time ratio B/A and whether the results agree.

    uv run python paper_evaluation/traj_ab.py A/collision.py B/collision.py [--modes soft,hard]
"""

import argparse
import importlib.util
import os
import sys
import time

import jax
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from traj_bench import (  # noqa: E402
    EE,
    IK_POINTS,
    MARGIN,
    URDF,
    BONES,
    CollisionModel,
    CombinedDerivativeObj,
    DistanceObjTraj,
    InverseKinematicsSolver,
    make_exact_clearance,
    provoking_targets,
    reachable_targets,
    start_poses,
)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--modes", default="soft,hard")
    p.add_argument("--n", type=int, default=10, help="trajectories per kind")
    p.add_argument("--rounds", type=int, default=3)
    args = p.parse_args()

    jax.config.update("jax_default_device", jax.devices("cpu")[0])
    solver = InverseKinematicsSolver(
        URDF, controlled_bones=BONES, threshold=1e-6, num_steps=300, compute_sdf=False
    )
    fk = solver.fk_solver
    base = CollisionModel.cached(fk, lower=solver.lower_bounds, upper=solver.upper_bounds)
    clear = make_exact_clearance(base, fk)
    variants = []
    for i, path in enumerate((args.a, args.b)):
        mod = load(path, f"collision_variant_{i}")
        model = mod.CollisionModel.__new__(mod.CollisionModel)
        model.__dict__.update(base.__dict__)
        variants.append((mod, model))
    n = args.n
    targets = reachable_targets(solver, n, seed=103) + provoking_targets(solver, clear, n, seed=104)
    jobs = list(zip(start_poses(solver, clear, 2 * n, seed=105), targets))

    def solve(variant, mode, start, target):
        mod, model = variant
        coll = mod.SelfCollisionObj(model, margin=MARGIN, hard=(mode == "hard"))
        t0 = time.perf_counter()
        traj, _, steps = solver.solve(
            initial_rotations=start,
            learning_rate=0.05,
            mandatory_objective_functions=[DistanceObjTraj(EE, [target], True)],
            optional_objective_functions=[CombinedDerivativeObj(3, weight=0.05), coll],
            ik_points=IK_POINTS,
            patience=200,
            verbose=False,
        )
        return time.perf_counter() - t0, steps, np.asarray(traj)

    for mode in args.modes.split(","):
        for v in variants:
            solve(v, mode, *jobs[0])  # compile
        best = np.full((2, len(jobs)), np.inf)
        out = [[None] * len(jobs) for _ in range(2)]
        for _ in range(args.rounds):
            for j, job in enumerate(jobs):
                for i in ((0, 1) if j % 2 else (1, 0)):
                    dt, steps, traj = solve(variants[i], mode, *job)
                    best[i, j] = min(best[i, j], dt)
                    out[i][j] = (steps, traj)
        ratio = best[1] / best[0]
        same = sum(np.allclose(a[1], b[1], atol=1e-5) for a, b in zip(*out))
        steps = [sum(o[0] for o in out[i]) for i in range(2)]
        print(
            f"{mode:4s} B/A time median {np.median(ratio):.3f} "
            f"(q25 {np.quantile(ratio, 0.25):.3f} q75 {np.quantile(ratio, 0.75):.3f}) | "
            f"A {np.sum(best[0]) / steps[0] * 1e6:6.0f} us/step, B {np.sum(best[1]) / steps[1] * 1e6:6.0f} | "
            f"same result {same}/{len(jobs)}"
        )


if __name__ == "__main__":
    main()
