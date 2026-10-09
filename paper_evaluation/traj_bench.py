"""Trajectory IK benchmark with self-collision.

Panda arm solving 10-point trajectories (DistanceObjTraj on the last frame,
CombinedDerivativeObj smoothing, SelfCollisionObj off/soft/hard), half of the
targets "provoking" (hand positions of self-colliding poses). Every
trajectory starts from its own random collision-free pose, so one bad solve
doesn't change the start of the next. Reports per-trajectory wall time,
steps, end miss, worst clearance over densely interpolated poses (exact,
every proxy point) and a smoothness measure.

Run from the repository root:

    uv run python paper_evaluation/traj_bench.py [--modes off,soft,hard] [--n 20]

Set JAX_IK_SRC to run against another checkout's src directory.
"""

import argparse
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.environ.get("JAX_IK_SRC", os.path.join(ROOT, "src")))
# Imported before collision_bench, which puts this repository's src first
import jax_ik.collision  # noqa: E402, F401
import jax_ik.ik  # noqa: E402, F401

from collision_bench import (  # noqa: E402
    BONES,
    EE,
    MARGIN,
    URDF,
    dense,
    ee,
    make_exact_clearance,
    provoking_targets,
    reachable_targets,
)
from jax_ik.collision import CollisionModel, SelfCollisionObj  # noqa: E402
from jax_ik.ik import InverseKinematicsSolver  # noqa: E402
from jax_ik.objectives import CombinedDerivativeObj, DistanceObjTraj  # noqa: E402

IK_POINTS = 10
N_TRAJ = 20  # trajectories per kind (reachable / provoking)
REPEATS = 3


def start_poses(solver, clear, n, seed):
    """Random poses with at least MARGIN clearance, around the middle of the
    joint ranges like the poses an arm ends up in."""
    rng = np.random.default_rng(seed)
    lo, hi = np.asarray(solver.lower_bounds), np.asarray(solver.upper_bounds)
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    out = []
    while len(out) < n:
        q = (mid + rng.uniform(-0.6, 0.6, mid.shape) * half).astype(np.float32)
        if float(clear(q[None])[0]) >= MARGIN:
            out.append(q)
    return out


def run(solver, model, mode, jobs):
    coll = [] if mode == "off" else [SelfCollisionObj(model, margin=MARGIN, hard=(mode == "hard"))]
    out = []
    for start, target in jobs:
        t0 = time.perf_counter()
        traj, _, steps = solver.solve(
            initial_rotations=start,
            learning_rate=0.05,
            mandatory_objective_functions=[DistanceObjTraj(EE, [target], True)],
            optional_objective_functions=[CombinedDerivativeObj(3, weight=0.05)] + coll,
            ik_points=IK_POINTS,
            patience=200,
            verbose=False,
        )
        out.append((time.perf_counter() - t0, steps, np.asarray(traj), target))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modes", default="off,soft,hard")
    p.add_argument("--repeats", type=int, default=REPEATS)
    p.add_argument("--n", type=int, default=N_TRAJ, help="trajectories per kind")
    args = p.parse_args()

    jax.config.update("jax_default_device", jax.devices("cpu")[0])
    solver = InverseKinematicsSolver(
        URDF, controlled_bones=BONES, threshold=1e-6, num_steps=300, compute_sdf=False
    )
    fk = solver.fk_solver
    model = CollisionModel.cached(fk, lower=solver.lower_bounds, upper=solver.upper_bounds)
    clear = make_exact_clearance(model, fk)
    n = args.n
    targets = reachable_targets(solver, n, seed=103) + provoking_targets(solver, clear, n, seed=104)
    jobs = list(zip(start_poses(solver, clear, 2 * n, seed=105), targets))

    for mode in args.modes.split(","):
        t0 = time.perf_counter()
        run(solver, model, mode, jobs[:1])
        compile_s = time.perf_counter() - t0
        runs = [run(solver, model, mode, jobs) for _ in range(args.repeats)]
        times = np.min([[r[0] for r in rs] for rs in runs], axis=0)
        res = runs[0]
        steps = np.array([r[1] for r in res])
        miss = np.array([np.linalg.norm(ee(solver, r[2][-1]) - r[3]) for r in res])
        clr = np.array([float(jnp.min(clear(jnp.asarray(dense(r[2]))))) for r in res])
        limit = MARGIN - 1e-5 if mode == "hard" else 0.0
        # Mean squared joint-space acceleration along the trajectories
        accel = np.mean([np.mean(np.square(np.diff(r[2], 2, axis=0))) for r in res])
        print(
            f"{mode:4s} compile {compile_s:5.1f}s | traj ms median {np.median(times) * 1e3:7.1f} "
            f"mean {np.mean(times) * 1e3:7.1f} | steps {steps.mean():5.1f} | "
            f"us/step {np.sum(times) / steps.sum() * 1e6:6.0f} | "
            f"miss mm med {np.median(miss) * 1e3:6.2f} ok {np.mean(miss < 0.02) * 100:5.1f}% | "
            f"worst clr mm {clr.min() * 1e3:7.2f} coll {int(np.sum(clr < limit)):2d} | "
            f"accel {accel:.2e}"
        )


if __name__ == "__main__":
    main()
