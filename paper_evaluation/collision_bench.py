"""Collision-enabled IK benchmark (frozen: do not edit while optimizing).

Panda arm with SelfCollisionObj in soft and hard mode on a fixed, seeded set
of single-pose solves and 10-point trajectories. About half of the targets
are "provoking": hand positions of self-colliding poses.

Scoring does not go through CollisionModel.evaluate/clearances (the code
being optimized). Clearances are recomputed here from the raw link proxies
(grids, sample points, radii) with every point of every checked pair looked
up, on every frame and on DENSE interpolated poses between frames.

Run from the repository root:

    uv run python paper_evaluation/collision_bench.py
"""

import os
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from jax_ik.collision import CollisionModel, SelfCollisionObj  # noqa: E402
from jax_ik.ik import InverseKinematicsSolver  # noqa: E402
from jax_ik.objectives import CombinedDerivativeObj, DistanceObjTraj  # noqa: E402

URDF = os.path.join(ROOT, "models", "panda", "urdf", "panda.urdf")
BONES = [f"panda_link{i}" for i in range(1, 8)]
EE = "panda_hand_tcp"
MARGIN = 0.005  # hard mode buffer, meters
HARD_TOL = 1e-5  # hard mode may not go below MARGIN - HARD_TOL
SUCCESS_DIST = 0.02
IK_POINTS = 10
DENSE = 15  # interpolated poses checked per segment
N_SINGLE = 12  # single-pose solves per kind (reachable / provoking) per mode
N_TRAJ = 5  # chained trajectories per kind per mode
REPEATS = 3  # timed repeats; each solve's time is its fastest repeat
TEST_FILES = [
    "tests/test_collision.py",
    "tests/test_collision_containment.py",
    "tests/test_panda_collision_trajectories.py",
]


def make_exact_clearance(model, fk_solver):
    """Minimum clearance over all checked link pairs for a batch of poses,
    using every sample point of both links in both directions."""
    links = model.links
    bone_idx = [fk_solver.bone_names.index(link.name) for link in links]
    spacing = float(model.spacing)

    def lookup(link, pts):
        coords = (pts - jnp.asarray(link.origin)) / spacing
        return jax.scipy.ndimage.map_coordinates(
            jnp.asarray(link.grid), coords.T, order=1, mode="constant", cval=link.outside
        )

    def one_pose(q):
        fk = fk_solver.compute_fk_from_angles(q)
        worst = jnp.float32(np.inf)
        for a, b in model.pairs:
            for src, dst in ((a, b), (b, a)):
                Ts, Td = fk[bone_idx[src]], fk[bone_idx[dst]]
                Rd, td = Td[:3, :3], Td[:3, 3]
                world = jnp.asarray(links[src].points) @ Ts[:3, :3].T + Ts[:3, 3]
                local = (world - td) @ Rd
                d = lookup(links[dst], local) - jnp.asarray(links[src].radii)
                worst = jnp.minimum(worst, jnp.min(d))
        return worst

    return jax.jit(jax.vmap(one_pose))


def dense(traj):
    t = np.linspace(0.0, 1.0, DENSE + 1)[:-1, None]
    segs = [traj[i] * (1 - t) + traj[i + 1] * t for i in range(len(traj) - 1)]
    return np.concatenate(segs + [traj[-1:]]).astype(np.float32)


def ee(solver, q):
    fk = solver.fk_solver
    out = fk.compute_fk_from_angles(jnp.asarray(q, jnp.float32))
    return np.asarray(fk.get_bone_head_tail_from_fk(out, EE)[0])


def reachable_targets(solver, n, seed):
    rng = np.random.default_rng(seed)
    lo, hi = np.asarray(solver.lower_bounds), np.asarray(solver.upper_bounds)
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    return [
        ee(solver, np.clip(mid + rng.uniform(-0.6, 0.6, mid.shape) * half, lo, hi))
        for _ in range(n)
    ]


def provoking_targets(solver, clear, n, seed):
    rng = np.random.default_rng(seed)
    lo, hi = np.asarray(solver.lower_bounds), np.asarray(solver.upper_bounds)
    out = []
    while len(out) < n:
        q = rng.uniform(lo, hi).astype(np.float32)
        if float(clear(q[None])[0]) < -0.01:
            out.append(ee(solver, q))
    return out


def build_jobs(solver, clear):
    """(kind, targets) groups; identical every run."""
    return [
        ("single", reachable_targets(solver, N_SINGLE, seed=101)),
        ("single", provoking_targets(solver, clear, N_SINGLE, seed=102)),
        ("traj", reachable_targets(solver, N_TRAJ, seed=103)),
        ("traj", provoking_targets(solver, clear, N_TRAJ, seed=104)),
    ]


def run_group(solver, model, start, mode, kind, targets):
    """Solve a group; returns per-solve (seconds, steps, trajectory, target)."""
    coll = [SelfCollisionObj(model, margin=MARGIN, hard=(mode == "hard"))]
    out = []
    pose = start
    for target in targets:
        mand = [DistanceObjTraj(EE, [target], True)]
        t0 = time.perf_counter()
        if kind == "single":
            res, _, steps = solver.solve(
                initial_rotations=start,
                learning_rate=0.05,
                mandatory_objective_functions=mand,
                optional_objective_functions=coll,
                patience=50,
                verbose=False,
            )
        else:
            res, _, steps = solver.solve(
                initial_rotations=pose,
                learning_rate=0.05,
                mandatory_objective_functions=mand,
                optional_objective_functions=[CombinedDerivativeObj(3, weight=0.05)] + coll,
                ik_points=IK_POINTS,
                patience=200,
                verbose=False,
            )
        dt = time.perf_counter() - t0
        res = np.asarray(res)
        out.append((dt, steps, res, target))
        if kind == "traj":
            pose = res[-1]
    return out


def run_tests():
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *TEST_FILES],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=900,
        )
    except subprocess.TimeoutExpired:
        print("tests timed out")
        return False
    tail = "\n".join(r.stdout.strip().splitlines()[-5:])
    print(tail)
    return r.returncode == 0


def main():
    jax.config.update("jax_default_device", jax.devices("cpu")[0])
    solver = InverseKinematicsSolver(
        URDF, controlled_bones=BONES, threshold=1e-6, num_steps=300, compute_sdf=False
    )
    fk = solver.fk_solver
    t0 = time.perf_counter()
    model = CollisionModel.cached(fk, lower=solver.lower_bounds, upper=solver.upper_bounds)
    print(f"model ready in {time.perf_counter() - t0:.1f}s, {len(model.pairs)} pairs")
    names = model.link_names
    checked = {frozenset((names[a], names[b])) for a, b in model.pairs}
    pairs_ok = checked >= EXPECTED_PAIRS
    if not pairs_ok:
        print(f"missing pairs: {EXPECTED_PAIRS - checked}")

    clear = make_exact_clearance(model, fk)
    lo, hi = np.asarray(solver.lower_bounds), np.asarray(solver.upper_bounds)
    start = ((lo + hi) / 2).astype(np.float32)
    assert float(clear(start[None])[0]) >= MARGIN
    jobs = build_jobs(solver, clear)

    # Warm-up pass: compiles every solve shape for both modes
    for mode in ("soft", "hard"):
        for kind, targets in jobs:
            run_group(solver, model, start, mode, kind, targets[:1])

    results = {}  # (mode, group, i) -> list of runs
    for _ in range(REPEATS):
        for mode in ("soft", "hard"):
            for g, (kind, targets) in enumerate(jobs):
                for i, r in enumerate(run_group(solver, model, start, mode, kind, targets)):
                    results.setdefault((mode, g, i), []).append(r)

    times, steps_total, n_ok, n_coll, worst = [], 0, 0, 0, np.inf
    per_mode = {"soft": [], "hard": []}
    for (mode, g, i), runs in sorted(results.items()):
        dt = min(r[0] for r in runs)
        _, steps, traj, target = runs[0]
        # Deterministic solver: every repeat must give the same answer
        for r in runs[1:]:
            if not np.allclose(r[2], traj, atol=1e-5):
                print(f"warning: nondeterministic result {mode} {g} {i}")
        times.append(dt)
        per_mode[mode].append(dt)
        steps_total += steps
        c = float(np.min(np.asarray(clear(jnp.asarray(dense(traj))))))
        worst = min(worst, c)
        limit = MARGIN - HARD_TOL if mode == "hard" else 0.0
        if c < limit:
            n_coll += 1
        if np.linalg.norm(ee(solver, traj[-1]) - target) < SUCCESS_DIST:
            n_ok += 1

    n = len(times)
    print(f"solves: {n}")
    print(f"soft_ms: {np.median(per_mode['soft']) * 1000:.2f}")
    print(f"hard_ms: {np.median(per_mode['hard']) * 1000:.2f}")
    print(f"mean_ms: {np.mean(times) * 1000:.2f}")
    print(f"total_steps: {steps_total}")
    tests_ok = run_tests() and pairs_ok
    print("---")
    print(f"solve_ms: {np.median(times) * 1000:.2f}")
    print(f"success_rate: {100.0 * n_ok / n:.2f}")
    print(f"worst_clearance_mm: {worst * 1000:.2f}")
    print(f"collisions: {n_coll}")
    print(f"tests: {'PASS' if tests_ok else 'FAIL'}")


# Link pairs checked today; the collision model must keep checking all of them
EXPECTED_PAIRS = {
    frozenset(("panda_" + a, "panda_" + b))
    for a, b in [
        ("link0", "link2"),
        ("link0", "link3"),
        ("link0", "link4"),
        ("link0", "link5"),
        ("link0", "link6"),
        ("link0", "link7"),
        ("link0", "hand"),
        ("link0", "leftfinger"),
        ("link0", "rightfinger"),
        ("link1", "link3"),
        ("link1", "link4"),
        ("link1", "link5"),
        ("link1", "link6"),
        ("link1", "link7"),
        ("link1", "hand"),
        ("link1", "leftfinger"),
        ("link1", "rightfinger"),
        ("link2", "link4"),
        ("link2", "link5"),
        ("link2", "link6"),
        ("link2", "link7"),
        ("link2", "hand"),
        ("link2", "leftfinger"),
        ("link2", "rightfinger"),
        ("link3", "link5"),
        ("link3", "link6"),
        ("link3", "link7"),
        ("link3", "hand"),
        ("link3", "leftfinger"),
        ("link3", "rightfinger"),
        ("link4", "link6"),
        ("link4", "link7"),
        ("link4", "hand"),
        ("link4", "leftfinger"),
        ("link4", "rightfinger"),
        ("link5", "link7"),
        ("link5", "hand"),
        ("link5", "leftfinger"),
        ("link5", "rightfinger"),
        ("link6", "hand"),
        ("link6", "leftfinger"),
        ("link6", "rightfinger"),
        ("link7", "leftfinger"),
        ("link7", "rightfinger"),
    ]
}

if __name__ == "__main__":
    main()
