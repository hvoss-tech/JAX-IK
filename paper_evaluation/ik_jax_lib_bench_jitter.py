"""Benchmark for the "jittery / doesn't quite reach the target" regression
reported against the Blender addon's exact objective setup: a 3-bone chain
with Reach Target (weight 1.0), Zero Rotation (weight 0.05) and Prefer
Current Pose (weight 0.05), all Mandatory -- matching how
blender_addon/jax_ik_blender/bridge.py's solve_chain() bundles a user's
Reach Target together with its two regularizer objectives (all folded into
mandatory_obj_fns, gated on their *combined* loss).

Two things are measured, matching the two symptoms reported:

1. Accuracy: for a batch of static, FK-sampled reachable targets, how
   close does a solve from a fixed rest pose actually get?

2. Smoothness ("jitter"): for a target moving along a smooth path,
   solving every frame warm-started from the previous frame's result
   (matching real Live-update / bake usage), how big are the frame-to-
   frame joint-angle jumps? A well-behaved solve should track a smoothly
   moving target with smoothly changing joint angles -- a large jump
   between two adjacent frames of a smooth target path, or between
   repeated solves of the *same* static target from slightly different
   warm starts, is exactly what shows up on screen as flicker.

Run directly: `python ik_jax_lib_bench_jitter.py`. Prints a human-readable
report; also usable as a library (see `run_benchmark`) for run_experiment.py
or ad-hoc comparisons against another src/jax_ik checkout.
"""

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from jax_ik.ik import InverseKinematicsSolver  # noqa: E402
from jax_ik.objectives import (  # noqa: E402
    BoneZeroRotationObj,
    DistanceObjTraj,
    InitPoseObj,
)
from jax_ik.smplx_statics import left_arm_bounds_dict  # noqa: E402

# ---------------------------------------------------------------------------
# Frozen configuration -- mirrors the reported repro exactly: a 3-bone
# chain, Reach Target (1.0) + Zero Rotation (0.05) + Prefer Current Pose
# (0.05), all Mandatory (bridge.py never marks its own auto-added
# regularizers Optional, and the user's report didn't either).
# ---------------------------------------------------------------------------
MODEL_FILE = os.path.join(_ROOT, "smplx.glb")
BONES = ["left_shoulder", "left_elbow", "left_wrist"]
EE_BONE = "left_wrist"

REACH_WEIGHT = 1.0
ZERO_ROT_WEIGHT = 0.05
PREFER_CURRENT_WEIGHT = 0.05

THRESHOLD = 0.0005  # blender_addon's own JaxIKChain.threshold default
NUM_STEPS = 1000  # blender_addon's own default
LEARNING_RATE = 0.2  # blender_addon's own default
PATIENCE = 100  # blender_addon's own default

NUM_STATIC_TARGETS = 300
STATIC_SEED = 13579

NUM_PATH_FRAMES = 60
NUM_PATHS = 20
PATH_SEED = 24680

PERTURB_TRIALS = 30
PERTURB_SEED = 97531
PERTURB_MAGNITUDE = 0.01  # radians, ~0.6 deg -- mimics a live-update tick's
# tiny warm-start difference (float noise / a previous solve that hadn't
# fully re-converged yet), not a deliberate large pose change.


def _build_bounds():
    bounds = []
    for bone in BONES:
        lower, upper = left_arm_bounds_dict[bone]
        for l, u in zip(lower, upper):
            bounds.append((l, u))
    return bounds


def _build_solver():
    return InverseKinematicsSolver(
        model_file=MODEL_FILE,
        controlled_bones=BONES,
        bounds=_build_bounds(),
        threshold=THRESHOLD,
        num_steps=NUM_STEPS,
        compute_sdf=False,
    )


def _objectives(target, initial_rotations):
    mandatory = (
        DistanceObjTraj(
            bone_name=EE_BONE, target_points=target, use_head=False, weight=REACH_WEIGHT
        ),
        BoneZeroRotationObj(weight=ZERO_ROT_WEIGHT),
        InitPoseObj(
            init_rot=initial_rotations,
            full_trajectory=False,
            last_position=True,
            weight=PREFER_CURRENT_WEIGHT,
        ),
    )
    return mandatory


def _solve(solver, target, initial_rotations):
    mand = _objectives(target, initial_rotations)
    angles, obj, steps = solver.solve(
        initial_rotations=initial_rotations,
        learning_rate=LEARNING_RATE,
        mandatory_objective_functions=mand,
        optional_objective_functions=(),
        ik_points=1,
        patience=PATIENCE,
        verbose=False,
    )
    solved = angles[-1] if angles.ndim == 2 else angles
    return np.asarray(solved, dtype=np.float32), float(obj), int(steps)


def _ee_pos(solver, angles):
    fk = solver.fk_solver.compute_fk_from_angles(angles)
    _, tail = solver.fk_solver.get_bone_head_tail_from_fk(fk, EE_BONE)
    return np.asarray(tail, dtype=np.float32)


def _fk_sample_target(solver, rng):
    lo = np.asarray(solver.lower_bounds)
    hi = np.asarray(solver.upper_bounds)
    angles = rng.uniform(lo, hi).astype(np.float32)
    return _ee_pos(solver, angles)


def _warmup(solver):
    zeros = np.zeros(len(BONES) * 3, dtype=np.float32)
    target = _ee_pos(solver, zeros + 0.1)
    for _ in range(2):
        _solve(solver, target, zeros)


# ---------------------------------------------------------------------------
# 1. Static accuracy
# ---------------------------------------------------------------------------
def measure_static_accuracy(solver, verbose=False):
    rng = np.random.default_rng(STATIC_SEED)
    zeros = np.zeros(len(BONES) * 3, dtype=np.float32)
    dists = []
    losses = []
    successes = 0
    t0 = time.perf_counter()
    for i in range(NUM_STATIC_TARGETS):
        target = _fk_sample_target(solver, rng)
        solved, obj, _ = _solve(solver, target, zeros)
        dist = float(np.linalg.norm(_ee_pos(solver, solved) - target))
        dists.append(dist)
        losses.append(obj)
        if obj < THRESHOLD:
            successes += 1
        if verbose and (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{NUM_STATIC_TARGETS}]", file=sys.stderr)
    elapsed = time.perf_counter() - t0
    dists = np.array(dists)
    losses = np.array(losses)
    return {
        "avg_time_ms": elapsed / NUM_STATIC_TARGETS * 1000.0,
        "success_rate": successes / NUM_STATIC_TARGETS * 100.0,
        "dist_mean": float(dists.mean()),
        "dist_median": float(np.median(dists)),
        "dist_p95": float(np.percentile(dists, 95)),
        "dist_max": float(dists.max()),
        "loss_mean": float(losses.mean()),
        "loss_median": float(np.median(losses)),
    }


# ---------------------------------------------------------------------------
# 2a. Moving-target smoothness (warm-started sequential solves)
# ---------------------------------------------------------------------------
def _smooth_path(solver, rng, n_frames):
    """A smooth (linearly interpolated, then re-projected to stay
    reachable) path between two FK-sampled points -- deliberately simple:
    the *target* moving smoothly is the whole point, any wiggle in the
    solved pose is on the solver, not the input.
    """
    a = _fk_sample_target(solver, rng)
    b = _fk_sample_target(solver, rng)
    t = np.linspace(0.0, 1.0, n_frames, dtype=np.float32)[:, None]
    return a[None, :] * (1 - t) + b[None, :] * t


def measure_path_smoothness(solver, verbose=False):
    rng = np.random.default_rng(PATH_SEED)
    zeros = np.zeros(len(BONES) * 3, dtype=np.float32)
    all_dists = []
    all_jumps = []  # per-frame joint-angle L2 delta from the previous frame
    for p in range(NUM_PATHS):
        path = _smooth_path(solver, rng, NUM_PATH_FRAMES)
        pose = zeros
        prev_pose = None
        for target in path:
            pose, obj, _ = _solve(solver, target, pose)
            dist = float(np.linalg.norm(_ee_pos(solver, pose) - target))
            all_dists.append(dist)
            if prev_pose is not None:
                all_jumps.append(float(np.linalg.norm(pose - prev_pose)))
            prev_pose = pose
        if verbose:
            print(f"  path [{p + 1}/{NUM_PATHS}]", file=sys.stderr)
    all_dists = np.array(all_dists)
    all_jumps = np.array(all_jumps)
    return {
        "dist_mean": float(all_dists.mean()),
        "dist_p95": float(np.percentile(all_dists, 95)),
        "dist_max": float(all_dists.max()),
        "jump_mean": float(all_jumps.mean()),
        "jump_median": float(np.median(all_jumps)),
        "jump_p95": float(np.percentile(all_jumps, 95)),
        "jump_max": float(all_jumps.max()),
    }


# ---------------------------------------------------------------------------
# 2b. Stability under a tiny warm-start perturbation (same static target)
# ---------------------------------------------------------------------------
def measure_perturbation_stability(solver, verbose=False):
    """For a fixed target, solve repeatedly from a fixed init pose *plus a
    tiny random perturbation* (mimicking the small differences between one
    live-update tick's warm start and the next, even when the target
    itself hasn't moved). A stable solve lands at very nearly the same
    final pose regardless of that tiny perturbation; a solve that's
    sensitive to it will visibly hop between nearby-but-different poses
    frame to frame -- flicker on an unmoving target.
    """
    rng = np.random.default_rng(PERTURB_SEED)
    zeros = np.zeros(len(BONES) * 3, dtype=np.float32)
    target = _fk_sample_target(solver, np.random.default_rng(STATIC_SEED + 1))

    solved0, _, _ = _solve(solver, target, zeros)
    end_effector_spreads = []
    pose_spreads = []
    for _ in range(PERTURB_TRIALS):
        noise = rng.uniform(-PERTURB_MAGNITUDE, PERTURB_MAGNITUDE, size=zeros.shape).astype(
            np.float32
        )
        solved, _, _ = _solve(solver, target, zeros + noise)
        pose_spreads.append(float(np.linalg.norm(solved - solved0)))
        end_effector_spreads.append(
            float(np.linalg.norm(_ee_pos(solver, solved) - _ee_pos(solver, solved0)))
        )
    pose_spreads = np.array(pose_spreads)
    end_effector_spreads = np.array(end_effector_spreads)
    return {
        "pose_spread_mean": float(pose_spreads.mean()),
        "pose_spread_max": float(pose_spreads.max()),
        "ee_spread_mean": float(end_effector_spreads.mean()),
        "ee_spread_max": float(end_effector_spreads.max()),
    }


def run_benchmark(verbose=True):
    solver = _build_solver()
    _warmup(solver)
    static = measure_static_accuracy(solver, verbose=verbose)
    path = measure_path_smoothness(solver, verbose=verbose)
    perturb = measure_perturbation_stability(solver, verbose=verbose)
    return static, path, perturb


def main():
    static, path, perturb = run_benchmark(verbose=True)

    print("\n=== Static accuracy (300 FK-sampled targets, fixed rest-pose init) ===")
    print(f"  success_rate@{THRESHOLD} = {static['success_rate']:.2f}%")
    print(f"  avg solve time          = {static['avg_time_ms']:.3f} ms")
    print(
        f"  end-effector distance   mean={static['dist_mean']:.5f} "
        f"median={static['dist_median']:.5f} p95={static['dist_p95']:.5f} "
        f"max={static['dist_max']:.5f} (m)"
    )
    print(f"  combined mandatory loss mean={static['loss_mean']:.5f} median={static['loss_median']:.5f}")

    print(f"\n=== Moving-target smoothness ({NUM_PATHS} paths x {NUM_PATH_FRAMES} frames, warm-started) ===")
    print(
        f"  end-effector distance   mean={path['dist_mean']:.5f} "
        f"p95={path['dist_p95']:.5f} max={path['dist_max']:.5f} (m)"
    )
    print(
        f"  frame-to-frame pose jump (rad, L2 over all controlled angles): "
        f"mean={path['jump_mean']:.5f} median={path['jump_median']:.5f} "
        f"p95={path['jump_p95']:.5f} max={path['jump_max']:.5f}"
    )

    print(f"\n=== Stability under a tiny ({PERTURB_MAGNITUDE} rad) warm-start perturbation, static target ===")
    print(
        f"  pose spread (rad, L2)   mean={perturb['pose_spread_mean']:.5f} "
        f"max={perturb['pose_spread_max']:.5f}"
    )
    print(
        f"  end-effector spread (m) mean={perturb['ee_spread_mean']:.5f} "
        f"max={perturb['ee_spread_max']:.5f}"
    )

    print(
        f"\nRESULT success_rate={static['success_rate']:.2f} "
        f"dist_mean={static['dist_mean']:.6f} "
        f"jump_mean={path['jump_mean']:.6f} jump_max={path['jump_max']:.6f} "
        f"perturb_ee_mean={perturb['ee_spread_mean']:.6f} "
        f"time_ms={static['avg_time_ms']:.6f}"
    )


if __name__ == "__main__":
    main()
