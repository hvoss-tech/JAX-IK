"""Benchmark for the user's exact Blender repro: a straight 3-bone chain
(head at the origin, tails at (0,0,1)/(0,0,3)/(0,0,5)) with Reach Target
(weight 1.0) + Zero Rotation (weight 0.1) + Prefer Current Pose (weight
0.1), all Mandatory -- matching blender_addon/jax_ik_blender/bridge.py's
solve_chain() bundling and its 2-frame (fixed-prefix + free) trajectory
convention exactly (see build_fk_solver()/solve_chain() there) -- while
dragging the target Empty in a straight line from (0,-3,-3) to (0,3,3)
over ~250 frames, warm-started frame to frame (each solve's init pose is
the previous frame's result, matching Live-update/drag usage).

A multi-frame pose-history + CombinedDerivativeObj smoothing variant was
tried here (and in bridge.py) to address jitter -- reverted. Two problems
surfaced: (1) BoneZeroRotationObj/InitPoseObj-style regularizers jnp.mean()
over *every* frame of the trajectory (see objectives.ObjectiveFunction
subclasses that aren't frame-selective), so extending the fixed prefix from
1 frame to N silently divides their effective per-solve gradient by N --
Zero Rotation at "0.1" was actually acting like ~0.02, which is what made
the smoothed version *look* more accurate at first (a weakened regularizer
lets Reach track more precisely) while leaving the chain less constrained
frame to frame; and (2) even after compensating for that dilution (scaling
Zero Rotation's weight by history length), the smoothness term's actual
effect on dist_std/jump_std was flat-to-noisy across every
history-length/weight tried, with no robust improvement region -- isolated
parameter points looked good surrounded by neighbors that were worse,
consistent with overfitting to this one drag path rather than a real
effect. Not worth the added complexity (a pose-history cache, another
Optional objective) for a benefit this unreliable, so both this benchmark
and bridge.py were reverted back to the simple 2-frame design.

The skeleton is built by hand (see `_build_skeleton`), the same way
bridge.py's build_skeleton_dict()/build_fk_solver() do it directly from
Blender bone data with no file round-trip -- this reproduces the *exact*
bone lengths (1, 2, 2) the user described; loading the equivalent chain
from a GLTF file would silently give the tip bone the wrong length (jax_ik
helper.load_skeleton_from_gltf defaults a leaf bone's length to 0.1,
unlike bridge.py's real bone.length-based construction).

Run directly: `python ik_jax_lib_bench_drag.py`. Prints accuracy/jitter
metrics for the whole drag, plus the worst individual frames (to localize
where jitter actually happens along the path).
"""

import os
import resource
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from jax_ik.ik import FKSolver, solve_ik  # noqa: E402
from jax_ik.objectives import (  # noqa: E402
    BoneZeroRotationObj,
    DistanceObjTraj,
    InitPoseObj,
)

# ---------------------------------------------------------------------------
# Frozen configuration -- mirrors the user's exact repro.
# ---------------------------------------------------------------------------
REACH_WEIGHT = 1.0
ZERO_ROT_WEIGHT = 0.1
PREFER_CURRENT_WEIGHT = 0.1

THRESHOLD = 0.0005  # blender_addon's own JaxIKChain.threshold default
NUM_STEPS = 1000  # blender_addon's own default
LEARNING_RATE = 0.2  # blender_addon's own default
PATIENCE = 100  # blender_addon's own default

DRAG_START = np.array([0.0, -3.0, -3.0], dtype=np.float32)
DRAG_END = np.array([0.0, 3.0, 3.0], dtype=np.float32)
DRAG_FRAMES = 250

TIP_BONE = "tip"
CONTROLLED_BONES = ["root", "mid", "tip"]
NUM_ANGLES = len(CONTROLLED_BONES) * 3


def _rot_x_90() -> np.ndarray:
    """Rotation mapping local +Y (a bone's rest-pose length axis, see
    FKSolver.get_bone_head_tail_from_fk's tail_local=[0, bone_length, 0])
    onto world +Z -- so a bone with no pose rotation points straight up
    along Z, matching the user's rest pose.
    """
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = np.array(
        [[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32
    )  # Rx(+90 deg)
    return m


def _build_skeleton() -> dict:
    """A straight 3-bone chain: root head at the origin, tails at
    (0,0,1), (0,0,3), (0,0,5) -- built the same way
    blender_addon/jax_ik_blender/bridge.py's build_skeleton_dict() does
    (a plain {name: {local_transform, children, parent, bone_length}}
    dict, no GLTF file round-trip), so bone_length matches the real bone
    length exactly instead of the GLTF loader's leaf-bone default.
    """
    root_local = _rot_x_90()  # world-oriented; root has no parent frame to be relative to
    identity_local = np.eye(4, dtype=np.float32)

    return {
        "root": {
            "name": "root",
            "local_transform": root_local,
            "children": ["mid"],
            "bone_length": 1.0,
            "parent": None,
        },
        "mid": {
            "name": "mid",
            # Parent-relative: translate along the parent's local +Y by the
            # parent's own length (root's tail, in root's frame) -- a
            # straight, unbent rest pose needs no additional rotation here.
            "local_transform": identity_local.copy(),
            "children": ["tip"],
            "bone_length": 2.0,
            "parent": "root",
        },
        "tip": {
            "name": "tip",
            "local_transform": identity_local.copy(),
            "children": [],
            "bone_length": 2.0,
            "parent": "mid",
        },
    }


def _fix_translation(skeleton: dict) -> dict:
    skeleton["mid"]["local_transform"][:3, 3] = [0.0, skeleton["root"]["bone_length"], 0.0]
    skeleton["tip"]["local_transform"][:3, 3] = [0.0, skeleton["mid"]["bone_length"], 0.0]
    return skeleton


def build_fk_solver() -> FKSolver:
    skeleton = _fix_translation(_build_skeleton())
    fk_solver = FKSolver.__new__(FKSolver)
    fk_solver.model_file = None
    fk_solver.file_type = ""
    fk_solver.limits = {}
    fk_solver.mesh_data = None
    fk_solver.sdf = None
    fk_solver.skeleton = skeleton
    fk_solver._finish_init(CONTROLLED_BONES, None)
    return fk_solver


def _tip_pos(fk_solver, angles):
    fk = fk_solver.compute_fk_from_angles(angles)
    _, tail = fk_solver.get_bone_head_tail_from_fk(fk, TIP_BONE)
    return np.asarray(tail, dtype=np.float32)


def _solve(fk_solver, lower, upper, target, current_pose):
    mand = (
        DistanceObjTraj(bone_name=TIP_BONE, target_points=target, use_head=False, weight=REACH_WEIGHT),
        BoneZeroRotationObj(weight=ZERO_ROT_WEIGHT),
        InitPoseObj(
            init_rot=current_pose, full_trajectory=False, last_position=True, weight=PREFER_CURRENT_WEIGHT
        ),
    )
    init_rot = np.stack([current_pose, current_pose], axis=0)
    mask = np.array([False, True])
    steps, final_traj, best_obj, _ = solve_ik(
        init_rot=init_rot,
        lower_bounds=lower,
        upper_bounds=upper,
        mandatory_obj_fns=mand,
        optional_obj_fns=(),
        fksolver=fk_solver,
        threshold=THRESHOLD,
        num_steps=NUM_STEPS,
        learning_rate=LEARNING_RATE,
        patience=PATIENCE,
        mask=mask,
    )
    new_pose = np.asarray(final_traj)[-1]
    return new_pose, float(best_obj), int(steps)


def run_benchmark(verbose=True):
    fk_solver = build_fk_solver()
    lower = np.full(NUM_ANGLES, -np.pi, dtype=np.float32)
    upper = np.full(NUM_ANGLES, np.pi, dtype=np.float32)

    t = np.linspace(0.0, 1.0, DRAG_FRAMES, dtype=np.float32)[:, None]
    path = DRAG_START[None, :] * (1 - t) + DRAG_END[None, :] * t

    # Warmup (not counted, and not part of the drag itself)
    warm_pose = np.zeros(NUM_ANGLES, dtype=np.float32)
    for _ in range(2):
        _solve(fk_solver, lower, upper, path[0], warm_pose)

    pose = np.zeros(NUM_ANGLES, dtype=np.float32)
    frames = []
    total_elapsed = 0.0
    for i, target in enumerate(path):
        t0 = time.perf_counter()
        new_pose, obj, steps = _solve(fk_solver, lower, upper, target, pose)
        total_elapsed += time.perf_counter() - t0
        dist = float(np.linalg.norm(_tip_pos(fk_solver, new_pose) - target))
        jump = float(np.linalg.norm(new_pose - pose))
        frames.append(
            {
                "frame": i,
                "target": target.copy(),
                "dist": dist,
                "jump": jump,
                "loss": obj,
                "steps": steps,
            }
        )
        pose = new_pose
        if verbose and (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{DRAG_FRAMES}]", file=sys.stderr)

    avg_time_ms = total_elapsed / len(path) * 1000.0
    return frames, avg_time_ms


def summarize(frames):
    dists = np.array([f["dist"] for f in frames])
    jumps = np.array([f["jump"] for f in frames[1:]])  # frame 0 has no previous frame
    return {
        "dist_mean": float(dists.mean()),
        "dist_std": float(dists.std()),
        "dist_median": float(np.median(dists)),
        "dist_p95": float(np.percentile(dists, 95)),
        "dist_max": float(dists.max()),
        "jump_mean": float(jumps.mean()),
        "jump_std": float(jumps.std()),
        "jump_median": float(np.median(jumps)),
        "jump_p95": float(np.percentile(jumps, 95)),
        "jump_max": float(jumps.max()),
    }


def main():
    frames, avg_time_ms = run_benchmark(verbose=True)
    summary = summarize(frames)

    print("\n=== Drag benchmark: (0,-3,-3) -> (0,3,3) over 250 frames, warm-started ===")
    print(
        f"  end-effector distance   mean={summary['dist_mean']:.5f} "
        f"std={summary['dist_std']:.5f} median={summary['dist_median']:.5f} "
        f"p95={summary['dist_p95']:.5f} max={summary['dist_max']:.5f} (m)"
    )
    print(
        f"  frame-to-frame pose jump (rad, L2): mean={summary['jump_mean']:.5f} "
        f"std={summary['jump_std']:.5f} median={summary['jump_median']:.5f} "
        f"p95={summary['jump_p95']:.5f} max={summary['jump_max']:.5f}"
    )

    # Localize the jitter: the frames whose jump is the largest.
    jump_threshold = summary["jump_median"] * 4.0
    worst = sorted(frames[1:], key=lambda f: -f["jump"])[:15]
    print(f"\n  Worst 15 frames by frame-to-frame jump (median={summary['jump_median']:.5f}):")
    print("    frame   target(y,z)         dist      jump      loss")
    for f in worst:
        y, z = f["target"][1], f["target"][2]
        print(
            f"    {f['frame']:5d}   ({y:6.3f},{z:6.3f})   {f['dist']:.5f}   "
            f"{f['jump']:.5f}   {f['loss']:.6f}"
        )
    flagged = [f["frame"] for f in frames[1:] if f["jump"] > jump_threshold]
    print(
        f"\n  {len(flagged)}/{len(frames) - 1} frames exceed 4x the median jump "
        f"({jump_threshold:.5f} rad); frame range: "
        f"{min(flagged) if flagged else '-'}..{max(flagged) if flagged else '-'}"
    )

    peak_memory_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(
        f"\nRESULT time_ms={avg_time_ms:.6f} dist_mean={summary['dist_mean']:.6f} "
        f"dist_std={summary['dist_std']:.6f} dist_max={summary['dist_max']:.6f} "
        f"jump_mean={summary['jump_mean']:.6f} jump_std={summary['jump_std']:.6f} "
        f"jump_max={summary['jump_max']:.6f} jump_p95={summary['jump_p95']:.6f} "
        f"memory_mb={peak_memory_mb:.1f}"
    )


if __name__ == "__main__":
    main()
