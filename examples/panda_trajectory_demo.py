"""Panda arm following a new random target every 5 seconds.

Each target gets a 10-point trajectory from the current pose, smoothed with
CombinedDerivativeObj (weight 0.05), which is then played back in the
realtime visualizer. Run from the repository root:

    python examples/panda_trajectory_demo.py

The same is available for any model via
``jax-ik-viz <model> --end_effector <bone> --random_targets 5``.
"""

import os

import jax
import numpy as np

from jax_ik.ik import InverseKinematicsSolver
from jax_ik.visualizer import RealtimeVisualizer, TrajectoryPlanner

URDF = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "models", "panda", "urdf", "panda.urdf"
)
BONES = [f"panda_link{i}" for i in range(1, 8)]
END_EFFECTOR = "panda_hand_tcp"

INTERVAL = 5.0  # seconds between targets
PLAYBACK = 2.0  # seconds to play each trajectory
IK_POINTS = 10  # trajectory frames after the current pose
SMOOTHING = 0.05  # CombinedDerivativeObj weight


def main():
    # CPU is faster than GPU for problems this small
    jax.config.update("jax_default_device", jax.devices("cpu")[0])

    solver = InverseKinematicsSolver(
        URDF,
        controlled_bones=BONES,
        threshold=1e-6,
        num_steps=1000,
        compute_sdf=False,
    )
    # Start mid-range, clear of joint limits and the straight-up zero pose
    mid = (np.asarray(solver.lower_bounds) + np.asarray(solver.upper_bounds)) / 2
    planner = TrajectoryPlanner(
        solver,
        [END_EFFECTOR],
        ik_points=IK_POINTS,
        smoothing=SMOOTHING,
        learning_rate=0.05,
        patience=200,
        initial_angles=mid,
    )
    viz = RealtimeVisualizer(solver.fk_solver, title="JAX-IK - Panda trajectories")
    viz.run_trajectory(planner, interval=INTERVAL, playback=PLAYBACK)


if __name__ == "__main__":
    main()
