"""Panda arm avoiding self-collisions while following trajectories.

Every 5 seconds the arm gets a new target and plays a smoothed 10-point
trajectory to it. Every other target is deliberately nasty: the hand
position of a random pose in which the arm collides with itself, so the
direct way there runs the arm into its own links.

The translucent shells are the collision proxies (low-res envelopes that
fully contain each link); they turn red where two links come within the
margin. Keys: m cycles the collision mode (off / soft / hard), c shows or
hides the proxies. Run from the repository root:

    python examples/panda_collision_demo.py

The same is available for any model via
``jax-ik-viz <model> --end_effector <bone> --random_targets 5 --collision hard``.
"""

import os

import jax
import numpy as np

from jax_ik.collision import CollisionModel
from jax_ik.ik import InverseKinematicsSolver
from jax_ik.visualizer import CollisionModes, RealtimeVisualizer, TrajectoryPlanner

URDF = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "models", "panda", "urdf", "panda.urdf"
)
BONES = [f"panda_link{i}" for i in range(1, 8)]
END_EFFECTOR = "panda_hand_tcp"
MARGIN = 0.005  # meters kept between links in hard mode


class ProvokingPlanner(TrajectoryPlanner):
    """Every other target comes from a self-colliding pose."""

    def __init__(self, *args, model, **kwargs):
        super().__init__(*args, **kwargs)
        fk = self.solver.fk_solver
        self._clear = jax.jit(lambda q: model.clearances(q, fk).min())
        self._count = 0

    def random_targets(self, spread=0.6):
        self._count += 1
        if self._count % 2:
            return super().random_targets(spread)
        while True:
            q = self._rng.uniform(self._lower, self._upper).astype(np.float32)
            if float(self._clear(q)) < -0.01:
                return self.end_effector_positions(q)


def main():
    jax.config.update("jax_default_device", jax.devices("cpu")[0])
    solver = InverseKinematicsSolver(
        URDF, controlled_bones=BONES, threshold=1e-6, num_steps=300, compute_sdf=False
    )
    print("Building collision model (cached after the first run)...")
    model = CollisionModel.cached(
        solver.fk_solver, lower=solver.lower_bounds, upper=solver.upper_bounds
    )
    mid = (np.asarray(solver.lower_bounds) + np.asarray(solver.upper_bounds)) / 2
    planner = ProvokingPlanner(
        solver, [END_EFFECTOR], initial_angles=mid, model=model, seed=0
    )
    viz = RealtimeVisualizer(solver.fk_solver, title="JAX-IK - Panda self-collision")
    viz.attach_collision(CollisionModes(model, margin=MARGIN, mode="hard"))
    viz.run_trajectory(planner, interval=5.0, playback=2.0)


if __name__ == "__main__":
    main()
