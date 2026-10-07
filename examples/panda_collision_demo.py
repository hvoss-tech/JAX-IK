"""Panda arm avoiding self-collisions while following trajectories.

Every 5 seconds the arm gets a new target and plays a smoothed 10-point
trajectory to it. Every other target is deliberately nasty: the hand
position of a random pose in which the arm collides with itself, so the
direct way there runs the arm into its own links.

The arm also has to stay off the floor and out of two spheres, each of
which can be switched on and off. The translucent shells are the collision
proxies (low-res envelopes that fully contain each link); they turn red
where two links come within the margin, and so do the floor and the
spheres. Keys: m cycles the collision mode (off / soft / hard), c shows or
hides the proxies, g switches the floor on and off, 1 and 2 the spheres.
Run from the repository root:

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
# The model is Y-up with the base resting on y = -1
FLOOR = -1.0
SPHERES = [((0.35, -0.55, 0.35), 0.1), ((-0.35, -0.7, 0.3), 0.08)]


class ProvokingPlanner(TrajectoryPlanner):
    """Every other target comes from a self-colliding pose. Targets under
    the floor or inside a sphere that is switched on are skipped."""

    def __init__(self, *args, model, modes, **kwargs):
        super().__init__(*args, **kwargs)
        fk = self.solver.fk_solver
        self._clear = jax.jit(lambda q: model.clearances(q, fk).min())
        self._modes = modes
        self._count = 0

    def _reachable(self, targets):
        o = self._modes.obstacle
        for p in targets:
            if o.floor_on > 0 and p @ o.up < o.floor_height + 0.05:
                return False
            for c, r, on in zip(o.sphere_centers, o.sphere_radii, o.sphere_on):
                if on > 0 and np.linalg.norm(p - c) < r + 0.05:
                    return False
        return True

    def random_targets(self, spread=0.6):
        self._count += 1
        while True:
            if self._count % 2:
                targets = super().random_targets(spread)
            else:
                q = self._rng.uniform(self._lower, self._upper).astype(np.float32)
                if float(self._clear(q)) >= -0.01:
                    continue
                targets = self.end_effector_positions(q)
            if self._reachable(targets):
                return targets


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
    # The base stands on the floor, so it is not checked against it
    obstacles = dict(floor=FLOOR, spheres=SPHERES, ignore_links=("panda_link0",))
    modes = CollisionModes(model, margin=MARGIN, mode="hard", obstacles=obstacles)
    planner = ProvokingPlanner(
        solver, [END_EFFECTOR], initial_angles=mid, model=model, modes=modes, seed=0
    )
    viz = RealtimeVisualizer(solver.fk_solver, title="JAX-IK - Panda collision")
    viz.attach_collision(modes)
    # Plans that end more than 5 mm from their target are replaced by new
    # ones; three candidates are planned in parallel to keep the buffer full
    viz.run_trajectory(planner, interval=5.0, playback=2.0, max_miss=0.005, candidates=3)


if __name__ == "__main__":
    main()
