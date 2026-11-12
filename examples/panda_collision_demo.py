"""Panda arm avoiding self-collisions while following trajectories.

The arm keeps moving between far apart targets, each reached with a
smoothed 10-point trajectory that starts as soon as the previous one is
done. Targets come from a reach map: the hand positions of 16384 random
poses (forward kinematics only, built in a few seconds), keeping only
those whose pose is clear of the arm itself and of the obstacles that are
switched on. Every other target is deliberately nasty: the hand position
of a pose in which the arm collides with itself, next to a reachable one,
so the direct way there runs the arm into its own links.

The arm also has to stay off the floor and out of eight spheres, each
of which can be switched on and off. The translucent shells are the collision
proxies (low-res envelopes that fully contain each link); they turn red
where two links come within the margin, and so do the floor and the
spheres. Keys: m cycles the collision mode (off / soft / hard), c shows or
hides the proxies, g switches the floor on and off, 1 to 8 the spheres.
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
from jax_ik.visualizer import CollisionModes, ReachMap, RealtimeVisualizer, TrajectoryPlanner

URDF = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "models", "panda", "urdf", "panda.urdf"
)
BONES = [f"panda_link{i}" for i in range(1, 8)]
END_EFFECTOR = "panda_hand_tcp"
MARGIN = 0.005  # meters kept between links in hard mode
# The model is Y-up with the base resting on y = -1
FLOOR = -1.0
# Eight spheres in a loose ring around the base, at varying heights
SPHERES = [
    ((0.35, -0.55, 0.35), 0.1),
    ((-0.35, -0.7, 0.3), 0.08),
    ((0.5, -0.4, 0.0), 0.08),
    ((0.05, -0.45, 0.55), 0.09),
    ((-0.5, -0.35, -0.05), 0.07),
    ((-0.35, -0.75, -0.35), 0.1),
    ((0.0, -0.5, -0.55), 0.08),
    ((0.4, -0.8, -0.38), 0.09),
]


class ProvokingPlanner(TrajectoryPlanner):
    """Every other target is the hand position of a self-colliding pose
    from the reach map, moved towards a collision-free neighbour so it can
    still be reached."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._colliding = self.reach_map.self_clear < -0.01
        self._count = 0

    def random_targets(self, spread=0.6, start=None):
        self._count += 1
        if self._count % 2 == 0:
            current = self.end_effector_positions(start)
            targets = self.reach_map.pick(current, self._rng, self.far, pool=self._colliding)
            if targets is not None:
                return targets
        return super().random_targets(spread, start)


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
    print("Building reach map...")
    reach = ReachMap(solver, [END_EFFECTOR], collision=modes, seed=0)
    print(f"Reach map: {int(reach.free().sum())} free poses in {reach.build_ms:.0f} ms")
    planner = ProvokingPlanner(
        solver, [END_EFFECTOR], initial_angles=mid, seed=0, reach_map=reach
    )
    viz = RealtimeVisualizer(solver.fk_solver, title="JAX-IK - Panda collision")
    viz.attach_collision(modes)
    # Dragging orbits around the base. Further back than the default view
    # and 45 degrees above the arm, so all the spheres around it are in
    # the picture
    viz.orbit_around("panda_link0")
    viz.look_down(elevation=45.0, zoom_out=1.6)
    # Up to 10 trajectories are planned ahead, in parallel: the poses they
    # end in are solved first, then the trajectories between them. Targets
    # that can't be reached within 5 mm are replaced by new ones, three
    # tried at a time
    viz.run_trajectory(planner, playback=2.0, max_miss=0.005, buffer=10, candidates=3)


if __name__ == "__main__":
    main()
