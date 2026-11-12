"""Realtime IK visualizer.

Two ways to use it:

* Interactive: drag one target sphere per end-effector and watch the model
  follow, solving IK every frame (``jax-ik-viz`` on the command line, or
  ``RealtimeVisualizer.run_interactive``).
* Streaming: drive the window from your own loop by pushing angle vectors
  with ``RealtimeVisualizer.update``.
* Trajectories: every few seconds pick new reachable targets, solve a short
  smoothed trajectory to them and play it back (``--random_targets``, or
  ``RealtimeVisualizer.run_trajectory``).
"""

import argparse
import copy
import pathlib
import time
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np

from jax_ik.helper import load_mesh_data_from_gltf, load_mesh_data_from_urdf
from jax_ik.ik import FKSolver, InverseKinematicsSolver
from jax_ik.objectives import CombinedDerivativeObj, DistanceObjTraj, InitPoseObj

MOVABLE_JOINTS = ("revolute", "continuous", "prismatic")


def default_controlled_bones(skeleton: dict, end_effector: str) -> list:
    """Movable URDF joints on the chain from the root to end_effector,
    root first. Mimic joints are skipped since their master drives them.
    """
    if end_effector not in skeleton:
        raise ValueError(f"Bone '{end_effector}' not found in skeleton.")
    chain = []
    cur = end_effector
    while cur is not None:
        bone = skeleton[cur]
        if bone.get("joint_type") in MOVABLE_JOINTS and "mimic" not in bone:
            chain.append(cur)
        cur = bone["parent"]
    return chain[::-1]


def load_mesh_data(fk_solver: FKSolver) -> dict:
    if fk_solver.file_type == ".urdf":
        return load_mesh_data_from_urdf(fk_solver.model_file, fk_solver)
    return load_mesh_data_from_gltf(fk_solver.model_file, fk_solver)


class MeshPoser:
    """Turns angle vectors into posed mesh vertices and skeleton segments.

    Same math as helper.deform_mesh, but jitted once with the mesh arrays as
    arguments, so posing a few hundred thousand vertices stays cheap enough
    to do every frame.
    """

    def __init__(self, fk_solver: FKSolver, mesh_data: dict):
        self.fk_solver = fk_solver
        self.mesh_data = mesh_data
        self.faces = np.asarray(mesh_data["faces"])
        self._bind_inv = jnp.linalg.inv(fk_solver.bind_fk)
        self._parents = np.asarray(fk_solver.parent_list)

        if "skin_joints" in mesh_data:
            self._weights = mesh_data["skin_weights"]
            self._joints = mesh_data["skin_joints"]
        else:
            # Rigid assignment is LBS with a single joint at full weight
            self._joints = mesh_data["vertex_assignment"][:, None]
            self._weights = jnp.ones_like(self._joints, dtype=jnp.float32)

        @jax.jit
        def _pose(angles, vertices, joints, weights, bind_inv):
            fk = fk_solver.compute_fk_from_angles(angles)
            bone_transforms = fk @ bind_inv
            blended = jnp.einsum("vk,vkij->vij", weights, bone_transforms[joints])
            posed = jnp.einsum("vij,vj->vi", blended, vertices)
            return posed[:, :3], fk[:, :3, 3]

        self._pose = _pose

    def pose(self, angles) -> tuple:
        """Returns (vertices (V, 3), bone head positions (N, 3))."""
        verts, heads = self._pose(
            jnp.asarray(angles, dtype=jnp.float32),
            self.mesh_data["vertices"],
            self._joints,
            self._weights,
            self._bind_inv,
        )
        return np.asarray(verts), np.asarray(heads)

    def skeleton_lines(self, heads: np.ndarray) -> np.ndarray:
        """Line segments (parent head -> child head) in pyvista's lines format."""
        pairs = [(p, i) for i, p in enumerate(self._parents) if p >= 0]
        if not pairs:
            return np.zeros(0, dtype=np.int64)
        return np.hstack([[2, p, c] for p, c in pairs]).astype(np.int64)


class IKController:
    """Runs a short warm-started IK solve per frame towards the current
    targets, so the pose follows a moving target smoothly.

    Each solve is anchored to the previous frame's pose (smooth_weight).
    Without it, redundant chains (e.g. a 7-DOF arm reaching a 3D point)
    drift along their null space between frames and visibly jitter even
    while the target is held still.
    """

    def __init__(
        self,
        solver: InverseKinematicsSolver,
        end_effectors: list,
        learning_rate: float = 0.02,
        smooth_weight: float = 0.1,
        patience: int = 20,
        initial_angles: np.ndarray = None,
        extra_objectives: list = None,
    ):
        self.solver = solver
        self.end_effectors = list(end_effectors)
        self.learning_rate = learning_rate
        self.smooth_weight = smooth_weight
        self.patience = patience
        # Added to every solve, e.g. a SelfCollisionObj; may be swapped live
        self.extra_objectives = list(extra_objectives or [])
        if initial_angles is None:
            initial_angles = np.zeros(solver.lower_bounds.shape, dtype=np.float32)
        # Zero isn't always a legal pose (e.g. the Panda's 4th joint)
        self.angles = np.clip(
            np.asarray(initial_angles, dtype=np.float32),
            np.asarray(solver.lower_bounds),
            np.asarray(solver.upper_bounds),
        )
        self.loss = float("inf")

    def end_effector_positions(self, angles=None) -> list:
        fk_solver = self.solver.fk_solver
        fk = fk_solver.compute_fk_from_angles(
            self.angles if angles is None else angles
        )
        return [
            np.asarray(fk_solver.get_bone_head_tail_from_fk(fk, name)[0])
            for name in self.end_effectors
        ]

    def step(self, targets: list) -> np.ndarray:
        mandatory = [
            DistanceObjTraj(
                bone_name=name,
                target_points=[np.asarray(target, dtype=np.float32)],
                use_head=True,
            )
            for name, target in zip(self.end_effectors, targets)
        ]
        optional = [
            InitPoseObj(
                init_rot=self.angles, last_position=True, weight=self.smooth_weight
            ),
            *self.extra_objectives,
        ]
        result, loss, _ = self.solver.solve(
            initial_rotations=self.angles,
            learning_rate=self.learning_rate,
            mandatory_objective_functions=mandatory,
            optional_objective_functions=optional,
            ik_points=1,
            patience=self.patience,
            verbose=False,
        )
        self.angles = np.asarray(result[-1], dtype=np.float32)
        self.loss = loss
        return self.angles


class TrajectoryPlanner:
    """Solves a whole trajectory of `ik_points` frames from the current pose
    to the targets in one go, smoothed with CombinedDerivativeObj. The
    targets only constrain the last frame, the frames in between are shaped
    by the smoothing.
    """

    def __init__(
        self,
        solver: InverseKinematicsSolver,
        end_effectors: list,
        ik_points: int = 10,
        smoothing: float = 0.05,
        learning_rate: float = 0.05,
        patience: int = 200,
        initial_angles: np.ndarray = None,
        seed: int = None,
        extra_objectives: list = None,
        reach_map: "ReachMap" = None,
        far: float = 0.3,
    ):
        self.solver = solver
        self.end_effectors = list(end_effectors)
        # With a reach map, targets come from it: the `far` fraction of
        # reachable, collision-free cells furthest from where the plan starts
        self.reach_map = reach_map
        self.far = far
        # Added to every plan, e.g. a SelfCollisionObj; may be swapped live
        self.extra_objectives = list(extra_objectives or [])
        self.ik_points = ik_points
        self.smoothing = smoothing
        self.learning_rate = learning_rate
        self.patience = patience
        # Reused for FK lookups and clipping the start pose into bounds
        self._fk = IKController(solver, end_effectors, initial_angles=initial_angles)
        self.pose = self._fk.angles
        self.trajectory = self.pose[None, :]
        self.loss = float("inf")
        self.steps = 0
        self.solve_ms = 0.0
        self._lower = np.asarray(solver.lower_bounds)
        self._upper = np.asarray(solver.upper_bounds)
        self._rng = np.random.default_rng(seed)

    def end_effector_positions(self, angles=None) -> list:
        return self._fk.end_effector_positions(self.pose if angles is None else angles)

    def random_targets(self, spread: float = 0.6, start=None) -> list:
        """End-effector positions of a random legal pose, so every target is
        reachable. `spread` is the fraction of each joint range (around its
        middle) to sample from.

        With a reach map, a far away point from it instead, measured from
        the end-effectors at `start` (default: the current pose).
        """
        if self.reach_map is not None:
            current = self.end_effector_positions(start)
            targets = self.reach_map.pick(current, self._rng, self.far)
            if targets is not None:
                return targets
        mid = (self._lower + self._upper) / 2
        half = (self._upper - self._lower) / 2
        q = mid + self._rng.uniform(-spread, spread, mid.shape) * half
        q = np.clip(q, self._lower, self._upper).astype(np.float32)
        return self.end_effector_positions(q)

    def plan(self, targets: list) -> np.ndarray:
        """Trajectory (ik_points + 1, D) from the current pose to `targets`;
        frame 0 is the current pose.
        """
        traj, loss, steps, ms = self.solve_trajectory(self.pose, targets)
        self.solve_ms = ms
        self.trajectory = traj
        self.loss = loss
        self.steps = steps
        return self.trajectory

    def solve_trajectory(self, start_pose, targets: list) -> tuple:
        """Like plan, but from `start_pose` and without changing the
        planner's state, so it can run in the background while the current
        trajectory plays. Returns (trajectory, loss, steps, solve ms).

        With a reach map, the solve starts from a guess that blends from
        `start_pose` to a map pose reaching near the targets, which gets far
        targets several times more often and faster than starting from a
        standing pose."""
        start = time.perf_counter()
        start_pose = np.asarray(start_pose, dtype=np.float32)
        mandatory = [
            DistanceObjTraj(
                bone_name=name,
                target_points=[np.asarray(target, dtype=np.float32)],
                use_head=True,
            )
            for name, target in zip(self.end_effectors, targets)
        ]
        optional = [
            CombinedDerivativeObj(max_order=3, weight=self.smoothing),
            *self.extra_objectives,
        ]
        guess = self.initial_guess(start_pose, targets)
        if guess is None:
            traj, loss, steps = self.solver.solve(
                initial_rotations=start_pose,
                learning_rate=self.learning_rate,
                mandatory_objective_functions=mandatory,
                optional_objective_functions=optional,
                ik_points=self.ik_points,
                patience=self.patience,
                verbose=False,
            )
        else:
            traj, loss, steps = self.solver.solve_guess(
                guess,
                learning_rate=self.learning_rate,
                mandatory_objective_functions=mandatory,
                optional_objective_functions=optional,
                patience=self.patience,
            )
        ms = (time.perf_counter() - start) * 1000.0
        return np.asarray(traj, dtype=np.float32), loss, steps, ms

    def initial_guess(self, start_pose, targets: list, candidates: int = 8):
        """(ik_points + 1, D) blend from `start_pose` to the pose of one of
        the `candidates` reach map samples closest to `targets`, or None
        without a reach map. Collision objectives (the extra objectives with
        a violation) must find nothing on the way: a hard constraint can't
        always repair a guess that runs through a link or an obstacle. If
        every blend collides there is no guess either."""
        if self.reach_map is None:
            return None
        poses = self.reach_map.poses_near(targets, candidates)
        if len(poses) == 0:
            return None
        start_pose = np.asarray(start_pose, dtype=np.float32)
        t = np.linspace(0.0, 1.0, self.ik_points + 1, dtype=np.float32)[None, :, None]
        guesses = start_pose + t * (poses[:, None] - start_pose)  # (K, T, D)
        checks = tuple(o for o in self.extra_objectives if hasattr(o, "violation"))
        if not checks:
            return guesses[0]
        clear = np.asarray(_guesses_clear(checks, jnp.asarray(guesses), self.solver.fk_solver))
        return guesses[int(np.argmax(clear))] if clear.any() else None

    def solve_end_pose(self, start_pose, targets: list) -> tuple:
        """A single pose reaching `targets`, for planning trajectories
        between known poses (solve_between). Starts from the reach map pose
        closest to the targets, or `start_pose` without a reach map, and
        keeps clear of collisions like a whole trajectory would. Returns
        (pose, miss in meters, solve ms)."""
        start = time.perf_counter()
        guess = None
        if self.reach_map is not None:
            poses = self.reach_map.poses_near(targets, 1)
            if len(poses):
                guess = poses[0]
        if guess is None:
            guess = np.asarray(start_pose, dtype=np.float32)
        guess = np.clip(guess, self._lower, self._upper).astype(np.float32)
        mandatory = [
            DistanceObjTraj(
                bone_name=name,
                target_points=[np.asarray(target, dtype=np.float32)],
                use_head=True,
            )
            for name, target in zip(self.end_effectors, targets)
        ]
        pose, _, _ = self.solver.solve_guess(
            guess[None, :],
            learning_rate=self.learning_rate,
            mandatory_objective_functions=mandatory,
            optional_objective_functions=list(self.extra_objectives),
            prefix_len=0,
            patience=self.patience,
        )
        pose = np.asarray(pose[0], dtype=np.float32)
        reached = self.end_effector_positions(pose)
        miss = max(float(np.linalg.norm(r - t)) for r, t in zip(reached, targets))
        return pose, miss, (time.perf_counter() - start) * 1000.0

    def path_guess(self, start_pose, end_pose, vias: int = 64, rng=None):
        """(ik_points + 1, D) guess from `start_pose` to `end_pose`: the
        straight blend if the collision objectives find nothing on it,
        otherwise the shortest clear detour through one of `vias` free reach
        map poses (picked with `rng`, default the planner's own). If every
        one collides, the one colliding least, for the solver to repair."""
        start_pose = np.asarray(start_pose, dtype=np.float32)
        end_pose = np.asarray(end_pose, dtype=np.float32)
        t = np.linspace(0.0, 1.0, self.ik_points + 1, dtype=np.float32)[:, None]
        straight = start_pose + t * (end_pose - start_pose)
        checks = tuple(o for o in self.extra_objectives if hasattr(o, "violation"))
        if not checks:
            return straight
        candidates = [straight]
        if self.reach_map is not None:
            free = np.nonzero(self.reach_map.free())[0]
            if len(free):
                rng = self._rng if rng is None else rng
                picked = rng.choice(free, size=min(vias, len(free)), replace=False)
                # Two straight pieces through the via pose, evenly timed
                knots = np.linspace(0.0, 2.0, self.ik_points + 1, dtype=np.float32)[:, None]
                first = np.minimum(knots, 1.0)
                second = np.maximum(knots - 1.0, 0.0)
                for via in self.reach_map.poses[picked]:
                    candidates.append(
                        start_pose + first * (via - start_pose) + second * (end_pose - via)
                    )
        guesses = np.stack(candidates)
        violation = np.asarray(
            _guesses_violation(checks, jnp.asarray(guesses), self.solver.fk_solver)
        )
        clear = violation <= 0
        if clear[0]:
            return straight
        if not clear.any():
            return guesses[int(np.argmin(violation))]
        length = np.abs(np.diff(guesses, axis=1)).sum(axis=(1, 2))
        return guesses[int(np.argmin(np.where(clear, length, np.inf)))]

    def trajectory_clear(self, traj: np.ndarray) -> bool:
        """Whether the collision objectives find nothing on `traj`."""
        checks = tuple(o for o in self.extra_objectives if hasattr(o, "violation"))
        if not checks:
            return True
        traj = jnp.asarray(traj, dtype=jnp.float32)[None]
        return bool(np.asarray(_guesses_clear(checks, traj, self.solver.fk_solver))[0])

    def solve_between(self, guess: np.ndarray) -> tuple:
        """Trajectory with the first and last frame of `guess` (from
        path_guess) kept as they are, smoothed and kept clear of collisions
        in between. Returns (trajectory, loss, steps, solve ms)."""
        start = time.perf_counter()
        optional = [
            CombinedDerivativeObj(max_order=3, weight=self.smoothing),
            *self.extra_objectives,
        ]
        traj, loss, steps = self.solver.solve_guess(
            np.asarray(guess, dtype=np.float32),
            learning_rate=self.learning_rate,
            optional_objective_functions=optional,
            patience=self.patience,
            suffix_len=1,
        )
        ms = (time.perf_counter() - start) * 1000.0
        return np.asarray(traj, dtype=np.float32), loss, steps, ms

    def end_effector_paths(self, traj: np.ndarray = None) -> list:
        """One (T, 3) path per end-effector through the trajectory frames."""
        traj = self.trajectory if traj is None else traj
        per_frame = [self.end_effector_positions(frame) for frame in traj]
        return [np.array([p[i] for p in per_frame]) for i in range(len(self.end_effectors))]

    def sample(self, u: float) -> np.ndarray:
        """Pose at playback progress u in [0, 1], linearly interpolated
        between trajectory frames. Also becomes the start of the next plan.
        """
        traj = self.trajectory
        if len(traj) == 1:
            self.pose = traj[0]
            return self.pose
        x = float(np.clip(u, 0.0, 1.0)) * (len(traj) - 1)
        i = min(int(x), len(traj) - 2)
        w = x - i
        self.pose = ((1 - w) * traj[i] + w * traj[i + 1]).astype(np.float32)
        return self.pose


@partial(jax.jit, static_argnums=2)
def _guesses_clear(objectives, guesses, fk_solver):
    """(K,) whether each guess trajectory is free of violations."""

    def clear(X):
        return jnp.all(jnp.stack([jnp.max(o.violation(X, fk_solver)) for o in objectives]) <= 0)

    return jax.vmap(clear)(guesses)


@partial(jax.jit, static_argnums=2)
def _guesses_violation(objectives, guesses, fk_solver):
    """(K,) total violation along each guess trajectory."""

    def total(X):
        return sum(jnp.sum(o.violation(X, fk_solver)) for o in objectives)

    return jax.vmap(total)(guesses)


class ReachMap:
    """End-effector positions the model can reach without colliding.

    Built from forward kinematics of `samples` random legal poses, so every
    point in it is reachable by construction and no IK has to be solved.
    Each sample keeps its clearance to the model's own links and to every
    obstacle of `collision` (a CollisionModes), so poses that collide are
    filtered out, also as obstacles are switched on and off. Obstacles must
    be cleared by an extra `clearance`, since the solver already starts
    pushing away from them before the margin.

    Targets are picked uniformly over the occupied cells of a grid
    (`cell` meters) rather than over the samples, which bunch up where
    many poses reach the same spot, and are interpolated towards a
    collision-free neighbour so they don't repeat the same few points.

    Obstacle clearances are computed once, for the obstacles' positions at
    build time.
    """

    def __init__(
        self,
        solver: InverseKinematicsSolver,
        end_effectors: list,
        collision: "CollisionModes" = None,
        samples: int = 16384,
        cell: float = None,
        clearance: float = 0.02,
        seed: int = None,
        batch: int = 4096,
    ):
        start = time.perf_counter()
        fk_solver = solver.fk_solver
        lower = np.asarray(solver.lower_bounds, np.float32)
        upper = np.asarray(solver.upper_bounds, np.float32)
        rng = np.random.default_rng(seed)
        self.poses = rng.uniform(lower, upper, (samples, len(lower))).astype(np.float32)
        self.collision = collision
        self.clearance = float(clearance)
        ee = [fk_solver.bone_names.index(name) for name in end_effectors]

        model = None if collision is None else collision.model
        obstacle = None if collision is None else collision.obstacle
        if obstacle is not None:
            # Every obstacle on: the switches are applied when picking
            obstacle = copy.copy(obstacle)
            obstacle.floor_on = np.ones((), np.float32)
            obstacle.sphere_on = np.ones_like(collision.obstacle.sphere_on)
        # Pairs further than this keep their cheap broad-phase bound
        near = collision.margin if collision is not None else 0.0

        def one(q):
            fk = fk_solver.compute_fk_from_angles(q)
            heads = jnp.stack([fk[i, :3, 3] for i in ee])
            self_clear = jnp.float32(1e3)
            if model is not None and model.pairs:
                # A capped budget only underestimates clearances, which at
                # worst drops a few reachable poses
                self_clear = jnp.min(model.evaluate(fk, fk_solver, near, max_checks=64)[0])
            if obstacle is None:
                return heads, self_clear, jnp.zeros(0, jnp.float32)
            return heads, self_clear, jnp.min(obstacle.clearances(q, fk_solver), axis=1)

        run = jax.jit(jax.vmap(one))
        heads, self_clear, obstacle_clear = [], [], []
        for i in range(0, samples, batch):
            chunk = self.poses[i : i + batch]
            pad = batch - len(chunk)
            # Same batch shape every time, so it compiles once
            out = run(np.concatenate([chunk, np.repeat(chunk[-1:], pad, 0)]) if pad else chunk)
            out = [np.asarray(o)[: len(chunk)] for o in out]
            heads.append(out[0])
            self_clear.append(out[1])
            obstacle_clear.append(out[2])
        self.positions = np.concatenate(heads)  # (N, E, 3)
        self.self_clear = np.concatenate(self_clear)  # (N,)
        self.obstacle_clear = np.concatenate(obstacle_clear)  # (N, obstacles)

        flat = self.positions.reshape(len(self.positions), -1)
        if cell is None:
            # About 30 cells across the largest extent
            cell = float(np.max(np.ptp(self.positions, axis=0))) / 30
        self.cell = float(cell)
        _, self._cells = np.unique(
            np.floor(flat / self.cell).astype(np.int64), axis=0, return_inverse=True
        )
        self._cells = self._cells.reshape(-1)
        self.build_ms = (time.perf_counter() - start) * 1000.0

    def free(self) -> np.ndarray:
        """(N,) mask of the samples clear of the model's own links by the
        collision margin and of the obstacles that are switched on by
        margin + clearance."""
        if self.collision is None:
            return np.ones(len(self.poses), bool)
        ok = self.self_clear >= self.collision.margin
        need = self.collision.margin + self.clearance
        o = self.collision.obstacle
        if o is not None:
            on = np.concatenate([[o.floor_on], o.sphere_on]) > 0
            ok &= np.all((self.obstacle_clear >= need) | ~on, axis=1)
        return ok

    def poses_near(self, targets: list, k: int = 8) -> np.ndarray:
        """(<= k, D) poses of the free samples closest to `targets`."""
        free = np.nonzero(self.free())[0]
        flat = self.positions[free].reshape(len(free), -1)
        targets = np.concatenate([np.asarray(t, np.float32).reshape(3) for t in targets])
        dist = np.linalg.norm(flat - targets, axis=1)
        k = min(k, len(free))
        nearest = np.argpartition(dist, k - 1)[:k] if k else np.zeros(0, int)
        return self.poses[free[nearest[np.argsort(dist[nearest])]]]

    def pick(self, current: list, rng, far: float = 0.3, pool: np.ndarray = None) -> list:
        """Targets (one point per end-effector) among the `far` fraction of
        reachable cells furthest from the `current` end-effector positions.

        `pool` (a mask over the samples) restricts which samples the target
        is picked from; it is then interpolated towards a free sample in a
        neighbouring cell, so it is reachable even if the pool sample's own
        pose collides. Returns None if nothing fits.
        """
        free = self.free()
        pool = free if pool is None else pool
        flat = self.positions.reshape(len(self.positions), -1)
        current = np.concatenate([np.asarray(c, np.float32).reshape(3) for c in current])
        dist = np.linalg.norm(flat - current, axis=1)
        if not pool.any() or not free.any():
            return None
        cutoff = np.quantile(dist[free], 1.0 - far) if far < 1.0 else 0.0
        candidates = np.nonzero(pool & (dist >= cutoff))[0]
        if len(candidates) == 0:
            # Nothing that far away: take the furthest the pool has
            candidates = np.nonzero(pool)[0]
            candidates = candidates[dist[candidates] >= dist[candidates].max() - self.cell]
        # Uniform over cells: weigh each sample by 1 / samples in its cell
        counts = np.bincount(self._cells[candidates])
        weights = 1.0 / counts[self._cells[candidates]]
        free_idx = np.nonzero(free)[0]
        for i in rng.choice(candidates, size=min(8, len(candidates)), p=weights / weights.sum()):
            near = np.linalg.norm(flat[free_idx] - flat[i], axis=1)
            partners = free_idx[(near <= 1.5 * self.cell) & (free_idx != i)]
            if len(partners) == 0:
                continue
            j = rng.choice(partners)
            w = rng.uniform()
            target = (1 - w) * self.positions[i] + w * self.positions[j]
            return [p.astype(np.float32) for p in target]
        return None


class CollisionModes:
    """The objectives for each collision mode, cycled with the 'm' key.

    With `obstacles` (keyword arguments of ObstacleCollisionObj, e.g.
    floor=..., spheres=[(center, radius), ...]) the links are also kept off
    a floor plane and out of spheres, each of which can be switched on and
    off live (toggle_obstacle).
    """

    NAMES = ("off", "soft", "hard")

    def __init__(self, model, margin=0.005, mode="hard", weight=None, obstacles=None):
        from jax_ik.collision import ObstacleCollisionObj, SelfCollisionObj

        self.model = model
        self.margin = float(margin)
        self.objectives = {
            "off": [],
            "soft": [SelfCollisionObj(model, margin=margin, weight=weight, hard=False)],
            "hard": [SelfCollisionObj(model, margin=margin, weight=weight, hard=True)],
        }
        self.obstacles = None
        if obstacles is not None:
            self.obstacles = {
                name: ObstacleCollisionObj(model, margin=margin, hard=name == "hard", **obstacles)
                for name in ("soft", "hard")
            }
            for name, obj in self.obstacles.items():
                self.objectives[name].append(obj)
        self.mode = mode

    @property
    def obstacle(self):
        """An ObstacleCollisionObj holding the current obstacle state."""
        return None if self.obstacles is None else self.obstacles["hard"]

    def obstacle_names(self) -> list:
        o = self.obstacle
        if o is None:
            return []
        return ["floor"] + [f"sphere {i + 1}" for i in range(len(o.sphere_radii))]

    def obstacle_on(self, index: int) -> bool:
        o = self.obstacle
        return bool(o.floor_on > 0) if index == 0 else bool(o.sphere_on[index - 1] > 0)

    def toggle_obstacle(self, index: int) -> bool:
        """Switch obstacle `index` (0 = floor, 1.. = spheres) on or off in
        every mode; returns whether it is on now."""
        on = not self.obstacle_on(index)
        for o in self.obstacles.values():
            # New arrays rather than in-place edits: a solve running in the
            # background keeps the state it started with
            if index == 0:
                o.floor_on = np.asarray(on, np.float32)
            else:
                flags = o.sphere_on.copy()
                flags[index - 1] = on
                o.sphere_on = flags
        return on

    def cycle(self):
        self.mode = self.NAMES[(self.NAMES.index(self.mode) + 1) % len(self.NAMES)]
        return self.mode

    @property
    def current(self):
        return self.objectives[self.mode]


class PlanPipeline:
    """Plans trajectories ahead in the background, one after the other.

    Every trajectory starts where the previous one ends, so the poses they
    end in are solved first: one at a time, but quickly (a single frame
    each), with `candidates` different targets tried in parallel and the
    first one within `max_miss` of its targets taken. The trajectories in
    between only depend on the two poses at their ends, so all of them, up
    to `buffer`, are solved in parallel. A trajectory the collision
    objectives still find something on is solved again from another guess,
    up to `attempts` times; then it is dropped together with the ones after
    it (they start where it ends), and planning goes on from the end of the
    one before. After `max_retries` failures in a row the next pose or
    trajectory is taken anyway.
    """

    def __init__(
        self,
        planner: TrajectoryPlanner,
        buffer: int = 2,
        candidates: int = 1,
        max_miss: float = None,
        max_retries: int = 20,
        workers: int = None,
        attempts: int = 3,
    ):
        self.planner = planner
        self.attempts = attempts
        self.buffer = buffer
        self.candidates = candidates
        self.max_miss = max_miss
        self.max_retries = max_retries
        self._pool = ThreadPoolExecutor(max_workers=workers or buffer + candidates)
        # Worker threads must not share the planner's generator
        self._rng = np.random.default_rng(planner._rng.integers(2**32))
        self._ahead, self._ending = [], []
        # Until the first plan is done, one solve at a time: parallel
        # first calls would all compile the same functions at once
        self._warm = False
        self.restart(planner.trajectory[-1])

    def restart(self, start_pose):
        """Drop everything planned and plan on from `start_pose`, e.g.
        after the collision setup changed."""
        self._drop(0)
        self._start = np.asarray(start_pose, dtype=np.float32).copy()
        self._retries = 0

    def _drop(self, index: int):
        """Drop the planned trajectories from `index` on and the end pose
        solves following them."""
        for slot in self._ahead[index:]:
            slot["future"].cancel()
        for future in self._ending:
            future.cancel()
        # Planned trajectories in playing order, each a dict with its
        # targets, end pose and the future solving the way there
        self._ahead = self._ahead[:index]
        # End pose solves for the next trajectory after _ahead
        self._ending = []

    @property
    def solving(self) -> int:
        return sum(not slot["future"].done() for slot in self._ahead)

    @property
    def planned(self) -> int:
        """Trajectories ready to play, in order."""
        n = 0
        for slot in self._ahead:
            if not slot["usable"]:
                break
            n += 1
        return n

    def ready(self) -> bool:
        return bool(self._ahead) and self._ahead[0]["usable"]

    def pop(self) -> tuple:
        """The next trajectory, as (targets, (trajectory, loss, steps,
        solve ms)); the time includes solving its end pose."""
        slot = self._ahead.pop(0)
        (traj, loss, steps, ms), _ = slot["future"].result()
        # Where planning goes on from if everything after it is dropped
        self._start = slot["end"]
        return slot["targets"], (traj, loss, steps, ms + slot["end_ms"])

    def _end_pose(self, start, targets):
        pose, miss, ms = self.planner.solve_end_pose(start, targets)
        return targets, pose, miss, ms

    def _between(self, start, end, seed):
        planner = self.planner
        guess = planner.path_guess(start, end, rng=np.random.default_rng(seed))
        result = planner.solve_between(guess)
        return result, planner.trajectory_clear(result[0])

    def poll(self):
        """Take in finished solves and keep the pipeline full."""
        planner = self.planner
        # Trajectories that still collide, dropped with everything after
        for i, slot in enumerate(self._ahead):
            if not slot["future"].done() or slot["usable"]:
                continue
            if slot["future"].result()[1] or self._retries >= self.max_retries:
                slot["usable"] = self._warm = True
                continue
            if slot["attempts"] < self.attempts:
                # Same end poses, so the plans after it stay valid
                slot["attempts"] += 1
                seed = self._rng.integers(2**32)
                slot["future"] = self._pool.submit(self._between, slot["start"], slot["end"], seed)
                continue
            self._retries += 1
            self._drop(i)
            break
        start = self._ahead[-1]["end"] if self._ahead else self._start
        for future in self._ending:
            if not future.done():
                continue
            targets, pose, miss, ms = future.result()
            if (
                self.max_miss is not None
                and miss > self.max_miss
                and self._retries < self.max_retries
            ):
                self._retries += 1
                continue
            self._retries = 0
            seed = self._rng.integers(2**32)
            self._ahead.append(
                dict(
                    targets=targets,
                    start=start,
                    end=pose,
                    end_ms=ms,
                    future=self._pool.submit(self._between, start, pose, seed),
                    # Solved, and clear or out of retries
                    usable=False,
                    attempts=1,
                )
            )
            # The other candidates started from where this one starts
            for other in self._ending:
                other.cancel()
            self._ending = []
            start = pose
            break
        self._ending = [f for f in self._ending if not f.done()]
        if len(self._ahead) >= (self.buffer if self._warm else 1):
            return
        for _ in range((self.candidates if self._warm else 1) - len(self._ending)):
            targets = planner.random_targets(start=start)
            self._ending.append(self._pool.submit(self._end_pose, start, targets))

    def shutdown(self):
        self._pool.shutdown(wait=False, cancel_futures=True)


class RealtimeVisualizer:
    """PyVista window showing a model posed by IK angle vectors."""

    def __init__(
        self,
        fk_solver: FKSolver,
        mesh_data: dict = None,
        show_mesh: bool = True,
        show_skeleton: bool = True,
        title: str = "JAX-IK",
    ):
        import pyvista as pv

        self.pv = pv
        self.fk_solver = fk_solver
        self.poser = MeshPoser(
            fk_solver, mesh_data if mesh_data is not None else load_mesh_data(fk_solver)
        )
        self.plotter = pv.Plotter(title=title)
        self.plotter.set_background("white")
        self.title = title

        angles = np.zeros(len(fk_solver.controlled_indices) * 3, dtype=np.float32)
        verts, heads = self.poser.pose(angles)
        self.mesh = None
        if show_mesh and len(self.poser.faces):
            faces = np.hstack(
                [np.full((len(self.poser.faces), 1), 3), self.poser.faces]
            ).astype(np.int64)
            self.mesh = pv.PolyData(verts, faces)
            self.plotter.add_mesh(
                self.mesh, color="lightsteelblue", smooth_shading=True
            )
        self.skeleton = None
        if show_skeleton:
            self.skeleton = pv.PolyData(heads, lines=self.poser.skeleton_lines(heads))
            self.plotter.add_mesh(
                self.skeleton,
                color="black",
                line_width=3,
                point_size=8,
                render_points_as_spheres=True,
            )

        self.scale = float(np.linalg.norm(np.ptp(verts if len(verts) else heads, axis=0)))
        self.status = self.plotter.add_text("", position="upper_left", font_size=10)
        self.plotter.view_xy()
        self.plotter.reset_camera()
        self._opened = False
        self.collision = None
        self._angles = angles

    # --- collision proxies -------------------------------------------------

    def attach_collision(self, modes: CollisionModes, show: bool = True):
        """Show the collision proxies (low-res envelopes enclosing each
        link), red where a link is inside the margin of another. 'c' toggles
        them, 'm' cycles the collision mode off/soft/hard."""
        import jax

        pv = self.pv
        model = modes.model
        self.collision = modes
        self._clearances = jax.jit(lambda q: model.clearances(q, self.fk_solver))
        parts, link_of_point, faces, offset = [], [], [], 0
        for i, link in enumerate(model.links):
            if link.envelope is None:
                continue
            v, f = link.envelope
            parts.append((i, v))
            link_of_point.append(np.full(len(v), i))
            faces.append(np.hstack([np.full((len(f), 1), 3), f + offset]))
            offset += len(v)
        self._envelope_parts = parts
        self._envelope_link = np.concatenate(link_of_point) if parts else np.zeros(0, int)
        self.envelope = None
        if parts:
            points = np.concatenate([v for _, v in parts])
            self.envelope = pv.PolyData(points, np.concatenate(faces).astype(np.int64))
            self.envelope["hit"] = np.zeros(len(points))
            self._envelope_actor = self.plotter.add_mesh(
                self.envelope,
                scalars="hit",
                cmap=["mediumseagreen", "crimson"],
                clim=[0, 1],
                opacity=0.25,
                show_scalar_bar=False,
            )
            self._envelope_actor.SetVisibility(show)
            self.plotter.add_key_event("c", self._toggle_envelope)
        if modes.obstacles is not None:
            self._attach_obstacles()
        self._update_collision(self._angles)

    # Floor colors: collision on, off, something inside the margin
    _FLOOR_COLORS = {"on": (0.86, 0.83, 0.77), "off": (0.93, 0.93, 0.93), "hit": (0.95, 0.72, 0.70)}
    # Direction the light (and so the arm's shadow) falls in
    _LIGHT = np.array([0.25, -1.0, 0.15])

    def _attach_obstacles(self):
        """Floor plane with the arm's shadow, and the spheres; number keys
        and 'g' switch them on and off."""
        pv = self.pv
        modes = self.collision
        o = modes.obstacle
        up = np.asarray(o.up, np.float64)
        self._clear_obstacles = jax.jit(lambda q, ob: jnp.min(ob.clearances(q, self.fk_solver), 1))
        base = up * float(o.floor_height)
        size = 2.5 * self.scale
        self.floor = pv.Plane(center=base, direction=up, i_size=size, j_size=size)
        self._floor_actor = self.plotter.add_mesh(
            self.floor, color=self._FLOOR_COLORS["on"], ambient=0.3, diffuse=0.7
        )
        # Shadow: the arm squashed onto the floor along the light direction,
        # a hair above it so it is drawn on top
        light = self._LIGHT / np.linalg.norm(self._LIGHT)
        self._shadow_plane = (up, float(o.floor_height) + 1e-3, light)
        self.shadow = None
        if self.mesh is not None:
            self.shadow = pv.PolyData(self._project_shadow(np.asarray(self.mesh.points)), self.mesh.faces)
            self._shadow_actor = self.plotter.add_mesh(self.shadow, color="gray", lighting=False)
        self._sphere_actors = []
        for c, r in zip(np.asarray(o.sphere_centers), np.asarray(o.sphere_radii)):
            sphere = pv.Sphere(radius=float(r), center=c, theta_resolution=48, phi_resolution=48)
            actor = self.plotter.add_mesh(sphere, color="steelblue", smooth_shading=True, specular=0.3)
            shadow = pv.PolyData(self._project_shadow(np.asarray(sphere.points)), sphere.faces)
            shadow_actor = self.plotter.add_mesh(shadow, color="gray", lighting=False)
            self._sphere_actors.append((actor, shadow_actor))

        def toggle(index):
            modes.toggle_obstacle(index)
            changed = getattr(self, "_collision_changed", None)
            if changed is not None:
                changed()
            self._update_collision(self._angles)
            self.plotter.render()

        self.plotter.add_key_event("g", lambda: toggle(0))
        for i in range(len(self._sphere_actors)):
            self.plotter.add_key_event(str(i + 1), lambda i=i: toggle(i + 1))

    def _project_shadow(self, points):
        up, height, light = self._shadow_plane
        t = (points @ up - height) / (light @ up)
        return points - t[:, None] * light

    def _toggle_envelope(self):
        actor = self._envelope_actor
        actor.SetVisibility(not actor.GetVisibility())
        self.plotter.render()

    def _update_collision(self, angles):
        modes = self.collision
        clear = np.asarray(self._clearances(np.asarray(angles, np.float32)))
        self.min_clearance = float(clear.min()) if len(clear) else float("inf")
        if modes.obstacles is not None and hasattr(self, "_floor_actor"):
            self._update_obstacles(angles)
        if self.envelope is None:
            return
        hit = np.zeros(len(modes.model.links))
        for (i, j), c in zip(modes.model.pairs, clear):
            if c < modes.margin:
                hit[i] = hit[j] = 1.0
        fk = np.asarray(self.fk_solver.compute_fk_from_angles(np.asarray(angles, np.float32)))
        names = self.fk_solver.bone_names
        moved = []
        for i, v in self._envelope_parts:
            T = fk[names.index(modes.model.links[i].name)]
            moved.append(v @ T[:3, :3].T + T[:3, 3])
        self.envelope.points = np.concatenate(moved)
        self.envelope["hit"] = hit[self._envelope_link]

    def _update_obstacles(self, angles):
        modes = self.collision
        clear = np.asarray(self._clear_obstacles(np.asarray(angles, np.float32), modes.obstacle))
        self.obstacle_clearance = clear
        hit = clear < modes.margin
        floor = "hit" if hit[0] else ("on" if modes.obstacle_on(0) else "off")
        color = np.asarray(self._FLOOR_COLORS[floor])
        self._floor_actor.prop.color = color
        shadow = color * 0.86
        if self.shadow is not None:
            self.shadow.points = self._project_shadow(np.asarray(self.mesh.points))
            self._shadow_actor.prop.color = shadow
        for i, (actor, shadow_actor) in enumerate(self._sphere_actors):
            on = modes.obstacle_on(i + 1)
            actor.prop.color = "crimson" if hit[i + 1] else "steelblue"
            actor.prop.opacity = 0.9 if on else 0.15
            shadow_actor.SetVisibility(on)
            shadow_actor.prop.color = shadow

    def obstacle_status(self) -> str:
        modes = self.collision
        if modes is None or modes.obstacles is None or not hasattr(self, "obstacle_clearance"):
            return ""
        parts = []
        for i, name in enumerate(modes.obstacle_names()):
            key = "g" if i == 0 else str(i)
            if modes.obstacle_on(i):
                parts.append(f"{name} ({key}): {self.obstacle_clearance[i] * 1000:.0f} mm")
            else:
                parts.append(f"{name} ({key}): off")
        return "obstacles   " + "   ".join(parts) + "\n"

    def collision_status(self) -> str:
        if self.collision is None:
            return ""
        state = "COLLIDING" if self.min_clearance < 0 else (
            "inside margin" if self.min_clearance < self.collision.margin else "clear"
        )
        return (
            f"collision: {self.collision.mode} (margin {self.collision.margin * 1000:.0f} mm)"
            f"   min clearance {self.min_clearance * 1000:.1f} mm, {state}\n"
            + self.obstacle_status()
            + "keys: m = cycle collision mode, c = show/hide proxies"
            + (", g / 1 / 2 ... = floor / spheres on/off" if self.collision.obstacles else "")
            + "\n"
        )

    def _bind_mode_key(self, apply):
        if self.collision is None:
            return

        def cycle():
            apply(self.collision.cycle())

        self.plotter.add_key_event("m", cycle)

    # --- posing -----------------------------------------------------------

    def set_pose(self, angles) -> None:
        self._angles = np.asarray(angles, dtype=np.float32)
        verts, heads = self.poser.pose(angles)
        if self.mesh is not None:
            self.mesh.points = verts
        if self.skeleton is not None:
            self.skeleton.points = heads
        if self.collision is not None:
            self._update_collision(angles)

    def look_down(self, elevation: float = 45.0, zoom_out: float = 1.0) -> None:
        """Turn the camera around its focal point so it looks down at the
        model from `elevation` degrees above the horizon, and move it
        `zoom_out` times further away."""
        cam = self.plotter.camera
        focal = np.asarray(cam.focal_point, np.float64)
        offset = np.asarray(cam.position, np.float64) - focal
        up = np.asarray(cam.up, np.float64)
        up /= np.linalg.norm(up)
        horizontal = offset - (offset @ up) * up
        if np.linalg.norm(horizontal) < 1e-9:
            # Looking straight down: any horizontal direction will do
            horizontal = np.cross(up, [1.0, 0.0, 0.0])
            if np.linalg.norm(horizontal) < 1e-9:
                horizontal = np.cross(up, [0.0, 0.0, 1.0])
        horizontal /= np.linalg.norm(horizontal)
        e = np.radians(elevation)
        distance = np.linalg.norm(offset) * zoom_out
        position = focal + distance * (np.cos(e) * horizontal + np.sin(e) * up)
        self.plotter.camera_position = [tuple(position), tuple(focal), tuple(up)]

    def orbit_around(self, bone: str) -> None:
        """Dragging with the left mouse button turns the camera around the
        vertical axis through `bone` (e.g. the base) and up and down, and
        the view stays upright. The focal point moves onto that axis at the
        height it had, and the camera moves along with it."""
        cam = self.plotter.camera
        up = np.asarray(cam.up, np.float64)
        up /= np.linalg.norm(up)
        _, heads = self.poser.pose(self._angles)
        pivot = heads[self.fk_solver.bone_names.index(bone)].astype(np.float64)
        focal = np.asarray(cam.focal_point, np.float64)
        moved = pivot + ((focal - pivot) @ up) * up
        position = np.asarray(cam.position, np.float64) + (moved - focal)
        self.plotter.camera_position = [tuple(position), tuple(moved), tuple(up)]
        if self.plotter.iren is not None:
            # Azimuth turns around the view up, which terrain style keeps
            self.plotter.enable_terrain_style()

    def set_status(self, text: str) -> None:
        self.status.set_text("upper_left", text)

    def update(self, angles) -> None:
        """Streaming mode: show `angles` and process window events. Call this
        from your own loop; the window opens on the first call.
        """
        if not self._opened:
            self.plotter.show(interactive_update=True, auto_close=False)
            self._opened = True
        self.set_pose(angles)
        self.plotter.update()

    # --- run modes ----------------------------------------------------------

    def run_interactive(self, controller: IKController, fps: float = 60.0) -> None:
        """Blocking: one draggable target sphere per end-effector; IK is
        solved towards the current sphere positions in a background thread,
        so the window stays responsive while a solve runs.
        """
        targets = controller.end_effector_positions()
        radius = 0.015 * self.scale
        colors = ["red", "green", "orange", "purple", "cyan", "magenta"]
        for i, name in enumerate(controller.end_effectors):

            def moved(center, i=i):
                targets[i] = np.asarray(center, dtype=np.float32)

            self.plotter.add_sphere_widget(
                moved,
                center=targets[i],
                radius=radius,
                color=colors[i % len(colors)],
                interaction_event="always",
            )

        if self.collision is not None:
            controller.extra_objectives = self.collision.current

            def apply(mode):
                controller.extra_objectives = self.collision.current
                last["targets"] = None  # re-solve with the new mode

            self._bind_mode_key(apply)

        worker = ThreadPoolExecutor(max_workers=1)
        last = {"solve_ms": 0.0, "future": None, "targets": None, "loss": float("inf")}

        def solve(current):
            start = time.perf_counter()
            angles = controller.step(current)
            return angles, (time.perf_counter() - start) * 1000.0

        def tick(_step=None):
            future = last["future"]
            if future is not None:
                if not future.done():
                    return
                angles, solve_ms = future.result()
                last["future"] = None
                last["solve_ms"] = 0.8 * last["solve_ms"] + 0.2 * solve_ms
                self.set_pose(angles)
                self.set_status(
                    f"solve {last['solve_ms']:.1f} ms   loss {controller.loss:.2e}\n"
                    + self.collision_status()
                    + "drag the spheres to move the targets"
                )
                self.plotter.render()

            # Nothing to do while the targets sit still and the pose has
            # settled (reached, or stopped improving on an unreachable target)
            current = [np.array(t) for t in targets]
            unchanged = last["targets"] is not None and all(
                np.array_equal(a, b) for a, b in zip(current, last["targets"])
            )
            settled = controller.loss < 1e-6 or controller.loss >= 0.999 * last["loss"]
            if unchanged and settled:
                return
            last["targets"] = current
            last["loss"] = controller.loss
            last["future"] = worker.submit(solve, current)

        # Compile before the window opens so the first drag isn't a stall
        print("Compiling IK solver...")
        tick()
        last["future"].result()
        tick()
        self.plotter.add_timer_event(
            max_steps=2**31 - 1, duration=max(1, int(1000 / fps)), callback=tick
        )
        self.plotter.show()
        worker.shutdown(wait=False, cancel_futures=True)

    def run_trajectory(
        self,
        planner: TrajectoryPlanner,
        interval: float = None,
        playback: float = 2.0,
        fps: float = 60.0,
        max_miss: float = None,
        max_retries: int = 20,
        buffer: int = 2,
        candidates: int = 1,
        workers: int = None,
    ) -> None:
        """Blocking: every `interval` seconds (default: back to back, as
        soon as the previous trajectory is done) pick new random reachable
        targets, plan a trajectory to them and play it back over `playback`
        seconds. Up to `buffer` trajectories are planned ahead in the
        background, in parallel on `workers` threads (see PlanPipeline);
        if planning falls behind, the model waits. `candidates` different
        targets are tried in parallel for each next one, and the first that
        is good enough is taken. Changing the collision mode or obstacles
        drops the planned ones. The planned end-effector paths are drawn as
        lines.

        With `max_miss` (meters), targets that can't be reached closer than
        that are dropped and new ones picked instead, up to `max_retries`
        times in a row (then they are used anyway).
        """
        pv = self.pv
        interval = playback if interval is None else interval
        playback = min(playback, interval)
        radius = 0.012 * self.scale
        colors = ["red", "green", "orange", "purple", "cyan", "magenta"]
        markers, paths = [], []
        for i in range(len(planner.end_effectors)):
            color = colors[i % len(colors)]
            marker = pv.Sphere(radius=radius)
            self.plotter.add_mesh(marker, color=color)
            markers.append((marker, np.array(marker.points)))
            path = pv.PolyData(np.zeros((planner.ik_points + 1, 3)))
            self.plotter.add_mesh(
                path,
                color=color,
                line_width=3,
                point_size=8,
                render_points_as_spheres=True,
                opacity=0.6,
            )
            paths.append(path)

        pipeline = PlanPipeline(planner, buffer, candidates, max_miss, max_retries, workers)
        state = {"n": 0, "dist": 0.0, "since": None}

        def flush():
            """Collision mode or obstacles changed: plans made for the old
            setup are dropped and planning restarts from the end of the
            trajectory playing now."""
            pipeline.restart(planner.trajectory[-1])

        if self.collision is not None:
            planner.extra_objectives = self.collision.current

            def apply(mode):
                planner.extra_objectives = self.collision.current
                flush()

            self._bind_mode_key(apply)
            self._collision_changed = flush

        def miss(targets, traj):
            final = planner.end_effector_paths(traj[-1:])
            return max(np.linalg.norm(f[0] - t) for f, t in zip(final, targets))

        def install(plan):
            targets, (traj, loss, steps, ms) = plan
            planner.trajectory, planner.loss, planner.steps, planner.solve_ms = (
                traj,
                loss,
                steps,
                ms,
            )
            for (marker, base), target in zip(markers, targets):
                marker.points = base + target
            for path, points in zip(paths, planner.end_effector_paths()):
                path.points = points
                path.lines = np.hstack([[len(points)], np.arange(len(points))])
            state.update(since=time.perf_counter(), n=state["n"] + 1, dist=miss(targets, traj))

        def tick(_step=None):
            pipeline.poll()
            elapsed = time.perf_counter() - state["since"]
            if elapsed >= interval and pipeline.ready():
                install(pipeline.pop())
                pipeline.poll()
                elapsed = 0.0
            u = elapsed / playback if playback > 0 else 1.0
            self.set_pose(planner.sample(u))
            waiting = elapsed >= interval
            self.set_status(
                f"target #{state['n']}   "
                + ("planning the next trajectory..." if waiting else f"next in {interval - elapsed:.1f}s")
                + f"   {pipeline.planned} planned ahead, {pipeline.solving} solving"
                + f"\n{planner.ik_points} trajectory points, smoothing {planner.smoothing}, "
                f"frame {min(u, 1.0) * planner.ik_points:.1f}/{planner.ik_points}\n"
                f"solved in {planner.solve_ms:.0f} ms ({planner.steps} steps), "
                f"final distance {state['dist'] * 1000:.1f} mm\n" + self.collision_status()
            )
            self.plotter.render()

        print("Compiling IK solver...")
        while not pipeline.ready():
            pipeline.poll()
            time.sleep(0.01)
        install(pipeline.pop())
        tick()
        self.plotter.add_timer_event(
            max_steps=2**31 - 1, duration=max(1, int(1000 / fps)), callback=tick
        )
        self.plotter.show()
        pipeline.shutdown()


def _smplx_preset(fk_solver: FKSolver, hand: str):
    from jax_ik.smplx_statics import left_arm_bounds_dict, right_arm_bounds_dict

    bounds_dict = left_arm_bounds_dict if hand == "left" else right_arm_bounds_dict
    bones = [f"{hand}_{b}" for b in ("collar", "shoulder", "elbow", "wrist")]
    bounds = []
    for bone in bones:
        lower, upper = bounds_dict[bone]
        bounds.extend(zip(lower, upper))
    return bones, bounds, f"{hand}_wrist"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Realtime IK visualizer: drag the target spheres and the model follows."
    )
    parser.add_argument("model_file", help="URDF, GLB or GLTF model")
    parser.add_argument(
        "--end_effector",
        nargs="+",
        default=None,
        help="End-effector bone(s), one target sphere each",
    )
    parser.add_argument(
        "--controlled_bones",
        nargs="+",
        default=None,
        help="Bones the IK may move. URDF default: movable joints on the "
        "chains to the end-effectors",
    )
    parser.add_argument(
        "--hand",
        choices=["left", "right"],
        default="left",
        help="SMPL-X preset arm when no bones are given",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Default 0.02 when dragging, 0.05 with --random_targets",
    )
    parser.add_argument(
        "--smooth_weight",
        type=float,
        default=0.1,
        help="How strongly each frame sticks to the previous pose",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help="Max IK steps per solve. Default 100 per frame when dragging (30 "
        "with --collision), 1000 per trajectory with --random_targets",
    )
    parser.add_argument(
        "--random_targets",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Instead of dragging, pick new reachable targets every SECONDS "
        "and play a smoothed trajectory to them",
    )
    parser.add_argument(
        "--ik_points", type=int, default=10, help="Trajectory frames (--random_targets)"
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.05,
        help="CombinedDerivativeObj weight for trajectories (--random_targets)",
    )
    parser.add_argument(
        "--playback",
        type=float,
        default=2.0,
        help="Seconds to play each trajectory (--random_targets)",
    )
    parser.add_argument(
        "--collision",
        choices=["off", "soft", "hard"],
        default="off",
        help="Self-collision handling (cycle live with the 'm' key)",
    )
    parser.add_argument(
        "--collision_margin",
        type=float,
        default=0.005,
        help="Clearance to keep between links, meters",
    )
    parser.add_argument(
        "--collision_resolution",
        type=float,
        default=0.01,
        help="Collision proxy resolution, meters (smaller is tighter, slower to build)",
    )
    parser.add_argument(
        "--collision_weight",
        type=float,
        default=None,
        help="Collision penalty weight (default 100 soft, 1000 hard)",
    )
    parser.add_argument(
        "--hide_collision",
        action="store_true",
        help="Start with the collision proxies hidden ('c' toggles)",
    )
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--no_mesh", action="store_true", help="Only draw the skeleton")
    parser.add_argument("--list_bones", action="store_true", help="Print bone names and exit")
    args = parser.parse_args(argv)

    jax.config.update("jax_default_device", jax.devices("cpu")[0])

    is_urdf = pathlib.Path(args.model_file).suffix.lower() == ".urdf"
    probe = FKSolver(args.model_file, controlled_bones=[], do_compute_sdf=False)
    if args.list_bones:
        for name in probe.bone_names:
            print(name, probe.skeleton[name].get("joint_type", ""))
        return

    bounds = None
    end_effectors = args.end_effector
    controlled = args.controlled_bones
    if not is_urdf and not controlled and "left_wrist" in probe.skeleton:
        controlled, bounds, preset_ee = _smplx_preset(probe, args.hand)
        end_effectors = end_effectors or [preset_ee]
    if not end_effectors:
        parser.error("--end_effector is required (use --list_bones to see names)")
    if not controlled:
        if not is_urdf:
            parser.error("--controlled_bones is required for GLB/GLTF models")
        controlled = []
        for ee in end_effectors:
            for bone in default_controlled_bones(probe.skeleton, ee):
                if bone not in controlled:
                    controlled.append(bone)
    if not is_urdf and bounds is None:
        bounds = [(-90, 90)] * 3 * len(controlled)

    trajectory_mode = args.random_targets is not None
    print(f"End-effectors: {end_effectors}")
    print(f"Controlled bones: {controlled}")
    solver = InverseKinematicsSolver(
        args.model_file,
        controlled_bones=controlled,
        bounds=bounds,
        threshold=1e-6,
        # Collision-aware frames cost more, warm starts keep few steps enough
        num_steps=args.num_steps
        or (1000 if trajectory_mode else (30 if args.collision != "off" else 100)),
        compute_sdf=False,
    )
    initial = None
    if is_urdf:
        # Mid-range is clear of the joint limits and, for most arms, of the
        # straight-up singular zero pose IK easily gets stuck in
        initial = (np.asarray(solver.lower_bounds) + np.asarray(solver.upper_bounds)) / 2
    viz = RealtimeVisualizer(
        solver.fk_solver,
        show_mesh=not args.no_mesh,
        title=f"JAX-IK - {pathlib.Path(args.model_file).name}",
    )
    if args.collision != "off":
        from jax_ik.collision import CollisionModel

        print("Building collision model (cached after the first run)...")
        model = CollisionModel.cached(
            solver.fk_solver,
            resolution=args.collision_resolution,
            lower=solver.lower_bounds if is_urdf else None,
            upper=solver.upper_bounds if is_urdf else None,
        )
        modes = CollisionModes(
            model, args.collision_margin, args.collision, args.collision_weight
        )
        viz.attach_collision(modes, show=not args.hide_collision)
    if trajectory_mode:
        planner = TrajectoryPlanner(
            solver,
            end_effectors,
            ik_points=args.ik_points,
            smoothing=args.smoothing,
            learning_rate=args.learning_rate or 0.05,
            initial_angles=initial,
        )
        viz.run_trajectory(
            planner, interval=args.random_targets, playback=args.playback, fps=args.fps
        )
    else:
        controller = IKController(
            solver,
            end_effectors,
            initial_angles=initial,
            learning_rate=args.learning_rate or 0.02,
            smooth_weight=args.smooth_weight,
        )
        viz.run_interactive(controller, fps=args.fps)


if __name__ == "__main__":
    main()
