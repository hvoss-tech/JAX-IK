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
import pathlib
import time
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
    ):
        self.solver = solver
        self.end_effectors = list(end_effectors)
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

    def random_targets(self, spread: float = 0.6) -> list:
        """End-effector positions of a random legal pose, so every target is
        reachable. `spread` is the fraction of each joint range (around its
        middle) to sample from.
        """
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
        trajectory plays. Returns (trajectory, loss, steps, solve ms)."""
        start = time.perf_counter()
        traj, loss, steps = self.solver.solve(
            initial_rotations=np.asarray(start_pose, dtype=np.float32),
            learning_rate=self.learning_rate,
            mandatory_objective_functions=[
                DistanceObjTraj(
                    bone_name=name,
                    target_points=[np.asarray(target, dtype=np.float32)],
                    use_head=True,
                )
                for name, target in zip(self.end_effectors, targets)
            ],
            optional_objective_functions=[
                CombinedDerivativeObj(max_order=3, weight=self.smoothing),
                *self.extra_objectives,
            ],
            ik_points=self.ik_points,
            patience=self.patience,
            verbose=False,
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


class CollisionModes:
    """The objectives for each self-collision mode, cycled with the 'm' key."""

    NAMES = ("off", "soft", "hard")

    def __init__(self, model, margin=0.005, mode="hard", weight=None):
        from jax_ik.collision import SelfCollisionObj

        self.model = model
        self.margin = float(margin)
        self.objectives = {
            "off": [],
            "soft": [SelfCollisionObj(model, margin=margin, weight=weight, hard=False)],
            "hard": [SelfCollisionObj(model, margin=margin, weight=weight, hard=True)],
        }
        self.mode = mode

    def cycle(self):
        self.mode = self.NAMES[(self.NAMES.index(self.mode) + 1) % len(self.NAMES)]
        return self.mode

    @property
    def current(self):
        return self.objectives[self.mode]


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
        self._update_collision(self._angles)

    def _toggle_envelope(self):
        actor = self._envelope_actor
        actor.SetVisibility(not actor.GetVisibility())
        self.plotter.render()

    def _update_collision(self, angles):
        modes = self.collision
        clear = np.asarray(self._clearances(np.asarray(angles, np.float32)))
        self.min_clearance = float(clear.min()) if len(clear) else float("inf")
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

    def collision_status(self) -> str:
        if self.collision is None:
            return ""
        state = "COLLIDING" if self.min_clearance < 0 else (
            "inside margin" if self.min_clearance < self.collision.margin else "clear"
        )
        return (
            f"collision: {self.collision.mode} (margin {self.collision.margin * 1000:.0f} mm)"
            f"   min clearance {self.min_clearance * 1000:.1f} mm, {state}\n"
            "keys: m = cycle collision mode, c = show/hide proxies\n"
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
        interval: float = 5.0,
        playback: float = 2.0,
        fps: float = 60.0,
    ) -> None:
        """Blocking: every `interval` seconds pick new random reachable
        targets, plan a trajectory to them and play it back over `playback`
        seconds. The next trajectory is planned in the background while the
        current one plays; if planning takes longer, the model waits for it.
        The planned end-effector paths are drawn as lines.
        """
        pv = self.pv
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

        if self.collision is not None:
            planner.extra_objectives = self.collision.current

            def apply(mode):
                planner.extra_objectives = self.collision.current

            self._bind_mode_key(apply)

        worker = ThreadPoolExecutor(max_workers=1)
        state = {"n": 0, "dist": 0.0, "future": None, "since": None}

        def start_planning():
            targets = planner.random_targets()
            start = planner.trajectory[-1].copy()
            state["future"] = worker.submit(
                lambda: (targets, planner.solve_trajectory(start, targets))
            )

        def install(result):
            targets, (traj, loss, steps, ms) = result
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
            final = planner.end_effector_paths(traj[-1:])
            state.update(
                since=time.perf_counter(),
                n=state["n"] + 1,
                dist=max(np.linalg.norm(f[0] - t) for f, t in zip(final, targets)),
            )
            # Plan the next one while this one plays
            start_planning()

        def tick(_step=None):
            elapsed = time.perf_counter() - state["since"]
            if elapsed >= interval and state["future"].done():
                install(state["future"].result())
                elapsed = 0.0
            u = elapsed / playback if playback > 0 else 1.0
            self.set_pose(planner.sample(u))
            waiting = elapsed >= interval
            self.set_status(
                f"target #{state['n']}   "
                + ("planning the next trajectory..." if waiting else f"next in {interval - elapsed:.1f}s")
                + f"\n{planner.ik_points} trajectory points, smoothing {planner.smoothing}, "
                f"frame {min(u, 1.0) * planner.ik_points:.1f}/{planner.ik_points}\n"
                f"solved in {planner.solve_ms:.0f} ms ({planner.steps} steps), "
                f"final distance {state['dist'] * 1000:.1f} mm\n" + self.collision_status()
            )
            self.plotter.render()

        print("Compiling IK solver...")
        start_planning()
        install(state["future"].result())
        tick()
        self.plotter.add_timer_event(
            max_steps=2**31 - 1, duration=max(1, int(1000 / fps)), callback=tick
        )
        self.plotter.show()
        worker.shutdown(wait=False, cancel_futures=True)


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
