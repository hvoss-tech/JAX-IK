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
from collections import deque
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
        max_miss: float = None,
        max_retries: int = 20,
        buffer: int = 2,
        candidates: int = 1,
    ) -> None:
        """Blocking: every `interval` seconds pick new random reachable
        targets, plan a trajectory to them and play it back over `playback`
        seconds. Trajectories are planned in the background, each starting
        where the previous one ends, so up to `buffer` of them are ready to
        play right away; if planning falls behind, the model waits.
        `candidates` plans to different targets run in parallel for each
        next one, and the first that is good enough is taken. Changing the
        collision mode or obstacles drops the buffered ones. The planned
        end-effector paths are drawn as lines.

        With `max_miss` (meters), a plan ending further than that from its
        targets is dropped and new targets are planned instead, up to
        `max_retries` times in a row (then the plan is used anyway).
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

        worker = ThreadPoolExecutor(max_workers=candidates)
        state = {
            "n": 0,
            "dist": 0.0,
            "since": None,
            # Accepted plans waiting to be played, as (targets, solve result)
            "queue": deque(),
            # Plans being solved, as (future, slot): only those for the
            # current slot (the next plan after the queue) can still be used
            "pending": [],
            "slot": 0,
            "retries": 0,
        }

        def flush():
            """Collision mode or obstacles changed: plans made for the old
            setup are dropped and planning restarts from the current one."""
            state["queue"].clear()
            state["slot"] += 1
            state["retries"] = 0

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

        def chain_start():
            """Where the next plan starts: the end of the last buffered plan,
            or of the one playing."""
            queue = state["queue"]
            return (queue[-1][1][0] if queue else planner.trajectory)[-1].copy()

        def keep_planning():
            """Collect finished background plans: the first one for the
            current slot that reaches its targets joins the queue, the rest
            are dropped. Then keep `candidates` plans (different targets
            from the same start) running while the buffer isn't full."""
            running = []
            for future, slot in state["pending"]:
                if not future.done():
                    running.append((future, slot))
                    continue
                if slot != state["slot"]:
                    continue
                targets, result = future.result()
                missed = (
                    max_miss is not None
                    and state["retries"] < max_retries
                    and miss(targets, result[0]) > max_miss
                )
                if missed:
                    state["retries"] += 1
                else:
                    state["retries"] = 0
                    state["queue"].append((targets, result))
                    state["slot"] += 1
            state["pending"] = running
            if len(state["queue"]) >= buffer:
                return
            start = chain_start()
            current = sum(slot == state["slot"] for _, slot in running)
            for _ in range(candidates - current):
                targets = planner.random_targets()
                future = worker.submit(
                    lambda targets=targets: (targets, planner.solve_trajectory(start, targets))
                )
                state["pending"].append((future, state["slot"]))

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
            keep_planning()
            elapsed = time.perf_counter() - state["since"]
            if elapsed >= interval and state["queue"]:
                install(state["queue"].popleft())
                keep_planning()
                elapsed = 0.0
            u = elapsed / playback if playback > 0 else 1.0
            self.set_pose(planner.sample(u))
            waiting = elapsed >= interval
            self.set_status(
                f"target #{state['n']}   "
                + ("planning the next trajectory..." if waiting else f"next in {interval - elapsed:.1f}s")
                + f"   {len(state['queue'])} planned ahead"
                + f"\n{planner.ik_points} trajectory points, smoothing {planner.smoothing}, "
                f"frame {min(u, 1.0) * planner.ik_points:.1f}/{planner.ik_points}\n"
                f"solved in {planner.solve_ms:.0f} ms ({planner.steps} steps), "
                f"final distance {state['dist'] * 1000:.1f} mm\n" + self.collision_status()
            )
            self.plotter.render()

        print("Compiling IK solver...")
        while not state["queue"]:
            keep_planning()
            time.sleep(0.01)
        install(state["queue"].popleft())
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
