"""Realtime self-collision model and objective.

Every link with geometry gets a low-resolution, *conservative* proxy built
once from its mesh, in the link's own frame:

* a signed distance grid whose interpolated value never exceeds the true
  distance to the link's mesh, so the proxy can only be larger than the
  mesh, never cut into it,
* sparse points on the surface, each with a radius covering all mesh
  surface closer to it than to any other point, grouped into small spatial
  chunks with a bounding sphere each, and
* a few spheres covering the link's whole volume (broad phase).

Links only move rigidly with their bones, so checking a pair means moving
one link's points into the other link's frame and looking them up in its
grid, in both directions. The result is a lower bound on the true distance
between the two meshes.

Everything is fixed-size array work so it runs well under jit and grad: all
(chunk, other link) combinations get a cheap bounding-sphere gap, and only
the `max_checks` closest are looked up in the grids. Combinations that are
not looked up keep their gap, which is itself a lower bound, so the result
stays conservative either way.

`SelfCollisionObj` turns this into an objective, as a soft penalty or as a
hard constraint the solver never lets get worse (see ik.solve_ik).
"""

import jax
import jax.numpy as jnp
import numpy as np
import trimesh
from jax.tree_util import register_pytree_node_class
from scipy.spatial import cKDTree
from trimesh.remesh import subdivide_to_size

from jax_ik.objectives import ObjectiveFunction

_SQRT3_2 = float(np.sqrt(3.0) / 2.0)
_VIOLATION_TOL = 1e-6
# Bump when the proxy format changes so stale caches are rebuilt
_CACHE_VERSION = 2


class LinkProxy:
    """Collision proxy of one link, everything in the link's bone frame."""

    def __init__(
        self, name, grid, origin, outside, points, radii, chunks, bounds, envelope
    ):
        self.name = name
        self.grid = np.asarray(grid, np.float32)  # conservative SDF at grid nodes
        self.origin = np.asarray(origin, np.float32)
        self.outside = float(outside)  # lower bound for anything off the grid
        self.points = np.asarray(points, np.float32)
        # Every surface point is within radii[i] of some points[i]
        self.radii = np.asarray(radii, np.float32)
        self.chunks = chunks  # list of point index arrays
        self.bounds = bounds  # (centers (n, 3), radii (n,)) covering the volume
        # Low-res watertight mesh enclosing the original: (vertices, faces)
        self.envelope = envelope


class CollisionModel:
    """Conservative low-res collision proxies for a model's links and the
    link pairs checked against each other.

    Hashes by identity so a jitted objective can hold it as a static value.
    """

    def __init__(self, links, pairs, resolution, chunk_size=32, max_checks=384):
        self.links = list(links)
        self.pairs = [tuple(p) for p in pairs]
        self.resolution = float(resolution)
        self.spacing = self.resolution / 2.0
        self.chunk_size = int(chunk_size)
        self.max_checks = int(max_checks)
        # Subtracted from broad-phase gaps so they never exceed the exact
        # lookup (see calibrate_gap_slack); keeps clearances continuous when
        # an entry switches between the two
        self.gap_slack = 0.0
        self._pack()

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    @property
    def link_names(self):
        return [link.name for link in self.links]

    @property
    def point_radius(self):
        """Largest surface coverage radius of any sample point."""
        return float(max((link.radii.max() for link in self.links if len(link.radii)), default=0.0))

    @property
    def slack(self):
        """Rough upper bound on how much larger than the meshes the proxies
        can make two links look (point coverage + distance sampling + grid,
        including grid nodes next to the surface counted as inside)."""
        h = self.spacing
        return self.point_radius + h / 2.0 + 3 * _SQRT3_2 * h

    # --- building ---------------------------------------------------------

    @classmethod
    def from_fk_solver(
        cls,
        fk_solver,
        mesh_data=None,
        resolution=0.01,
        links=None,
        lower=None,
        upper=None,
        n_samples=300,
        always_colliding=0.95,
        min_link_fraction=0.03,
        max_checks=384,
    ):
        """Build proxies from the model's mesh.

        Args:
            fk_solver: An unpruned FKSolver (its bind pose defines the frames).
            mesh_data: Mesh data from load_mesh_data_from_urdf/gltf (loaded
                if not given).
            resolution: Surface point spacing (meters); the distance grid
                uses half of it. Smaller is tighter but slower to build.
            links: Bones to build proxies for. Default: URDF links with real
                geometry, or for skinned meshes every bone owning at least
                `min_link_fraction` of the vertices (smaller bones are merged
                into their nearest included ancestor).
            lower, upper: Angle-vector bounds of fk_solver's controlled bones.
                When given, random poses are sampled and link pairs that
                collide in at least `always_colliding` of them are not
                checked (like MoveIt's default collision matrix). Otherwise
                pairs colliding in the rest pose are not checked.
            max_checks: How many (chunk, link) combinations get the exact
                grid lookup per pose.
        """
        if mesh_data is None:
            from jax_ik.helper import load_mesh_data_from_gltf, load_mesh_data_from_urdf

            if fk_solver.file_type == ".urdf":
                mesh_data = load_mesh_data_from_urdf(fk_solver.model_file, fk_solver)
            else:
                mesh_data = load_mesh_data_from_gltf(fk_solver.model_file, fk_solver)
        geometry = _link_geometry(fk_solver, mesh_data, links, min_link_fraction)
        bind = np.asarray(fk_solver.bind_fk, np.float64)

        proxies = []
        for name, (verts, faces) in geometry.items():
            T_inv = np.linalg.inv(bind[fk_solver.bone_names.index(name)])
            local = verts @ T_inv[:3, :3].T + T_inv[:3, 3]
            mesh = trimesh.Trimesh(local, faces, process=False)
            proxies.append(_build_proxy(name, mesh, resolution, chunk_size=32))

        names = [p.name for p in proxies]
        model = cls(proxies, _non_adjacent_pairs(fk_solver, names), resolution, max_checks=max_checks)
        poses = _sample_poses(fk_solver, lower, upper, n_samples)
        model.calibrate_gap_slack(fk_solver, poses)
        keep = _pairs_to_keep(model, fk_solver, poses, always_colliding)
        model.pairs = [p for p, k in zip(model.pairs, keep) if k]
        model._pack()
        return model

    def calibrate_gap_slack(self, fk_solver, poses, safety=1.25):
        """Measure how much a broad-phase gap can exceed the exact lookup for
        nearby entries, over `poses`, and subtract that (with a safety
        factor) from all gaps.

        Both are lower bounds on the true distance, so this only matters for
        continuity: without it, an entry moving into exact-check range can
        jump from "clear" to "inside the margin", which makes the hard
        constraint reject motion that is actually fine.
        """
        if self.n_entries == 0:
            return
        self.gap_slack = 0.0
        bone_idx = np.asarray([fk_solver.bone_names.index(n) for n in self.link_names])

        def diff(q):
            Ts = fk_solver.compute_fk_from_angles(q)[bone_idx]
            gap = self._gaps(Ts)
            exact, _ = self._narrow(Ts, gap, self.n_entries, 0.0)
            # Exact values are capped at the grid padding; compare only
            # where that cap can't be what's measured
            near = gap < self.links[0].outside - self.resolution
            return jnp.max(jnp.where(near, gap - exact, 0.0))

        worst = float(jnp.max(jax.jit(jax.vmap(diff))(jnp.asarray(poses))))
        self.gap_slack = max(worst, 0.0) * safety

    @classmethod
    def cached(cls, fk_solver, cache_dir=None, **kwargs):
        """from_fk_solver, but stored on disk and reused while the model
        file and the arguments stay the same."""
        import hashlib
        import os
        import pickle

        cache_dir = cache_dir or os.path.join(
            os.path.expanduser("~"), ".cache", "jax_ik", "collision"
        )
        key_parts = [
            _CACHE_VERSION,
            os.path.abspath(fk_solver.model_file),
            os.path.getmtime(fk_solver.model_file),
            fk_solver.bone_names,
            fk_solver.controlled_bones,
            sorted((k, np.asarray(v).tolist() if v is not None else None) for k, v in kwargs.items()),
        ]
        key = hashlib.sha256(repr(key_parts).encode()).hexdigest()[:24]
        path = os.path.join(cache_dir, f"{key}.pkl")
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception:  # noqa: BLE001
                pass
        model = cls.from_fk_solver(fk_solver, **kwargs)
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(model, f)
        return model

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _pack(self):
        """Stack all per-link data into fixed-size arrays."""
        L = len(self.links)
        if L == 0:
            self.n_entries = 0
            return
        shape = np.max([link.grid.shape for link in self.links], axis=0)
        grids = np.empty((L, *shape), np.float32)
        for i, link in enumerate(self.links):
            # Cells beyond a link's own grid are off its grid: same bound
            grids[i] = link.outside
            gx, gy, gz = link.grid.shape
            grids[i, :gx, :gy, :gz] = link.grid
        self.grids = grids
        self.grid_shape = tuple(int(n) for n in shape)
        self.origins = np.stack([link.origin for link in self.links])
        self.outside = float(min(link.outside for link in self.links))

        nb = max(len(link.bounds[1]) for link in self.links)
        self.bound_centers = np.zeros((L, nb, 3), np.float32)
        self.bound_radii = np.full((L, nb), -1e3, np.float32)  # padding never wins
        for i, link in enumerate(self.links):
            c, r = link.bounds
            self.bound_centers[i, : len(r)] = c
            self.bound_radii[i, : len(r)] = r

        chunk_pts, chunk_rad, chunk_link, chunk_c, chunk_r = [], [], [], [], []
        for i, link in enumerate(self.links):
            for idx in link.chunks:
                P = link.points[idx]
                c = 0.5 * (P.min(0) + P.max(0))
                r = float((np.sqrt(((P - c) ** 2).sum(1)) + link.radii[idx]).max())
                pad = np.concatenate([idx, np.full(self.chunk_size - len(idx), idx[0])])
                chunk_pts.append(link.points[pad])
                chunk_rad.append(link.radii[pad])
                chunk_link.append(i)
                chunk_c.append(c)
                chunk_r.append(r)
        self.chunk_points = np.asarray(chunk_pts, np.float32)
        self.chunk_radii = np.asarray(chunk_rad, np.float32)
        self.chunk_link = np.asarray(chunk_link, np.int32)
        self.chunk_center = np.asarray(chunk_c, np.float32)
        self.chunk_radius = np.asarray(chunk_r, np.float32)

        # One entry per (chunk, other link) to look up, per pair and direction
        e_chunk, e_target, e_pair = [], [], []
        for p, (a, b) in enumerate(self.pairs):
            for src, dst in ((a, b), (b, a)):
                for k in np.nonzero(self.chunk_link == src)[0]:
                    e_chunk.append(k)
                    e_target.append(dst)
                    e_pair.append(p)
        self.e_chunk = np.asarray(e_chunk, np.int32)
        self.e_target = np.asarray(e_target, np.int32)
        self.e_pair = np.asarray(e_pair, np.int32)
        self.n_entries = len(e_chunk)

    # --- runtime (jax) ----------------------------------------------------

    def evaluate(self, fk, fk_solver, margin=0.0, max_checks=None):
        """(per-pair clearance lower bounds, soft penalty) for one pose.

        Clearances are exact (as lower bounds) wherever they are below
        `margin`; further away they may be the cheaper broad-phase bound.

        Args:
            fk: Global bone transforms from fk_solver.compute_fk_from_angles.
            fk_solver: The (possibly pruned) FK solver that produced fk.
            margin: Clearance below which the penalty starts.
            max_checks: Cap on exact lookups (default: the model's). Below
                the full amount, clearances stay lower bounds but pairs
                beyond the cap may read as closer than they are, so only
                use it where a rough answer is fine (e.g. a steering
                penalty).
        """
        P = len(self.pairs)
        if P == 0:
            return jnp.zeros(0, jnp.float32), jnp.float32(0.0)
        bone_idx = np.asarray([fk_solver.bone_names.index(n) for n in self.link_names])
        Ts = fk[bone_idx]  # (L, 4, 4)

        gap = self._gaps(Ts)
        # Several consumers below; without the barrier XLA fuses the whole
        # distance computation into each of them and recomputes it
        gap = jax.lax.optimization_barrier(gap)

        # Narrow phase on the closest entries only. Entries further than the
        # margin can't violate it, so only as many as are within it need the
        # exact check; pick the smallest size tier that fits
        if max_checks is not None and max_checks <= 64:
            # Small fixed budget: always look up that many, no switch
            entry_clear, penalty = self._narrow(Ts, gap, min(max_checks, self.n_entries), margin)
            clear = jax.ops.segment_min(entry_clear, self.e_pair, num_segments=P)
            return clear, penalty

        tiers = self._tiers(max_checks)
        need = jnp.sum(gap < margin)
        tier = jnp.searchsorted(jnp.asarray(tiers), need)
        tier = jnp.minimum(tier, len(tiers) - 1)
        # Tier 0: nothing within the margin, the gaps already say all there is
        branches = [lambda g: (g, jnp.float32(0.0))] + [
            (lambda k: lambda g: self._narrow(Ts, g, k, margin))(k) for k in tiers[1:]
        ]
        entry_clear, penalty = jax.lax.switch(tier, branches, gap)
        clear = jax.ops.segment_min(entry_clear, self.e_pair, num_segments=P)
        return clear, penalty

    def _gaps(self, Ts):
        """Broad-phase lower bound per entry: every chunk sphere against
        every link's volume spheres as one dense matrix (cheaper than
        gathering per entry)."""
        R, t = Ts[:, :3, :3], Ts[:, :3, 3]
        chunk_link = jnp.asarray(self.chunk_link)
        cc = jnp.einsum("kij,kj->ki", R[chunk_link], self.chunk_center) + t[chunk_link]
        bc = jnp.einsum("lij,lnj->lni", R, self.bound_centers) + t[:, None, :]
        # |a - b|^2 = |a|^2 - 2ab + |b|^2: a matmul is much faster on CPU
        # than broadcasting the differences
        d2 = (
            jnp.sum(cc * cc, axis=-1)[:, None, None]
            - 2.0 * jnp.einsum("kj,lnj->kln", cc, bc)
            + jnp.sum(bc * bc, axis=-1)[None]
        )
        dist = jnp.sqrt(jnp.maximum(d2, 0.0) + 1e-12) - self.bound_radii[None]
        chunk_gap = jnp.min(dist, axis=-1) - self.chunk_radius[:, None]  # (C, L)
        return chunk_gap[self.e_chunk, self.e_target] - self.gap_slack  # (E,)

    def _tiers(self, max_checks=None):
        cap = self.max_checks if max_checks is None else max_checks
        top = min(cap, self.n_entries)
        return [0] + sorted({min(k, top) for k in (16, 64, 256)} | {top})

    def _narrow(self, Ts, gap, k, margin):
        """Exact grid lookups for the k entries with the smallest gap."""
        _, sel = jax.lax.top_k(-gap, k)
        chunk = jnp.asarray(self.e_chunk)[sel]
        src = jnp.asarray(self.chunk_link)[chunk]
        dst = jnp.asarray(self.e_target)[sel]
        rel = jnp.einsum("kij,kjl->kil", _rigid_inverse(Ts[dst]), Ts[src])
        pts = jnp.einsum("kij,kmj->kmi", rel[:, :3, :3], jnp.asarray(self.chunk_points)[chunk])
        pts = pts + rel[:, None, :3, 3]
        coords = (pts - jnp.asarray(self.origins)[dst][:, None, :]) / self.spacing
        d = self._trilinear(dst, coords) - jnp.asarray(self.chunk_radii)[chunk]  # (k, m)
        # Entries that were not looked up keep their gap, also a lower bound
        d_min = jnp.min(d, axis=1)
        entry_clear = gap.at[sel].set(d_min)
        # Penalize each checked chunk's deepest point: summing every point
        # would scale with the point density and drown out other objectives
        return entry_clear, jnp.sum(jax.nn.relu(margin - d_min) ** 2)

    def _trilinear(self, link, coords):
        """Interpolate grid `link` (k,) at grid coordinates (k, m, 3).

        Direct 8-corner lookup on the flattened grids: much cheaper to
        differentiate than map_coordinates over the stacked 4D array, which
        would also interpolate along the link axis.
        """
        X, Y, Z = self.grid_shape
        dims = jnp.asarray([X, Y, Z])
        c0 = jnp.floor(coords)
        f = coords - c0
        i = c0.astype(jnp.int32)
        inside = jnp.all((i >= 0) & (i <= dims - 2), axis=-1)
        i = jnp.clip(i, 0, dims - 2)
        base = link[:, None] * (X * Y * Z) + i[..., 0] * (Y * Z) + i[..., 1] * Z + i[..., 2]
        corners = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)])
        offsets = jnp.asarray(corners @ np.array([Y * Z, Z, 1]))
        vals = jnp.asarray(self.grids.reshape(-1))[base[..., None] + offsets]  # (k, m, 8)
        w = jnp.where(corners[:, 0] == 1, f[..., 0:1], 1 - f[..., 0:1])
        w = w * jnp.where(corners[:, 1] == 1, f[..., 1:2], 1 - f[..., 1:2])
        w = w * jnp.where(corners[:, 2] == 1, f[..., 2:3], 1 - f[..., 2:3])
        # Off the grid: at least the padding away from the mesh
        return jnp.where(inside, jnp.sum(w * vals, axis=-1), self.outside)

    def clearances(self, angles, fk_solver, margin=0.05):
        """Per-pair clearance lower bounds (meters) for one angle vector,
        exact (as lower bounds) for pairs closer than `margin`."""
        fk = fk_solver.compute_fk_from_angles(jnp.asarray(angles, jnp.float32))
        return self.evaluate(fk, fk_solver, margin)[0]

    def envelope_world(self, fk, fk_solver):
        """Envelope meshes moved to the pose: list of (link index, vertices, faces)."""
        fk = np.asarray(fk)
        out = []
        for i, link in enumerate(self.links):
            if link.envelope is None:
                continue
            T = fk[fk_solver.bone_names.index(link.name)]
            v, f = link.envelope
            out.append((i, v @ T[:3, :3].T + T[:3, 3], f))
        return out


def _rigid_inverse(T):
    R = T[..., :3, :3]
    Rt = jnp.swapaxes(R, -1, -2)
    t = -jnp.einsum("...ij,...j->...i", Rt, T[..., :3, 3])
    inv = jnp.zeros_like(T).at[..., :3, :3].set(Rt).at[..., :3, 3].set(t)
    return inv.at[..., 3, 3].set(1.0)


# --- proxy construction (numpy, offline) ----------------------------------


def _build_proxy(name, mesh, resolution, chunk_size=32, n_bounds=16):
    h = resolution / 2.0
    # Off-grid points read as `pad` away, so margins up to about pad work
    pad = max(3 * resolution, 0.05)
    lo = mesh.bounds[0] - pad
    shape = np.ceil((mesh.bounds[1] + pad - lo) / h).astype(int) + 1
    axes = [lo[k] + h * np.arange(shape[k]) for k in range(3)]
    nodes = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)

    # Distance d to dense surface samples overestimates the true distance by
    # at most the sample spacing (every point of a subdivided triangle is
    # within its longest edge of a corner) and never underestimates it.
    # Lower bounds on the signed distance: d - edge outside, -d inside
    edge = h / 2.0
    dense, _ = subdivide_to_size(mesh.vertices, mesh.faces, max_edge=edge)
    # Nodes further than `cap` only need to know that they are: the cap is
    # still a lower bound outside, and inside it stays negative
    cap = max(pad, 0.1)
    dist, _ = cKDTree(dense).query(nodes, distance_upper_bound=cap, workers=-1)
    dist = np.minimum(dist, cap)

    # Inside: filled voxels. Voxels touching the surface count as inside,
    # which can only make the proxy larger
    filled = mesh.voxelized(h).fill()
    inside = filled.is_filled(nodes)
    sdf = np.where(inside, -dist, dist - edge)

    # Trilinear interpolation of a 1-Lipschitz function is off by at most
    # half a cell diagonal
    grid = (sdf - _SQRT3_2 * h).reshape(shape)

    # Sparse samples on the surface: per surface voxel, the dense sample
    # closest to its center. Each gets the radius of the surface it covers:
    # its furthest dense sample (nearest-sample assignment) plus the dense
    # samples' own spacing bound
    centers = mesh.voxelized(resolution).points
    dense_tree = cKDTree(dense)
    _, pick = dense_tree.query(centers, workers=-1)
    points = dense[np.unique(pick)]
    dist_to_point, owner = cKDTree(points).query(dense, workers=-1)
    radii = np.zeros(len(points))
    np.maximum.at(radii, owner, dist_to_point)
    radii = radii + edge
    chunks = _split_chunks(points, chunk_size)

    # Broad-phase spheres over the whole volume (vertices + filled voxels)
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    volume = np.concatenate(
        [mesh.vertices, (filled.points[:, None] + corners * h / 2).reshape(-1, 3)]
    )
    bounds = _cover_spheres(volume, n_bounds)
    envelope = _envelope_mesh(grid, lo, h)
    # Points off the grid are at least `pad` from the mesh's bounding box
    return LinkProxy(name, grid, lo, pad, points, radii, chunks, bounds, envelope)


def _split_chunks(points, size):
    """Split point indices into spatially compact groups of <= size."""
    out = []
    stack = [np.arange(len(points))]
    while stack:
        idx = stack.pop()
        if len(idx) <= size:
            out.append(idx)
            continue
        P = points[idx]
        axis = int(np.argmax(P.max(0) - P.min(0)))
        order = idx[np.argsort(P[:, axis])]
        half = len(order) // 2
        stack += [order[:half], order[half:]]
    return out


def _enclosing_sphere(P, iters=40):
    c = P.mean(0)
    for k in range(iters):
        far = P[np.argmax(((P - c) ** 2).sum(1))]
        c = c + (far - c) / (k + 2)
    return c, float(np.sqrt(((P - c) ** 2).sum(1)).max())


def _cover_spheres(P, n):
    """Up to n spheres that together contain every point of P."""
    clusters = [P]
    spheres = [_enclosing_sphere(P)]
    while len(clusters) < n:
        i = int(np.argmax([s[1] if len(c) > 1 else -1.0 for c, s in zip(clusters, spheres)]))
        Q = clusters[i]
        if len(Q) < 2:
            break
        axis = int(np.argmax(Q.max(0) - Q.min(0)))
        m = np.median(Q[:, axis])
        a, b = Q[Q[:, axis] <= m], Q[Q[:, axis] > m]
        if len(a) == 0 or len(b) == 0:
            break
        clusters[i : i + 1] = [a, b]
        spheres[i : i + 1] = [_enclosing_sphere(a), _enclosing_sphere(b)]
    return (
        np.asarray([s[0] for s in spheres], np.float32),
        np.asarray([s[1] for s in spheres], np.float32),
    )


def _envelope_mesh(grid, origin, h):
    """Zero level set of the conservative grid: a low-res watertight mesh
    containing the original mesh."""
    try:
        from skimage.measure import marching_cubes

        verts, faces, _, _ = marching_cubes(grid, level=0.0, spacing=(h, h, h))
    except Exception:  # noqa: BLE001
        return None
    return verts + origin, faces


def _link_geometry(fk_solver, mesh_data, links, min_link_fraction):
    """{bone name: (vertices in bind pose, faces)} per collision link."""
    verts = np.asarray(mesh_data["vertices"])[:, :3].astype(np.float64)
    faces = np.asarray(mesh_data["faces"])
    names = fk_solver.bone_names

    if "skin_joints" in mesh_data:
        weights = np.asarray(mesh_data["skin_weights"])
        joints = np.asarray(mesh_data["skin_joints"])
        owner = joints[np.arange(len(joints)), np.argmax(weights, axis=1)]
        if links is None:
            counts = np.bincount(owner, minlength=len(names))
            links = [n for n, c in zip(names, counts) if c >= min_link_fraction * len(verts)]
    else:
        owner = np.asarray(mesh_data["vertex_assignment"])
        if links is None:
            links = _urdf_links_with_geometry(fk_solver)

    unknown = [n for n in links if n not in names]
    if unknown:
        raise ValueError(f"Collision links not found in skeleton: {unknown}")

    # Vertices of bones that are not collision links go to the nearest
    # included ancestor (e.g. fingers into the hand)
    included = set(links)
    target = []
    for name in names:
        cur = name
        while cur is not None and cur not in included:
            cur = fk_solver.skeleton[cur]["parent"]
        target.append(cur or "")
    vert_link = np.asarray(target, dtype=object)[owner]
    face_link = vert_link[faces[:, 0]]

    geometry = {}
    for name in links:
        f = faces[face_link == name]
        if len(f) == 0:
            continue
        used, remap = np.unique(f, return_inverse=True)
        geometry[name] = (verts[used], remap.reshape(-1, 3))
    return geometry


def _urdf_links_with_geometry(fk_solver):
    import urchin

    robot = urchin.URDF.load(fk_solver.model_file, lazy_load_meshes=True)
    out = []
    for link in robot.links:
        geoms = [v.geometry for v in link.visuals] + [c.geometry for c in link.collisions]
        has_mesh = any(g is not None and g.mesh is not None for g in geoms)
        if has_mesh and link.name in fk_solver.bone_names:
            out.append(link.name)
    return out


def _non_adjacent_pairs(fk_solver, link_names):
    """All link pairs except each link with its nearest collision-link
    ancestor (joined parts always touch)."""
    included = set(link_names)
    parent = {}
    for name in link_names:
        cur = fk_solver.skeleton[name]["parent"]
        while cur is not None and cur not in included:
            cur = fk_solver.skeleton[cur]["parent"]
        parent[name] = cur
    pairs = []
    for i, a in enumerate(link_names):
        for j in range(i + 1, len(link_names)):
            b = link_names[j]
            if parent[a] != b and parent[b] != a:
                pairs.append((i, j))
    return pairs


def _sample_poses(fk_solver, lower, upper, n_samples):
    """Random poses within bounds, or just the rest pose without bounds."""
    if lower is not None and upper is not None and n_samples > 0:
        lower = np.asarray(lower, np.float32)
        upper = np.asarray(upper, np.float32)
        rng = np.random.default_rng(0)
        return rng.uniform(lower, upper, (n_samples, lower.shape[0])).astype(np.float32)
    return np.zeros((1, len(fk_solver.controlled_indices) * 3), np.float32)


def _pairs_to_keep(model, fk_solver, poses, threshold):
    if not model.pairs:
        return []
    if len(poses) == 1:
        threshold = 0.5
    clear = jax.jit(jax.vmap(lambda q: model.clearances(q, fk_solver, 0.0)))
    hits = np.mean(np.asarray(clear(jnp.asarray(poses))) < 0, axis=0)
    return hits < threshold


# --- objective --------------------------------------------------------------


@register_pytree_node_class
class SelfCollisionObj(ObjectiveFunction):
    """Keeps the model's links from intersecting each other.

    Soft (hard=False): penalty on how far proxy points sit inside another
    link's proxy, or inside the `margin` around it, traded off against the
    other objectives.

    Hard (hard=True): the solver never accepts a step that pushes any link
    pair's clearance further below `margin`. A collision-free start
    therefore stays collision-free, and a colliding start can only improve.
    Poses between consecutive trajectory frames are checked too
    (`substeps`), so motion can't tunnel through a link between frames. The
    penalty is still added, but already starts `influence` before the margin
    so it steers the pose around obstacles instead of only stopping at them.

    Args:
        model: CollisionModel of the solver's skeleton.
        margin: Required clearance in meters (a buffer zone around the
            proxies).
        weight: Penalty weight (default 100 soft, 1000 hard).
        hard: Enforce as a hard constraint.
        influence: Hard mode: extra distance beyond the margin where the
            penalty already pushes away.
        substeps: Interpolated poses checked between frames (hard mode).
        penalty_checks: Exact lookups used for the penalty per pose; the
            hard check always uses the model's full amount.
    """

    def __init__(
        self,
        model,
        margin=0.0,
        weight=None,
        hard=False,
        influence=0.02,
        substeps=2,
        penalty_checks=32,
    ):
        self.model = model
        self.margin = np.asarray(margin, np.float32)
        if weight is None:
            # Hard mode: strong steering keeps the pose off the boundary,
            # which avoids costly repairs and getting stuck on it
            weight = 1000.0 if hard else 100.0
        self.weight = np.asarray(weight, np.float32)
        self.hard = bool(hard)
        self.influence = np.asarray(influence if hard else 0.0, np.float32)
        self.substeps = int(substeps)
        # The penalty only steers (hard mode) or trades off (soft mode), so
        # it looks at the closest entries only; the hard check is exact
        self.penalty_checks = int(penalty_checks)

    @property
    def is_hard_constraint(self):
        return self.hard

    def referenced_bones(self):
        return tuple(self.model.link_names)

    def get_params(self):
        # The base class would turn arrays into lists; pass everything as is
        return {
            "model": self.model,
            "margin": self.margin,
            "weight": self.weight,
            "hard": self.hard,
            "influence": self.influence,
            "substeps": self.substeps,
            "penalty_checks": self.penalty_checks,
        }

    def update_params(self, params):
        for key in ("model", "hard", "substeps", "penalty_checks"):
            if key in params:
                setattr(self, key, params[key])
        for key in ("margin", "weight", "influence"):
            if key in params:
                setattr(self, key, np.asarray(params[key], np.float32))

    def _frame(self, cfg, fk_solver, margin, max_checks=None):
        fk = fk_solver.compute_fk_from_angles(cfg)
        return self.model.evaluate(fk, fk_solver, margin, max_checks)

    def __call__(self, X, fk_solver):
        if not self.model.pairs:
            return jnp.float32(0.0)
        # Frame 0 of a trajectory is always the fixed start pose: its
        # penalty is a constant
        X = X[None] if X.ndim == 1 else (X[1:] if X.shape[0] > 1 else X)
        reach = self.margin + self.influence
        # lax.map, not vmap: under vmap the narrow-phase size switch would
        # turn into computing every size for every frame
        _, penalty = jax.lax.map(
            lambda cfg: self._frame(cfg, fk_solver, reach, self.penalty_checks), X
        )
        return jnp.mean(penalty) * self.weight

    def check_poses(self, X):
        """Frames plus `substeps` interpolated poses between neighbours.
        Frame 0 of a trajectory is the fixed start pose and can't change, so
        it is left out; the motion away from it is still checked."""
        X = X[None] if X.ndim == 1 else X
        if X.shape[0] < 2:
            return X
        if self.substeps <= 0:
            return X[1:]
        t = np.linspace(0.0, 1.0, self.substeps + 2)[1:-1].astype(np.float32)
        mids = X[:-1, None] * (1 - t[None, :, None]) + X[1:, None] * t[None, :, None]
        return jnp.concatenate([X[1:], mids.reshape(-1, X.shape[-1])])

    def violation(self, X, fk_solver, extra=0.0):
        """How far each checked pose and pair is inside the margin (>= 0).

        `extra` widens the margin, e.g. to get a gradient for constraints
        that are just touching it."""
        if not self.model.pairs:
            return jnp.zeros(1, jnp.float32)
        margin = self.margin + extra
        poses = self.check_poses(X)
        clear, _ = jax.lax.map(lambda cfg: self._frame(cfg, fk_solver, margin), poses)
        # A micrometer of slack so poses resting exactly on the margin don't
        # register float noise as violations
        return jax.nn.relu(margin - clear - _VIOLATION_TOL).reshape(-1)
