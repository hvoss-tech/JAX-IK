import numpy as np
import pytest
import trimesh
from conftest import PANDA_ARM
from jax.tree_util import tree_flatten, tree_unflatten

import jax
import jax.numpy as jnp
from jax_ik.collision import TRAJECTORY_CHECKS_PER_FRAME, CollisionModel, SelfCollisionObj
from jax_ik.helper import load_mesh_data_from_urdf
from jax_ik.ik import InverseKinematicsSolver
from jax_ik.objectives import DistanceObjTraj

R = 0.1  # sphere radius
ARM = 0.3  # swing radius of the moving sphere

TWO_SPHERES = """<?xml version="1.0"?>
<robot name="spheres">
  <link name="base"/>
  <link name="a"><visual><geometry><mesh filename="sphere.stl"/></geometry></visual></link>
  <link name="arm"/>
  <link name="b"><visual><geometry><mesh filename="sphere.stl"/></geometry></visual></link>
  <joint name="fix_a" type="fixed">
    <parent link="base"/><child link="a"/><origin xyz="0.3 0 0"/>
  </joint>
  <joint name="swing" type="revolute">
    <parent link="base"/><child link="arm"/><axis xyz="0 0 1"/>
    <limit lower="-3.1" upper="3.1" effort="1" velocity="1"/>
  </joint>
  <joint name="fix_b" type="fixed">
    <parent link="arm"/><child link="b"/><origin xyz="0.3 0 0"/>
  </joint>
</robot>
"""


def sphere_distance(theta):
    # a sits at (ARM, 0), b at ARM * (cos t, sin t)
    return 2 * ARM * abs(np.sin(theta / 2)) - 2 * R


@pytest.fixture(scope="module")
def spheres(tmp_path_factory):
    d = tmp_path_factory.mktemp("spheres")
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=R)
    mesh.export(str(d / "sphere.stl"))
    urdf = d / "spheres.urdf"
    urdf.write_text(TWO_SPHERES)
    solver = InverseKinematicsSolver(
        str(urdf), controlled_bones=["arm"], threshold=1e-8, num_steps=300, compute_sdf=False
    )
    model = CollisionModel.from_fk_solver(
        solver.fk_solver,
        resolution=0.01,
        lower=solver.lower_bounds,
        upper=solver.upper_bounds,
        n_samples=100,
    )
    # Icosphere faces sit slightly inside the true sphere
    facet = R - np.min(np.linalg.norm(mesh.triangles_center, axis=1))
    return solver, model, facet


@pytest.fixture(scope="module")
def panda(panda_urdf):
    solver = InverseKinematicsSolver(
        panda_urdf, controlled_bones=PANDA_ARM, threshold=1e-6, num_steps=300, compute_sdf=False
    )
    # Default settings, built once and cached on disk
    model = CollisionModel.cached(
        solver.fk_solver, lower=solver.lower_bounds, upper=solver.upper_bounds
    )
    return solver, model


def _angles(theta):
    return np.array([theta, 0.0, 0.0], np.float32)


# --- proxies -------------------------------------------------------------------


def test_links_pairs_and_geometry(spheres):
    _, model, _ = spheres
    # Geometry-less links get no proxy; a and b are not adjacent
    assert model.link_names == ["a", "b"]
    assert model.pairs == [(0, 1)]
    for link in model.links:
        assert link.envelope is not None
        assert len(link.points) > 0


def test_proxy_never_overstates_distance(panda):
    solver, model = panda
    fk = solver.fk_solver
    md = load_mesh_data_from_urdf(fk.model_file, fk)
    V, F, A = (np.asarray(md[k]) for k in ("vertices", "faces", "vertex_assignment"))
    bind = np.asarray(fk.bind_fk, np.float64)
    rng = np.random.default_rng(0)
    for link in model.links:
        b = fk.bone_names.index(link.name)
        mesh = trimesh.Trimesh(V[:, :3], F[A[F[:, 0]] == b])
        mesh.remove_unreferenced_vertices()
        mesh.apply_transform(np.linalg.inv(bind[b]))
        X = mesh.sample(800) + rng.normal(scale=0.03, size=(800, 3))
        _, dist, _ = trimesh.proximity.closest_point(mesh, X)
        coords = jnp.asarray((X - link.origin) / model.spacing)
        G = np.asarray(
            jax.scipy.ndimage.map_coordinates(
                jnp.asarray(link.grid), coords.T, order=1, mode="constant", cval=link.outside
            )
        )
        # Never more than the unsigned distance to the surface
        assert np.all(G <= dist + 1e-6), link.name
        # The surface itself is inside the proxy
        coords = jnp.asarray((mesh.vertices - link.origin) / model.spacing)
        Gv = jax.scipy.ndimage.map_coordinates(jnp.asarray(link.grid), coords.T, order=1)
        assert np.all(np.asarray(Gv) <= 1e-6), link.name


def test_surface_points_cover_mesh(spheres):
    solver, model, _ = spheres
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=R)
    for link in model.links:
        dense = mesh.sample(4000)
        d = np.min(np.linalg.norm(dense[:, None] - link.points[None], axis=-1), axis=1)
        assert d.max() <= model.point_radius + 1e-9


@pytest.mark.parametrize("theta", [0.75, 0.8, 0.9, 1.0, 1.2, 2.0, 3.0, -1.0])
def test_clearance_is_conservative(spheres, theta):
    solver, model, facet = spheres
    c = float(model.clearances(_angles(theta), solver.fk_solver)[0])
    # Never claims more room than there is
    assert c <= sphere_distance(theta) + 2 * facet


@pytest.mark.parametrize("theta", [0.72, 0.75, 0.8, 0.85])
def test_clearance_is_tight_up_close(spheres, theta):
    # Within the evaluation margin (5cm) the exact lookup is used and the
    # proxy only adds a little on top of the true distance
    solver, model, facet = spheres
    true = sphere_distance(theta)
    assert 0 < true < 0.05
    c = float(model.clearances(_angles(theta), solver.fk_solver)[0])
    assert c >= true - model.slack - 2 * facet


@pytest.mark.parametrize("theta", [0.0, 0.2, 0.4, 0.6, -0.5])
def test_overlap_always_reads_negative(spheres, theta):
    solver, model, _ = spheres
    assert sphere_distance(theta) < 0
    assert float(model.clearances(_angles(theta), solver.fk_solver)[0]) < 0


def test_far_apart_reads_clear(spheres):
    solver, model, _ = spheres
    assert float(model.clearances(_angles(3.0), solver.fk_solver)[0]) > 0.2


def test_skipped_entries_never_hide_collisions(panda):
    solver, model = panda
    fk = solver.fk_solver
    full = CollisionModel(model.links, model.pairs, model.resolution, max_checks=10**6)
    full.gap_slack = model.gap_slack
    lo, hi = np.asarray(solver.lower_bounds), np.asarray(solver.upper_bounds)
    Q = jnp.asarray(np.random.default_rng(1).uniform(lo, hi, (150, lo.size)).astype(np.float32))
    # Chunked: the full check vmapped over all poses at once needs > 12 GB
    a = np.asarray(jax.lax.map(lambda q: model.clearances(q, fk, 0.0), Q, batch_size=10))
    b = np.asarray(jax.lax.map(lambda q: full.clearances(q, fk, 0.0), Q, batch_size=10))
    assert not np.any((a >= 0) & (b < 0))


def test_adjacent_pairs_are_not_checked(panda):
    _, model = panda
    names = model.link_names
    pairs = {(names[i], names[j]) for i, j in model.pairs}
    for a, b in [("panda_link0", "panda_link1"), ("panda_link3", "panda_link4"), ("panda_link7", "panda_hand")]:
        assert (a, b) not in pairs and (b, a) not in pairs
    # Links without real geometry get no proxy
    assert "panda_link8" not in names and "panda_hand_tcp" not in names


def test_unknown_link_raises(panda):
    solver, _ = panda
    with pytest.raises(ValueError, match="nope"):
        CollisionModel.from_fk_solver(solver.fk_solver, links=["nope"])


def test_envelope_follows_links(spheres):
    solver, model, _ = spheres
    fk = solver.fk_solver
    out0 = model.envelope_world(fk.compute_fk_from_angles(_angles(0.0)), fk)
    out1 = model.envelope_world(fk.compute_fk_from_angles(_angles(1.0)), fk)
    # a is fixed, b moves
    assert np.allclose(out0[0][1], out1[0][1])
    assert not np.allclose(out0[1][1], out1[1][1])


def test_cached_model_is_reused(spheres, tmp_path):
    solver, _, _ = spheres
    m1 = CollisionModel.cached(solver.fk_solver, cache_dir=str(tmp_path), resolution=0.02)
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    m2 = CollisionModel.cached(solver.fk_solver, cache_dir=str(tmp_path), resolution=0.02)
    assert m2.pairs == m1.pairs and np.allclose(m2.grids, m1.grids)
    CollisionModel.cached(solver.fk_solver, cache_dir=str(tmp_path), resolution=0.03)
    assert len(list(tmp_path.iterdir())) == 2


# --- objective -------------------------------------------------------------------


# --- grouped broad phase and shared penalty budget ----------------------------


def _random_link_transforms(solver, model, n, seed):
    fk = solver.fk_solver
    lo, hi = np.asarray(solver.lower_bounds), np.asarray(solver.upper_bounds)
    Q = np.random.default_rng(seed).uniform(lo, hi, (n, lo.size)).astype(np.float32)
    F = jax.vmap(fk.compute_fk_from_angles)(jnp.asarray(Q))
    idx = np.asarray([fk.bone_names.index(name) for name in model.link_names])
    return jnp.asarray(Q), F, F[:, idx]


def test_groups_cover_every_entry_once(panda):
    _, model = panda
    ids = model.ge_entries[model.ge_entries >= 0]
    assert np.array_equal(np.sort(ids), np.arange(model.n_entries))
    for g, dst, members in zip(model.ge_group, model.ge_target, model.ge_entries):
        members = members[members >= 0]
        assert len(members) > 0
        chunks = model.e_chunk[members]
        # All of a group entry's entries come from the group's chunks and
        # look up the same other link
        assert np.all(model.chunk_link[chunks] == model.group_link[g])
        assert np.all(model.e_target[members] == dst)
        # The group sphere encloses its chunks' spheres
        reach = np.linalg.norm(model.chunk_center[chunks] - model.group_center[g], axis=1)
        assert np.all(reach + model.chunk_radius[chunks] <= model.group_radius[g] + 1e-6)


def test_group_gaps_bound_entry_gaps(panda):
    solver, model = panda
    _, _, Ts = _random_link_transforms(solver, model, 20, seed=3)
    group_gap = np.asarray(model._group_gaps(Ts)).reshape(len(Ts), -1)
    gap = np.asarray(jax.vmap(model._gaps)(Ts))
    ent = model.ge_entries
    member_gap = np.where(ent >= 0, gap[:, np.maximum(ent, 0)], np.inf).min(-1)
    assert np.all(group_gap <= member_gap + 1e-5)


@pytest.mark.parametrize("margin", [0.005, 0.025])
def test_full_budget_penalty_matches_exact_evaluation(panda, margin):
    # With a budget covering every entry, the grouped selection must find
    # every entry within the margin: same value and gradient as the exact
    # narrow phase of evaluate
    solver, model = panda
    fk = solver.fk_solver
    Q, _, _ = _random_link_transforms(solver, model, 6, seed=4)
    full = CollisionModel(model.links, model.pairs, model.resolution, max_checks=10**6)
    full.gap_slack = model.gap_slack

    def exact(q):
        return full.evaluate(fk.compute_fk_from_angles(q), fk, margin)[1]

    def grouped(q):
        return model.penalty(fk.compute_fk_from_angles(q), fk, margin, model.n_entries)

    hits = 0
    for q in Q:
        a, b = float(exact(q)), float(grouped(q))
        assert b == pytest.approx(a, rel=1e-4, abs=1e-9)
        ga, gb = np.asarray(jax.grad(exact)(q)), np.asarray(jax.grad(grouped)(q))
        assert np.allclose(gb, ga, rtol=1e-3, atol=1e-6)
        hits += a > 0
    assert hits > 0  # the random poses do collide


def test_batched_penalty_with_full_budget_is_sum_of_poses(panda):
    solver, model = panda
    fk = solver.fk_solver
    _, F, _ = _random_link_transforms(solver, model, 4, seed=5)
    batched = float(model.penalty(F, fk, 0.01, model.n_entries * len(F)))
    single = sum(float(model.penalty(f, fk, 0.01, model.n_entries)) for f in F)
    assert single > 0
    assert batched == pytest.approx(single, rel=1e-4)


def test_shared_budget_goes_to_the_colliding_frame(panda):
    solver, model = panda
    fk = solver.fk_solver
    lo, hi = np.asarray(solver.lower_bounds), np.asarray(solver.upper_bounds)
    clear = ((lo + hi) / 2).astype(np.float32)
    Q, _, _ = _random_link_transforms(solver, model, 30, seed=6)
    margin = 0.005
    hit = [q for q in Q if float(model.penalty(fk.compute_fk_from_angles(q), fk, margin, 16)) > 0]
    frames = jnp.asarray(np.stack([clear] * 9 + [np.asarray(hit[0])]))
    F = jax.vmap(fk.compute_fk_from_angles)(frames)
    # 16 lookups for all 10 frames, far fewer than 16 per frame
    value, grad = jax.value_and_grad(lambda F: model.penalty(F, fk, margin, 16))(F)
    assert float(value) > 0
    grad = np.asarray(grad)
    assert np.any(grad[9] != 0)
    assert np.all(grad[:9] == 0)


@pytest.mark.parametrize("budget", ["padded_first", "full"])
def test_selection_never_counts_padding_or_duplicates(panda, budget):
    # Group entries of links with fewer chunks than a group holds are padded;
    # the padding must never be picked, even when those group entries are
    # the closest ones, and no entry may be picked twice (it would count
    # twice in the penalty)
    solver, model = panda
    _, _, Ts = _random_link_transforms(solver, model, 2, seed=7)
    F = len(Ts)
    padded = np.any(model.ge_entries < 0, axis=1)
    assert padded.any()
    if budget == "full":
        group_gap = model._group_gaps(Ts)
        k = model.n_entries * F
    else:
        group_gap = jnp.asarray(np.tile(np.where(padded, -1.0, 1.0).astype(np.float32), F))
        k = 3 * int(padded.sum()) * F
    frame, entry, valid = (np.asarray(a) for a in model._select(Ts.reshape(-1, 4, 4), group_gap, F, k))
    assert valid.all() and len(entry) == k
    picked = set(zip(frame.tolist(), entry.tolist()))
    assert len(picked) == k
    if budget == "full":
        assert picked == {(f, e) for f in range(F) for e in range(model.n_entries)}


def test_trajectory_penalty_shares_one_budget(panda, monkeypatch):
    solver, model = panda
    fk = solver.fk_solver
    Q, _, _ = _random_link_transforms(solver, model, 11, seed=8)
    budgets = []
    penalty = model.penalty

    def spy(fk_batch, fk_solver, margin, max_checks):
        budgets.append((fk_batch.shape[0], max_checks))
        return penalty(fk_batch, fk_solver, margin, max_checks)

    monkeypatch.setattr(model, "penalty", spy)
    SelfCollisionObj(model, margin=0.01)(Q, fk)
    # Frame 0 is the fixed start: 10 frames share one default budget
    assert budgets == [(10, 10 * TRAJECTORY_CHECKS_PER_FRAME)]
    budgets.clear()
    # Hard mode keeps the single-pose amount per frame
    SelfCollisionObj(model, margin=0.01, hard=True, penalty_checks=20)(Q, fk)
    assert budgets == [(10, 10 * 20)]
    budgets.clear()
    SelfCollisionObj(model, margin=0.01, trajectory_checks=24)(Q, fk)
    assert budgets == [(10, 24)]


def test_trajectory_penalty_is_mean_over_frames(panda):
    solver, model = panda
    fk = solver.fk_solver
    Q, F, _ = _random_link_transforms(solver, model, 11, seed=9)
    obj = SelfCollisionObj(model, margin=0.01, weight=3.0, trajectory_checks=10 * model.n_entries)
    per_frame = [float(model.penalty(f, fk, 0.01, model.n_entries)) for f in F[1:]]
    assert max(per_frame) > 0
    assert float(obj(Q, fk)) == pytest.approx(3.0 * np.mean(per_frame), rel=1e-4)


def test_soft_penalty_zero_when_clear_positive_when_close(spheres):
    solver, model, _ = spheres
    fk = solver.fk_solver
    obj = SelfCollisionObj(model, margin=0.01)
    assert float(obj(jnp.asarray(_angles(3.0)), fk)) == 0.0
    assert float(obj(jnp.asarray(_angles(0.2)), fk)) > 0.0
    # Its gradient pushes the spheres apart (increases theta)
    g = jax.grad(lambda x: obj(x, fk))(jnp.asarray(_angles(0.5)))
    assert float(g[0]) < 0


def test_violation_only_inside_margin(spheres):
    solver, model, _ = spheres
    fk = solver.fk_solver
    obj = SelfCollisionObj(model, margin=0.02, hard=True, substeps=0)
    assert float(np.sum(obj.violation(jnp.asarray(_angles(3.0)), fk))) == 0.0
    assert float(np.sum(obj.violation(jnp.asarray(_angles(0.3)), fk))) > 0.0


def test_check_poses_skip_fixed_start(spheres):
    solver, model, _ = spheres
    obj = SelfCollisionObj(model, hard=True, substeps=2)
    X = jnp.asarray(np.array([[0.0, 0, 0], [3.0, 0, 0], [6.0, 0, 0]], np.float32))
    poses = np.asarray(obj.check_poses(X))
    # 2 free frames + 2 substeps per segment, frame 0 itself left out
    assert poses.shape == (2 + 2 * 2, 3)
    assert not np.any(np.all(poses == 0.0, axis=1))
    assert np.allclose(sorted(poses[:, 0]), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def test_objective_pytree_and_params(spheres):
    solver, model, _ = spheres
    obj = SelfCollisionObj(model, margin=0.01, hard=True)
    leaves, treedef = tree_flatten(obj)
    back = tree_unflatten(treedef, leaves)
    assert back.model is model and back.hard
    params = obj.get_params()
    assert params["model"] is model
    other = SelfCollisionObj(model)
    other.update_params(params)
    assert other.hard and float(other.margin) == pytest.approx(0.01)
    assert other.trajectory_checks is None
    other.update_params(SelfCollisionObj(model, trajectory_checks=48).get_params())
    assert other.trajectory_checks == 48
    assert obj.referenced_bones() == ("a", "b")
    assert float(SelfCollisionObj(model).weight) == 100.0
    assert float(SelfCollisionObj(model, hard=True).weight) == 1000.0


# --- solving -----------------------------------------------------------------------


def _reach_center_of_a(solver, obj, start):
    # Target b's center at a's center: only reachable by overlapping
    fk = solver.fk_solver
    a_pos = np.asarray(fk.get_bone_head_tail_from_fk(fk.compute_fk_from_angles(_angles(0.0)), "b")[0])
    result, _, _ = solver.solve(
        initial_rotations=_angles(start),
        learning_rate=0.05,
        mandatory_objective_functions=[DistanceObjTraj("b", [a_pos], use_head=True)],
        optional_objective_functions=[obj] if obj else [],
        patience=100,
        verbose=False,
    )
    return result


def test_hard_constraint_stops_at_the_margin(spheres):
    solver, model, _ = spheres
    fk = solver.fk_solver
    margin = 0.02
    free = _reach_center_of_a(solver, None, 2.0)
    assert float(model.clearances(free[-1], fk)[0]) < 0  # without it they overlap

    obj = SelfCollisionObj(model, margin=margin, hard=True)
    hard = _reach_center_of_a(solver, obj, 2.0)
    c = float(model.clearances(hard[-1], fk)[0])
    assert c >= margin - 1e-5
    # It still got close: the steering penalty only starts pushing within
    # `influence` of the margin
    assert c <= margin + float(obj.influence) + 0.005
    # And the motion from the start never cut through either
    poses = np.linspace(hard[0], hard[-1], 50)
    assert min(float(model.clearances(q, fk)[0]) for q in poses) >= margin - 1e-5


def test_soft_constraint_allows_slight_overlap_tradeoff(spheres):
    solver, model, _ = spheres
    fk = solver.fk_solver
    soft = _reach_center_of_a(solver, SelfCollisionObj(model, margin=0.0, weight=1.0), 2.0)
    stiff = _reach_center_of_a(solver, SelfCollisionObj(model, margin=0.0, weight=1000.0), 2.0)
    c_soft = float(model.clearances(soft[-1], fk)[0])
    c_stiff = float(model.clearances(stiff[-1], fk)[0])
    # A weak penalty gives way more than a strong one
    assert c_soft < c_stiff
    assert c_stiff > -0.01


def test_hard_constraint_from_colliding_start_does_not_get_worse(spheres):
    solver, model, _ = spheres
    obj = SelfCollisionObj(model, margin=0.0, hard=True, substeps=0)
    start = _angles(0.3)
    v0 = float(np.sum(obj.violation(jnp.asarray(np.stack([start, start])), solver.fk_solver)))
    result = _reach_center_of_a(solver, obj, 0.3)
    v1 = float(np.sum(obj.violation(jnp.asarray(result), solver.fk_solver)))
    assert v0 > 0 and v1 <= v0


def test_hard_trajectory_stays_clear_between_frames(spheres):
    solver, model, _ = spheres
    fk = solver.fk_solver
    obj = SelfCollisionObj(model, margin=0.01, hard=True)
    a_pos = np.asarray(fk.get_bone_head_tail_from_fk(fk.compute_fk_from_angles(_angles(-2.5)), "b")[0])
    # From theta=2.5 to -2.5: the short way passes straight through a
    traj, _, _ = solver.solve(
        initial_rotations=_angles(2.5),
        learning_rate=0.05,
        mandatory_objective_functions=[DistanceObjTraj("b", [a_pos], use_head=True)],
        optional_objective_functions=[obj],
        ik_points=6,
        patience=100,
        verbose=False,
    )
    dense = np.concatenate([np.linspace(traj[i], traj[i + 1], 20) for i in range(len(traj) - 1)])
    assert min(float(model.clearances(q, fk)[0]) for q in dense) >= 0.01 - 1e-5


def test_panda_hard_solve_never_collides(panda):
    solver, model = panda
    fk = solver.fk_solver
    lo, hi = np.asarray(solver.lower_bounds), np.asarray(solver.upper_bounds)
    mid = ((lo + hi) / 2).astype(np.float32)
    clear = jax.jit(lambda q: model.clearances(q, fk))
    assert float(clear(mid).min()) >= 0.005  # a collision-free start
    ee = lambda q: np.asarray(fk.get_bone_head_tail_from_fk(fk.compute_fk_from_angles(jnp.asarray(q)), "panda_hand_tcp")[0])
    rng = np.random.default_rng(3)
    targets = []
    while len(targets) < 3:  # hand positions of self-colliding poses
        q = rng.uniform(lo, hi).astype(np.float32)
        if float(clear(q).min()) < -0.01:
            targets.append(ee(q))
    obj = SelfCollisionObj(model, margin=0.005, hard=True)
    for target in targets:
        result, _, _ = solver.solve(
            initial_rotations=mid,
            learning_rate=0.05,
            mandatory_objective_functions=[DistanceObjTraj("panda_hand_tcp", [target], True)],
            optional_objective_functions=[obj],
            patience=50,
            verbose=False,
        )
        assert float(np.sum(obj.violation(jnp.asarray(result), fk))) == 0.0
        assert float(clear(result[-1]).min()) >= 0.005 - 1e-5


# --- reach map -----------------------------------------------------------------


@pytest.fixture(scope="module")
def panda_reach(panda):
    from jax_ik.visualizer import CollisionModes, ReachMap

    solver, model = panda
    obstacles = dict(floor=-1.0, spheres=[((0.35, -0.55, 0.35), 0.1)], ignore_links=("panda_link0",))
    modes = CollisionModes(model, margin=0.005, mode="hard", obstacles=obstacles)
    reach = ReachMap(solver, ["panda_hand_tcp"], collision=modes, samples=4096, seed=0)
    return solver, model, modes, reach


def test_reach_map_keeps_only_collision_free_poses(panda_reach):
    solver, model, modes, reach = panda_reach
    fk = solver.fk_solver
    clear = jax.jit(lambda q: model.clearances(q, fk).min())
    obstacle = jax.jit(lambda q: jnp.min(modes.obstacle.clearances(q, fk, checks=4096), axis=1))
    free = np.nonzero(reach.free())[0]
    assert 0 < len(free) < len(reach.poses)
    for i in free[:: max(1, len(free) // 40)]:
        assert float(clear(reach.poses[i])) >= 0.005 - 1e-5
        assert float(obstacle(reach.poses[i]).min()) >= 0.005 + reach.clearance - 1e-5
    # Some dropped poses really collide with the arm itself
    assert (reach.self_clear < 0).any()


def test_reach_map_follows_obstacle_switches(panda_reach):
    _, _, modes, reach = panda_reach
    both = reach.free()
    sphere_blocked = reach.obstacle_clear[:, 1] < 0.005 + reach.clearance
    assert not (both & sphere_blocked).any()
    modes.toggle_obstacle(1)  # sphere off
    try:
        without = reach.free()
        assert (without & sphere_blocked).any()
        assert np.array_equal(without & ~sphere_blocked, both)
    finally:
        modes.toggle_obstacle(1)
    assert np.array_equal(reach.free(), both)


def test_reach_map_guess_never_runs_through_a_collision(panda_reach):
    from jax_ik.visualizer import TrajectoryPlanner

    solver, _, modes, reach = panda_reach
    mid = (np.asarray(solver.lower_bounds) + np.asarray(solver.upper_bounds)) / 2
    planner = TrajectoryPlanner(
        solver, ["panda_hand_tcp"], initial_angles=mid, seed=1, reach_map=reach
    )
    planner.extra_objectives = modes.current
    found = 0
    for _ in range(10):
        targets = planner.random_targets()
        guess = planner.initial_guess(planner.pose, targets)
        if guess is None:
            continue
        found += 1
        X = jnp.asarray(guess)
        for obj in modes.current:
            assert float(jnp.max(obj.violation(X, solver.fk_solver))) <= 0.0
    assert found > 0
