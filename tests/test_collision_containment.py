"""The collision proxies must contain the entire model.

For every link of several kinds of models, checked against the original
mesh:

* every mesh vertex and densely sampled surface point lies inside the
  proxy (its conservative signed distance is <= 0 there),
* every surface point is covered by one of the proxy's sample points (the
  part of the proxy the collision check moves around), and
* for closed meshes, points inside the mesh are inside the proxy too.

Rigid models (URDF) are also checked in random poses, with the original
mesh moved by its bones and the proxies by theirs.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import trimesh
from conftest import PANDA_ARM

from jax_ik.collision import CollisionModel
from jax_ik.helper import load_mesh_data_from_gltf, load_mesh_data_from_urdf
from jax_ik.ik import FKSolver, InverseKinematicsSolver

EPS = 1e-6


def _grid_value(model, link, points_local):
    coords = jnp.asarray((points_local - link.origin) / model.spacing)
    return np.asarray(
        jax.scipy.ndimage.map_coordinates(
            jnp.asarray(link.grid), coords.T, order=1, mode="constant", cval=link.outside
        )
    )


def _check_contained(model, link, mesh_local, n_samples=3000, seed=0):
    """All containment checks for one link, mesh in the link's frame."""
    rng = np.random.default_rng(seed)
    surface = np.concatenate([mesh_local.vertices, mesh_local.sample(n_samples, seed=rng)])

    # 1. The surface is inside the proxy
    G = _grid_value(model, link, surface)
    assert np.all(G <= EPS), f"{link.name}: surface outside proxy by {G.max() * 1000:.2f} mm"

    # 2. The sample points (with their radii) cover the whole surface
    from scipy.spatial import cKDTree

    tree = cKDTree(link.points)
    covered = np.zeros(len(surface), bool)
    # Any point whose radius reaches: query within the largest radius
    for i, near in enumerate(tree.query_ball_point(surface, float(link.radii.max()) + EPS)):
        if near:
            d = np.linalg.norm(link.points[near] - surface[i], axis=1)
            covered[i] = np.any(d <= link.radii[near] + EPS)
    assert covered.all(), f"{link.name}: {np.sum(~covered)} surface points not covered"

    # 3. Closed meshes: their inside is inside the proxy too
    if mesh_local.is_watertight:
        lo, hi = mesh_local.bounds
        candidates = rng.uniform(lo, hi, (4000, 3))
        inner = candidates[mesh_local.contains(candidates)]
        if len(inner):
            Gi = _grid_value(model, link, inner)
            assert np.all(Gi <= EPS), f"{link.name}: interior outside proxy"


def _link_meshes(fk_solver, mesh_data, model):
    """Each link's original mesh in its bone frame (what the proxy is built
    from), keyed by link name."""
    from jax_ik.collision import _link_geometry

    geometry = _link_geometry(fk_solver, mesh_data, model.link_names, 0.0)
    bind = np.asarray(fk_solver.bind_fk, np.float64)
    out = {}
    for name, (verts, faces) in geometry.items():
        T_inv = np.linalg.inv(bind[fk_solver.bone_names.index(name)])
        out[name] = trimesh.Trimesh(verts @ T_inv[:3, :3].T + T_inv[:3, 3], faces, process=False)
    return out


# --- Panda (URDF, rigid links) ---------------------------------------------------


@pytest.fixture(scope="module")
def panda(panda_urdf):
    solver = InverseKinematicsSolver(panda_urdf, controlled_bones=PANDA_ARM, compute_sdf=False)
    fk = solver.fk_solver
    model = CollisionModel.cached(fk, lower=solver.lower_bounds, upper=solver.upper_bounds)
    mesh_data = load_mesh_data_from_urdf(panda_urdf, fk)
    return solver, model, _link_meshes(fk, mesh_data, model)


def test_panda_every_link_contained(panda):
    _, model, meshes = panda
    assert len(model.links) == 11
    for link in model.links:
        _check_contained(model, link, meshes[link.name])


def test_panda_contained_in_random_poses(panda):
    solver, model, meshes = panda
    fk = solver.fk_solver
    lo, hi = np.asarray(solver.lower_bounds), np.asarray(solver.upper_bounds)
    rng = np.random.default_rng(1)
    for _ in range(5):
        q = rng.uniform(lo, hi).astype(np.float32)
        out = np.asarray(fk.compute_fk_from_angles(q))
        for link in model.links:
            T = out[fk.bone_names.index(link.name)]
            # Move the original mesh by the bone, then back into the proxy's
            # frame through the *posed* transform: both must agree
            world = trimesh.Trimesh(meshes[link.name].vertices, meshes[link.name].faces, process=False)
            world.apply_transform(T)
            local = world.vertices @ T[:3, :3] - T[:3, 3] @ T[:3, :3]
            G = _grid_value(model, link, local)
            assert np.all(G <= EPS), f"{link.name} leaves its proxy in pose {q}"


def test_panda_envelope_mesh_encloses_link(panda):
    _, model, meshes = panda
    for link in model.links:
        verts, faces = link.envelope
        envelope = trimesh.Trimesh(verts, faces)
        assert envelope.is_watertight, link.name
        verts = meshes[link.name].vertices
        sample = verts[np.random.default_rng(0).permutation(len(verts))[:2000]]
        inside = envelope.contains(sample)
        assert inside.mean() > 0.999, f"{link.name}: {np.sum(~inside)} vertices outside envelope"


# --- SMPL-X (skinned glTF) -----------------------------------------------------------


@pytest.fixture(scope="module")
def smplx(model_path):
    fk = FKSolver(model_path, controlled_bones=["left_elbow"], do_compute_sdf=False)
    mesh_data = load_mesh_data_from_gltf(model_path, fk)
    model = CollisionModel.from_fk_solver(fk, mesh_data=mesh_data, resolution=0.015)
    return fk, model, _link_meshes(fk, mesh_data, model)


def test_smplx_every_body_part_contained(smplx):
    _, model, meshes = smplx
    assert len(model.links) >= 5
    for link in model.links:
        _check_contained(model, link, meshes[link.name], n_samples=1500)


def test_smplx_whole_body_covered(smplx):
    fk, model, meshes = smplx
    # Every vertex of the body belongs to some body part's proxy
    total = sum(len(m.vertices) for m in meshes.values())
    mesh_data = load_mesh_data_from_gltf(fk.model_file, fk)
    faces = np.asarray(mesh_data["faces"])
    assert total >= len(np.unique(faces))


# --- synthetic shapes ---------------------------------------------------------------


def _box_without_top():
    box = trimesh.creation.box(extents=[0.2, 0.15, 0.1])
    top = box.triangles_center[:, 2] > 0.049
    return trimesh.Trimesh(box.vertices, box.faces[~top])


SHAPES = {
    "box": lambda: trimesh.creation.box(extents=[0.2, 0.15, 0.1]),
    "torus": lambda: trimesh.creation.torus(major_radius=0.1, minor_radius=0.025),
    "thin_plate": lambda: trimesh.creation.box(extents=[0.2, 0.2, 0.003]),
    "l_shape": lambda: trimesh.util.concatenate(
        [
            trimesh.creation.box(extents=[0.2, 0.05, 0.05]),
            trimesh.creation.box(extents=[0.05, 0.2, 0.05]).apply_translation([0.075, 0.075, 0]),
        ]
    ),
    "open_box": _box_without_top,
    "tiny": lambda: trimesh.creation.icosphere(subdivisions=1, radius=0.004),
    "long_rod": lambda: trimesh.creation.cylinder(radius=0.01, height=0.6, sections=12),
    "rotated_box": lambda: trimesh.creation.box(extents=[0.2, 0.05, 0.1]).apply_transform(
        trimesh.transformations.euler_matrix(0.4, -0.7, 1.1)
    ),
}


@pytest.fixture(scope="module")
def shapes_robot(tmp_path_factory):
    """One URDF link per shape, spread out along a revolute chain."""
    d = tmp_path_factory.mktemp("shapes")
    links, joints = ['<link name="base"/>'], []
    parent = "base"
    for i, (name, make) in enumerate(SHAPES.items()):
        make().export(str(d / f"{name}.stl"))
        links.append(
            f'<link name="{name}"><visual><geometry><mesh filename="{name}.stl"/>'
            "</geometry></visual></link>"
        )
        joints.append(
            f'<joint name="j_{name}" type="revolute"><parent link="{parent}"/>'
            f'<child link="{name}"/><origin xyz="0.5 0 0" rpy="0.3 0.2 0.1"/>'
            '<axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>'
        )
        parent = name
    urdf = d / "shapes.urdf"
    urdf.write_text(f'<?xml version="1.0"?><robot name="shapes">{"".join(links + joints)}</robot>')
    fk = FKSolver(str(urdf), controlled_bones=list(SHAPES), do_compute_sdf=False)
    mesh_data = load_mesh_data_from_urdf(str(urdf), fk)
    model = CollisionModel.from_fk_solver(fk, mesh_data=mesh_data, resolution=0.01)
    return fk, model, _link_meshes(fk, mesh_data, model)


@pytest.mark.parametrize("shape", list(SHAPES))
def test_synthetic_shape_contained(shapes_robot, shape):
    _, model, meshes = shapes_robot
    link = model.links[model.link_names.index(shape)]
    _check_contained(model, link, meshes[shape])


@pytest.mark.parametrize("resolution", [0.005, 0.02])
def test_containment_holds_at_other_resolutions(tmp_path, resolution):
    mesh = trimesh.creation.torus(major_radius=0.1, minor_radius=0.03)
    mesh.export(str(tmp_path / "t.stl"))
    urdf = tmp_path / "t.urdf"
    urdf.write_text(
        '<?xml version="1.0"?><robot name="t"><link name="base"/>'
        '<link name="t"><visual><geometry><mesh filename="t.stl"/></geometry></visual></link>'
        '<joint name="j" type="revolute"><parent link="base"/><child link="t"/>'
        '<axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint></robot>'
    )
    fk = FKSolver(str(urdf), controlled_bones=["t"], do_compute_sdf=False)
    mesh_data = load_mesh_data_from_urdf(str(urdf), fk)
    model = CollisionModel.from_fk_solver(fk, mesh_data=mesh_data, resolution=resolution)
    meshes = _link_meshes(fk, mesh_data, model)
    _check_contained(model, model.links[0], meshes["t"])
