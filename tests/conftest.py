import os
import sys
import urllib.request

import numpy as np
import pytest

# Ensure local src/ is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from jax_ik.smplx_statics import left_arm_bounds_dict  # added

SMPLX_FILENAME = "smplx.glb"
SMPLX_URL = "https://uni-bielefeld.sciebo.de/s/B5StwQdiR4DW5mc/download"


def _ensure_smplx_model(path: str):
    """Ensure smplx.glb exists; attempt download if missing.
    Skips tests gracefully if download fails.
    Set env JAX_IK_SKIP_DOWNLOAD=1 to disable auto download.
    """
    if os.path.exists(path):
        return
    if os.environ.get("JAX_IK_SKIP_DOWNLOAD", "0") == "1":
        pytest.skip(f"{SMPLX_FILENAME} missing and auto-download disabled")
    try:
        print(f"Attempting download of {SMPLX_FILENAME} (~may take a moment)...")
        with urllib.request.urlopen(SMPLX_URL, timeout=60) as resp:  # nosec B310
            data = resp.read()
        if not data:
            raise RuntimeError("empty response")
        with open(path, "wb") as f:
            f.write(data)
        print(f"Downloaded {SMPLX_FILENAME} ({len(data)} bytes)")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"{SMPLX_FILENAME} not available and download failed: {e}")


# Central model path fixture
@pytest.fixture(scope="session")
def model_path():
    path = os.path.abspath(os.path.join(PROJECT_ROOT, SMPLX_FILENAME))
    _ensure_smplx_model(path)
    if not os.path.exists(path):
        pytest.skip(
            f"{SMPLX_FILENAME} not available – skipping IK tests that need real model"
        )
    return path


@pytest.fixture(scope="session")
def basic_controlled_bones():
    return ["left_collar", "left_shoulder", "left_elbow", "left_wrist"]


@pytest.fixture(scope="session")
def angle_dim(basic_controlled_bones):
    return len(basic_controlled_bones) * 3


def _build_bounds(controlled_bones):
    bounds = []
    for bone in controlled_bones:
        if bone not in left_arm_bounds_dict:
            raise KeyError(f"Bone {bone} not found in left_arm_bounds_dict")
        lower, upper = left_arm_bounds_dict[bone]
        for l, u in zip(lower, upper):
            bounds.append((l, u))  # degrees; solver converts to radians
    return bounds


@pytest.fixture(scope="session")
def solver(model_path, basic_controlled_bones):
    try:
        from jax_ik.ik import InverseKinematicsSolver
    except Exception as e:  # noqa: BLE001
        pytest.skip(
            f"Cannot import InverseKinematicsSolver (maybe optional dep missing like pyvista): {e}"
        )
    bounds = _build_bounds(basic_controlled_bones)
    solver = InverseKinematicsSolver(
        model_file=model_path,
        controlled_bones=basic_controlled_bones,
        bounds=bounds,
        threshold=1e-6,
        num_steps=60,
        compute_sdf=False,
    )
    return solver


@pytest.fixture()
def zero_angles(angle_dim):
    return np.zeros(angle_dim, dtype=np.float32)


@pytest.fixture()
def two_frame_traj(zero_angles):
    return np.stack([zero_angles, zero_angles.copy()], axis=0)


PANDA_URDF = os.path.join(PROJECT_ROOT, "models", "panda", "urdf", "panda.urdf")
PANDA_ARM = [f"panda_link{i}" for i in range(1, 8)]

# Small 3-joint chain (revolute z, revolute y, prismatic x) with no meshes:
# fast to build and fully predictable
TINY_URDF = """<?xml version="1.0"?>
<robot name="tiny">
  <link name="base"/>
  <link name="l1"/>
  <link name="l2"/>
  <link name="slider"/>
  <link name="tip"/>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0.1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.5" upper="1.5" effort="1" velocity="1"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="l1"/><child link="l2"/>
    <origin xyz="0 0 0.5"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="0.5" effort="1" velocity="1"/>
  </joint>
  <joint name="j3" type="prismatic">
    <parent link="l2"/><child link="slider"/>
    <origin xyz="0 0 0.4"/>
    <axis xyz="1 0 0"/>
    <limit lower="0.0" upper="0.2" effort="1" velocity="1"/>
  </joint>
  <joint name="j4" type="fixed">
    <parent link="slider"/><child link="tip"/>
    <origin xyz="0 0 0.1"/>
  </joint>
</robot>
"""


@pytest.fixture(scope="session")
def panda_urdf():
    if not os.path.exists(PANDA_URDF):
        pytest.skip("Panda URDF not available")
    return PANDA_URDF


@pytest.fixture(scope="session")
def tiny_urdf(tmp_path_factory):
    path = tmp_path_factory.mktemp("urdf") / "tiny.urdf"
    path.write_text(TINY_URDF)
    return str(path)


class StubFK:
    """Minimal stand-in for FKSolver with exactly known geometry.

    Bones: "root" (identity) and "b" (child of root). The first 3 angles
    translate "b", the next 3 (if present) rotate it as XYZ Euler. The tail
    sits bone_length along the bone's local +y.
    """

    def __init__(self, bone_length=1.0):
        self.bone_names = ["root", "b"]
        self.parent_list = [-1, 0]
        self.skeleton = {
            "root": {"bone_length": bone_length},
            "b": {"bone_length": bone_length},
        }
        self.bone_length = bone_length
        self.sdf = None
        self.mesh_data = None

    def compute_fk_from_angles(self, cfg):
        import jax.numpy as jnp

        from jax_ik.ik import tf_euler_to_matrix

        cfg = jnp.asarray(cfg, jnp.float32)
        T = jnp.eye(4, dtype=jnp.float32)
        if cfg.shape[0] >= 6:
            T = tf_euler_to_matrix(cfg[3:6])
        T = T.at[:3, 3].set(cfg[:3])
        return jnp.stack([jnp.eye(4, dtype=jnp.float32), T])

    def get_bone_head_tail_from_fk(self, fk, bone_name):
        import jax.numpy as jnp

        g = fk[self.bone_names.index(bone_name)]
        tail = g @ jnp.asarray([0.0, self.bone_length, 0.0, 1.0], jnp.float32)
        return g[:3, 3], tail[:3]


@pytest.fixture()
def stub_fk():
    return StubFK()
