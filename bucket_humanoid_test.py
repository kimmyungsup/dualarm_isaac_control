"""
Isaac Sim 5.1 upper-body humanoid URDF task-space keyboard test.

This script imports ``humanoid_urdf_assemble`` into Isaac Sim, builds a light-weight
URDF kinematics model, and drives either arm in task space from the keyboard.  It
was written to avoid external Lula descriptor files: joint chains, DOF indices,
base/world transforms, and left/right end-effector links are resolved from the
URDF and the loaded Isaac articulation at runtime.

Run from this repository with Isaac Sim Python, for example:
    ./python.sh /workspace/dualarm_isaac_control/bucket_humanoid_test.py
"""

from __future__ import annotations

import math
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
SOURCE_URDF_PATH = REPO_ROOT / "humanoid_urdf_assemble" / "urdf" / "humanoid_urdf_assemble.urdf"
ISAAC_READY_URDF_PATH = REPO_ROOT / "humanoid_urdf_assemble" / "urdf" / "humanoid_urdf_assemble_isaac_ready.urdf"
MESH_DIR = REPO_ROOT / "humanoid_urdf_assemble" / "meshes"
ROBOT_PRIM_PATH = "/World/bucket_humanoid"
ROOT_LINK_NAME = "base_link"

HEADLESS = False
FIX_BASE = True
DISABLE_GRAVITY = True
ADD_GROUND_PLANE = True

KP = 800.0
KD = 80.0
HOLD_SECONDS = 0.5
DEFAULT_DT = 1.0 / 60.0
POSITION_STEP_MPS = 0.12
ROTATION_STEP_RPS = 0.8
MAX_TARGET_OFFSET_FROM_START = np.array([0.55, 0.55, 0.55], dtype=np.float64)
IK_DAMPING = 0.06
IK_MAX_STEP_RAD = 0.08
IK_POS_GAIN = 4.0
IK_ORI_GAIN = 2.0
IK_MAX_LINEAR_ERROR = 0.12
IK_MAX_ANGULAR_ERROR = 0.45

# The SolidWorks-exported humanoid URDF in this repository has zero-width joint
# limits and references meshes that are not present in the folder.  The runtime
# Isaac-ready copy expands the limits and redirects missing meshes to one of the
# placeholder STL files below so import and kinematic tests can proceed.
DEFAULT_LOWER_LIMIT = -math.pi
DEFAULT_UPPER_LIMIT = math.pi
DEFAULT_EFFORT = 250.0
DEFAULT_VELOCITY = 2.0
FALLBACK_MESHES = ("base_link.STL", "link1_L.STL", "link1_R.STL")


# -----------------------------------------------------------------------------
# Math helpers
# -----------------------------------------------------------------------------
def unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return np.zeros_like(v, dtype=np.float64) if n < eps else np.asarray(v, dtype=np.float64) / n


def skew(v: Sequence[float]) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=np.float64)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def rot_from_axis_angle(axis: Sequence[float], angle: float) -> np.ndarray:
    a = unit(np.asarray(axis, dtype=np.float64))
    K = skew(a)
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def rpy_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def xyz_rpy_to_tf(xyz: Sequence[float], rpy: Sequence[float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rpy_to_rot(float(rpy[0]), float(rpy[1]), float(rpy[2]))
    T[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return T


def quat_wxyz_to_rot(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = unit(np.asarray(q, dtype=np.float64))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_xyzw_to_wxyz(q: Sequence[float]) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def rot_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    tr = float(np.trace(R))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    return unit(q)


def quat_mul(q1: Sequence[float], q2: Sequence[float]) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(q1, dtype=np.float64)
    w2, x2, y2, z2 = np.asarray(q2, dtype=np.float64)
    return unit(
        np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float64,
        )
    )


def quat_from_axis_angle(axis: Sequence[float], angle: float) -> np.ndarray:
    a = unit(np.asarray(axis, dtype=np.float64))
    half = 0.5 * float(angle)
    return unit(np.array([math.cos(half), *(math.sin(half) * a)], dtype=np.float64))


def orientation_error_vector(R_current: np.ndarray, R_target: np.ndarray) -> np.ndarray:
    R_err = R_target @ R_current.T
    return 0.5 * np.array(
        [R_err[2, 1] - R_err[1, 2], R_err[0, 2] - R_err[2, 0], R_err[1, 0] - R_err[0, 1]],
        dtype=np.float64,
    )


def clamp_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n <= max_norm or n < 1e-12 else v * (max_norm / n)


def parse_vec(text: Optional[str], default: Sequence[float]) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=np.float64)
    return np.array([float(x) for x in text.split()], dtype=np.float64)


# -----------------------------------------------------------------------------
# URDF preparation and kinematics
# -----------------------------------------------------------------------------
@dataclass
class JointInfo:
    name: str
    parent: str
    child: str
    axis: np.ndarray
    origin_T: np.ndarray
    lower: float
    upper: float


@dataclass
class ArmModel:
    name: str
    joint_names: List[str]
    ee_link: str
    joints: List[JointInfo]
    joint_indices: List[int]
    lower: np.ndarray
    upper: np.ndarray
    target_pos_world: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    target_quat_world: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
    start_target_pos_world: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))


def ensure_isaac_ready_urdf() -> Path:
    if not SOURCE_URDF_PATH.exists():
        raise FileNotFoundError(f"URDF not found: {SOURCE_URDF_PATH}")

    tree = ET.parse(SOURCE_URDF_PATH)
    root = tree.getroot()

    # SolidWorks URDFs often use package:// mesh paths.  Isaac's URDF importer is
    # much more reliable when every mesh path is converted to an absolute local
    # path.  Also handle .STL/.stl case differences and only require a fallback
    # mesh when the URDF actually references a missing mesh.
    mesh_files = [p for p in MESH_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".stl"] if MESH_DIR.exists() else []
    mesh_lookup = {p.name.lower(): p for p in mesh_files}
    fallback_path = next((mesh_lookup.get(name.lower()) for name in FALLBACK_MESHES if name.lower() in mesh_lookup), None)
    if fallback_path is None and mesh_files:
        fallback_path = mesh_files[0]

    mesh_redirects = 0
    missing_meshes = []
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename", "")
        mesh_name = Path(filename).name
        mesh_path = mesh_lookup.get(mesh_name.lower())
        if mesh_path is None:
            missing_meshes.append(mesh_name or filename)
            if fallback_path is None:
                raise FileNotFoundError(f"Mesh '{mesh_name}' not found and no STL fallback exists in {MESH_DIR}")
            mesh.attrib["filename"] = str(fallback_path.resolve())
            mesh_redirects += 1
        else:
            mesh.attrib["filename"] = str(mesh_path.resolve())

    widened_limits = 0
    for joint in root.findall("joint"):
        if joint.attrib.get("type") not in {"revolute", "continuous", "prismatic"}:
            continue
        limit = joint.find("limit")
        if limit is None:
            limit = ET.SubElement(joint, "limit")
        lower = float(limit.attrib.get("lower", "0"))
        upper = float(limit.attrib.get("upper", "0"))
        effort = float(limit.attrib.get("effort", "0"))
        velocity = float(limit.attrib.get("velocity", "0"))
        if upper <= lower:
            limit.attrib["lower"] = str(DEFAULT_LOWER_LIMIT)
            limit.attrib["upper"] = str(DEFAULT_UPPER_LIMIT)
            widened_limits += 1
        if effort <= 0:
            limit.attrib["effort"] = str(DEFAULT_EFFORT)
        if velocity <= 0:
            limit.attrib["velocity"] = str(DEFAULT_VELOCITY)

    tree.write(ISAAC_READY_URDF_PATH, encoding="utf-8", xml_declaration=True)
    print(
        f"[URDF] wrote Isaac-ready copy: {ISAAC_READY_URDF_PATH}\n"
        f"       widened_limits={widened_limits}, redirected_missing_meshes={mesh_redirects}"
    )
    if missing_meshes:
        print(f"[URDF] missing mesh references redirected to {fallback_path.name}: {sorted(set(missing_meshes))}")
    return ISAAC_READY_URDF_PATH


def load_joints_from_urdf(urdf_path: Path) -> Dict[str, JointInfo]:
    root = ET.parse(urdf_path).getroot()
    joints: Dict[str, JointInfo] = {}
    for elem in root.findall("joint"):
        if elem.attrib.get("type") not in {"revolute", "continuous"}:
            continue
        name = elem.attrib["name"]
        parent = elem.find("parent").attrib["link"]
        child = elem.find("child").attrib["link"]
        origin = elem.find("origin")
        xyz = parse_vec(origin.attrib.get("xyz") if origin is not None else None, [0, 0, 0])
        rpy = parse_vec(origin.attrib.get("rpy") if origin is not None else None, [0, 0, 0])
        axis_elem = elem.find("axis")
        axis = unit(parse_vec(axis_elem.attrib.get("xyz") if axis_elem is not None else None, [1, 0, 0]))
        limit = elem.find("limit")
        lower = float(limit.attrib.get("lower", DEFAULT_LOWER_LIMIT)) if limit is not None else DEFAULT_LOWER_LIMIT
        upper = float(limit.attrib.get("upper", DEFAULT_UPPER_LIMIT)) if limit is not None else DEFAULT_UPPER_LIMIT
        joints[name] = JointInfo(name, parent, child, axis, xyz_rpy_to_tf(xyz, rpy), lower, upper)
    return joints


def discover_arm_chains(joints: Dict[str, JointInfo], dof_names: Sequence[str]) -> Dict[str, Tuple[List[str], str]]:
    dof_set = set(dof_names)
    child_to_joint = {j.child: j for j in joints.values()}
    terminal_links = sorted({j.child for j in joints.values()} - {j.parent for j in joints.values()})
    chains: Dict[str, Tuple[List[str], str]] = {}
    for terminal in terminal_links:
        names: List[str] = []
        link = terminal
        while link in child_to_joint:
            j = child_to_joint[link]
            names.append(j.name)
            link = j.parent
        names.reverse()
        if not names or not all(name in dof_set for name in names):
            continue
        suffix = names[-1].split("_")[-1].lower()
        if suffix in {"l", "left"}:
            chains["left"] = (names, terminal)
        elif suffix in {"r", "right"}:
            chains["right"] = (names, terminal)
        else:
            chains[f"chain_{len(chains) + 1}"] = (names, terminal)

    if "left" not in chains or "right" not in chains:
        grouped: Dict[str, List[str]] = {"left": [], "right": []}
        for name in dof_names:
            lower = name.lower()
            if lower.endswith("_l") or "left" in lower:
                grouped["left"].append(name)
            elif lower.endswith("_r") or "right" in lower:
                grouped["right"].append(name)
        for arm_name, names in grouped.items():
            if arm_name not in chains and names:
                ee = joints[names[-1]].child
                chains[arm_name] = (names, ee)

    if not chains:
        usable = [name for name in dof_names if name in joints]
        if len(usable) >= 6:
            chains["arm"] = (usable[:6], joints[usable[5]].child)
    return chains


def fk_and_jacobian(arm: ArmModel, q_arm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    T = np.eye(4, dtype=np.float64)
    origins: List[np.ndarray] = []
    axes_world: List[np.ndarray] = []
    for joint, q in zip(arm.joints, q_arm):
        T = T @ joint.origin_T
        origins.append(T[:3, 3].copy())
        axes_world.append(T[:3, :3] @ joint.axis)
        T[:3, :3] = T[:3, :3] @ rot_from_axis_angle(joint.axis, float(q))
    p_ee = T[:3, 3]
    J = np.zeros((6, len(arm.joints)), dtype=np.float64)
    for i, (origin, axis) in enumerate(zip(origins, axes_world)):
        J[:3, i] = np.cross(axis, p_ee - origin)
        J[3:, i] = axis
    return T, J


def solve_arm_ik_step(arm: ArmModel, q_arm: np.ndarray, base_T_world: np.ndarray) -> Tuple[np.ndarray, float, float]:
    T_base_ee, J_base = fk_and_jacobian(arm, q_arm)
    R_world_base = base_T_world[:3, :3]
    p_world = base_T_world[:3, 3] + R_world_base @ T_base_ee[:3, 3]
    R_world_ee = R_world_base @ T_base_ee[:3, :3]
    R_target = quat_wxyz_to_rot(arm.target_quat_world)

    pos_err_world = clamp_norm(arm.target_pos_world - p_world, IK_MAX_LINEAR_ERROR)
    ori_err_world = clamp_norm(orientation_error_vector(R_world_ee, R_target), IK_MAX_ANGULAR_ERROR)
    err_world = np.concatenate([IK_POS_GAIN * pos_err_world, IK_ORI_GAIN * ori_err_world])

    J_world = J_base.copy()
    J_world[:3, :] = R_world_base @ J_base[:3, :]
    J_world[3:, :] = R_world_base @ J_base[3:, :]
    JJt = J_world @ J_world.T
    dq = J_world.T @ np.linalg.solve(JJt + (IK_DAMPING * IK_DAMPING) * np.eye(6), err_world)
    dq = clamp_norm(dq, IK_MAX_STEP_RAD)
    q_next = np.clip(q_arm + dq, arm.lower, arm.upper)
    return q_next, float(np.linalg.norm(pos_err_world)), float(np.linalg.norm(ori_err_world))


# -----------------------------------------------------------------------------
# Isaac Sim API helpers
# -----------------------------------------------------------------------------
def get_dof_names(articulation) -> List[str]:
    for attr in ("dof_names", "joint_names"):
        value = getattr(articulation, attr, None)
        if value is not None:
            return list(value)
    for fn_name in ("get_dof_names", "get_joint_names"):
        fn = getattr(articulation, fn_name, None)
        if callable(fn):
            try:
                return list(fn())
            except Exception:
                pass
    raise RuntimeError("Could not read DOF/joint names from Isaac articulation")


def import_urdf_to_stage(urdf_path: Path, prim_path: str) -> str:
    import omni.kit.commands
    import omni.usd
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    try:
        import omni.kit.app

        ext_mgr = omni.kit.app.get_app().get_extension_manager()
        for ext_name in ("isaacsim.asset.importer.urdf", "omni.importer.urdf"):
            try:
                ext_mgr.set_extension_enabled_immediate(ext_name, True)
            except Exception:
                pass
    except Exception:
        pass

    stage = omni.usd.get_context().get_stage()
    root_prim = stage.DefinePrim(Sdf.Path(prim_path), "Xform")
    UsdGeom.Xformable(root_prim).AddTranslateOp().Set((0.0, 0.0, 1.0))

    import_config = None
    try:
        from isaacsim.asset.importer.urdf import _urdf

        import_config = _urdf.ImportConfig()
    except Exception:
        try:
            from omni.importer.urdf import _urdf

            import_config = _urdf.ImportConfig()
        except Exception:
            import_config = None

    if import_config is not None:
        for attr, value in {
            "merge_fixed_joints": False,
            "fix_base": FIX_BASE,
            "make_default_prim": False,
            "self_collision": False,
            "import_inertia_tensor": True,
            "distance_scale": 1.0,
            "density": 0.0,
        }.items():
            try:
                setattr(import_config, attr, value)
            except Exception:
                pass

    command_attempts = [
        ("URDFParseAndImportFile", {"urdf_path": str(urdf_path), "import_config": import_config, "dest_path": prim_path}),
        ("URDFParseAndImportFile", {"urdf_path": str(urdf_path), "import_config": import_config}),
    ]
    last_error: Optional[Exception] = None
    for command, kwargs in command_attempts:
        try:
            clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
            result = omni.kit.commands.execute(command, **clean_kwargs)
            print(f"[URDF] {command} result={result}")
            stage = omni.usd.get_context().get_stage()
            root = stage.GetPrimAtPath(prim_path)
            candidates = []
            if root and root.IsValid():
                candidates = [str(prim.GetPath()) for prim in Usd.PrimRange(root) if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
            if not candidates:
                candidates = [str(prim.GetPath()) for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
            articulation_path = candidates[0] if candidates else prim_path
            print(f"[URDF] articulation_path={articulation_path}")
            return articulation_path
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"URDF import failed for {urdf_path}: {last_error}")


def disable_robot_gravity(world, robot_prim_path: str) -> None:
    from pxr import Usd, UsdPhysics

    try:
        from isaacsim.core.prims import RigidPrimView
    except Exception:
        from omni.isaac.core.prims import RigidPrimView

    root = world.stage.GetPrimAtPath(robot_prim_path)
    rigid_paths = [str(prim.GetPath()) for prim in Usd.PrimRange(root) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    if not rigid_paths:
        print(f"[WARN] No rigid bodies found under {robot_prim_path}; gravity disable skipped")
        return
    views = []
    for i, path in enumerate(rigid_paths):
        view = RigidPrimView(prim_paths_expr=path, name=f"bucket_humanoid_rigid_{i}", reset_xform_properties=False)
        world.scene.add(view)
        views.append(view)
    world.reset()
    for view in views:
        view.disable_gravities()
    print(f"[INFO] Gravity disabled for {len(rigid_paths)} rigid bodies")


def resolve_base_transform(robot, verbose: bool = False) -> Tuple[np.ndarray, str]:
    pos, quat_raw = robot.get_world_pose()
    pos = np.asarray(pos, dtype=np.float64)
    quat_raw = np.asarray(quat_raw, dtype=np.float64)
    candidates = {
        "as_returned_wxyz": quat_raw,
        "xyzw_to_wxyz": quat_xyzw_to_wxyz(quat_raw),
    }
    best_name = "as_returned_wxyz"
    best_q = candidates[best_name]
    # Isaac Core normally returns WXYZ.  Keep the returned ordering unless it is
    # obviously invalid; print both so coordinate-frame issues are visible.
    if not np.isfinite(best_q).all() or abs(np.linalg.norm(best_q) - 1.0) > 0.25:
        best_name = "xyzw_to_wxyz"
        best_q = candidates[best_name]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_wxyz_to_rot(best_q)
    T[:3, 3] = pos
    if verbose:
        print(f"[FRAME] articulation world pose position={np.round(pos, 5)} quat_raw={np.round(quat_raw, 5)} using={best_name}")
    return T, best_name


def build_arm_models(joints: Dict[str, JointInfo], dof_names: Sequence[str]) -> Dict[str, ArmModel]:
    chains = discover_arm_chains(joints, dof_names)
    arms: Dict[str, ArmModel] = {}
    for arm_name, (joint_names, ee_link) in chains.items():
        indices = [dof_names.index(name) for name in joint_names]
        joint_infos = [joints[name] for name in joint_names]
        lower = np.array([j.lower for j in joint_infos], dtype=np.float64)
        upper = np.array([j.upper for j in joint_infos], dtype=np.float64)
        arms[arm_name] = ArmModel(arm_name, joint_names, ee_link, joint_infos, indices, lower, upper)
        print(f"[CHAIN] {arm_name}: ee_link={ee_link}, joints={joint_names}, dof_indices={indices}")
    if not arms:
        raise RuntimeError("No controllable arm chain could be resolved from URDF + Isaac DOF names")
    return arms


def current_world_pose(arm: ArmModel, q_all: np.ndarray, base_T_world: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    q_arm = q_all[arm.joint_indices]
    T_base_ee, _ = fk_and_jacobian(arm, q_arm)
    T_world_ee = base_T_world @ T_base_ee
    return T_world_ee[:3, 3].copy(), rot_to_quat_wxyz(T_world_ee[:3, :3])




def get_simulation_app_class():
    """Resolve SimulationApp across Isaac Sim 4.x/5.x and pip/kit launch modes."""
    errors = []
    candidates = (
        ("isaacsim.simulation_app", "SimulationApp"),
        ("omni.isaac.kit", "SimulationApp"),
        ("isaacsim", "SimulationApp"),
    )
    for module_name, attr_name in candidates:
        try:
            module = __import__(module_name, fromlist=[attr_name])
            cls = getattr(module, attr_name, None)
            if cls is not None:
                return cls
            errors.append(f"{module_name}.{attr_name}=None")
        except Exception as exc:
            errors.append(f"{module_name}: {exc}")
    raise ImportError(
        "Could not import Isaac Sim SimulationApp. Run this script with Isaac Sim's python/python.sh, "
        "or source the official Isaac Sim environment before using a conda python. Tried: "
        + " | ".join(errors)
    )


# -----------------------------------------------------------------------------
# Main application
# -----------------------------------------------------------------------------
def main() -> None:
    urdf_path = ensure_isaac_ready_urdf()

    SimulationApp = get_simulation_app_class()
    simulation_app = SimulationApp({"headless": HEADLESS})

    import carb
    import omni.usd
    try:
        from isaacsim.core.api import World
    except Exception:
        from omni.isaac.core import World
    try:
        from isaacsim.core.utils.types import ArticulationAction
    except Exception:
        from omni.isaac.core.utils.types import ArticulationAction

    try:
        from isaacsim.core.api.articulations import Articulation
    except Exception:
        from omni.isaac.core.articulations import Articulation

    omni.usd.get_context().new_stage()
    robot_prim_path = import_urdf_to_stage(urdf_path, ROBOT_PRIM_PATH)

    world = World()
    if ADD_GROUND_PLANE:
        world.scene.add_default_ground_plane()
    robot = Articulation(robot_prim_path)
    world.scene.add(robot)
    world.reset()

    if DISABLE_GRAVITY:
        disable_robot_gravity(world, robot_prim_path)

    controller = robot.get_articulation_controller()
    dof_names = get_dof_names(robot)
    num_dof = int(robot.num_dof)
    print(f"[INFO] num_dof={num_dof}")
    print(f"[INFO] Isaac DOF names={dof_names}")
    if hasattr(controller, "set_gains"):
        controller.set_gains(np.ones(num_dof) * KP, np.ones(num_dof) * KD)

    joints = load_joints_from_urdf(urdf_path)
    arms = build_arm_models(joints, dof_names)
    arm_order = [name for name in ("right", "left", "arm") if name in arms] + [name for name in arms if name not in {"right", "left", "arm"}]
    active_arm_name = arm_order[0]
    control_target = active_arm_name
    control_frame = "world"

    base_T_world, _ = resolve_base_transform(robot, verbose=True)
    q_all = np.asarray(robot.get_joint_positions(), dtype=np.float64)
    for arm in arms.values():
        pos, quat = current_world_pose(arm, q_all, base_T_world)
        arm.target_pos_world = pos
        arm.target_quat_world = quat
        arm.start_target_pos_world = pos.copy()

    hold_action = ArticulationAction(joint_positions=q_all, joint_velocities=np.zeros_like(q_all))
    for _ in range(int(max(0.0, HOLD_SECONDS) / DEFAULT_DT)):
        controller.apply_action(hold_action)
        world.step(render=False)

    pressed = set()
    last_pressed = set()
    should_quit = False
    input_iface = carb.input.acquire_input_interface()
    keyboard = None
    try:
        import omni.appwindow

        keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    except Exception:
        keyboard = None
    if keyboard is None and hasattr(input_iface, "get_keyboard"):
        keyboard = input_iface.get_keyboard()
    if keyboard is None:
        raise RuntimeError("Keyboard device handle not found")

    def on_keyboard_event(event, *args, **kwargs):
        nonlocal should_quit
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            pressed.add(event.input)
            if event.input == carb.input.KeyboardInput.ESCAPE:
                should_quit = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            pressed.discard(event.input)

    input_iface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)

    def is_down(key) -> bool:
        return key in pressed

    def selected_arms() -> List[ArmModel]:
        if control_target == "both" and "left" in arms and "right" in arms:
            return [arms["right"], arms["left"]]
        return [arms[control_target]]

    print("\nControls:")
    print("  TAB: cycle active/view arm")
    print("  1: right arm, 2: left arm, 3: both arms (if both exist)")
    print("  Arrow Left/Right: world X, Arrow Up/Down: world Y")
    print("  Shift + Arrow Up/Down: world Z")
    print("  U/O: roll +/-, I/K: pitch +/-, J/L: yaw +/-")
    print("  F: toggle world/local translation frame")
    print("  R: reset selected arm target(s) to current EE pose")
    print("  ESC: quit\n")

    log_counter = 0
    last_time = time.time()
    while simulation_app.is_running() and not should_quit:
        now = time.time()
        dt = max(1e-4, min(0.05, now - last_time))
        last_time = now

        newly_pressed = pressed - last_pressed
        last_pressed = set(pressed)

        if carb.input.KeyboardInput.TAB in newly_pressed:
            idx = (arm_order.index(active_arm_name) + 1) % len(arm_order)
            active_arm_name = arm_order[idx]
            if control_target != "both":
                control_target = active_arm_name
            print(f"[INFO] active_arm={active_arm_name}, control_target={control_target}")
        if carb.input.KeyboardInput.KEY_1 in newly_pressed and "right" in arms:
            active_arm_name = "right"
            control_target = "right"
            print("[INFO] control_target=RIGHT")
        if carb.input.KeyboardInput.KEY_2 in newly_pressed and "left" in arms:
            active_arm_name = "left"
            control_target = "left"
            print("[INFO] control_target=LEFT")
        if carb.input.KeyboardInput.KEY_3 in newly_pressed and "right" in arms and "left" in arms:
            control_target = "both"
            print("[INFO] control_target=BOTH")
        if carb.input.KeyboardInput.F in newly_pressed:
            control_frame = "local" if control_frame == "world" else "world"
            print(f"[INFO] control_frame={control_frame}")

        base_T_world, _ = resolve_base_transform(robot)
        q_all = np.asarray(robot.get_joint_positions(), dtype=np.float64)

        if carb.input.KeyboardInput.R in newly_pressed:
            for arm in selected_arms():
                pos, quat = current_world_pose(arm, q_all, base_T_world)
                arm.target_pos_world = pos
                arm.target_quat_world = quat
                arm.start_target_pos_world = pos.copy()
            print(f"[INFO] reset target(s): {control_target}")

        shift = is_down(carb.input.KeyboardInput.LEFT_SHIFT) or is_down(carb.input.KeyboardInput.RIGHT_SHIFT)
        delta = np.zeros(3, dtype=np.float64)
        step = POSITION_STEP_MPS * dt
        if shift:
            delta[2] += (float(is_down(carb.input.KeyboardInput.UP)) - float(is_down(carb.input.KeyboardInput.DOWN))) * step
        else:
            delta[0] += (float(is_down(carb.input.KeyboardInput.RIGHT)) - float(is_down(carb.input.KeyboardInput.LEFT))) * step
            delta[1] += (float(is_down(carb.input.KeyboardInput.UP)) - float(is_down(carb.input.KeyboardInput.DOWN))) * step

        da = ROTATION_STEP_RPS * dt
        roll = (float(is_down(carb.input.KeyboardInput.U)) - float(is_down(carb.input.KeyboardInput.O))) * da
        pitch = (float(is_down(carb.input.KeyboardInput.I)) - float(is_down(carb.input.KeyboardInput.K))) * da
        yaw = (float(is_down(carb.input.KeyboardInput.J)) - float(is_down(carb.input.KeyboardInput.L))) * da

        for arm in selected_arms():
            if np.linalg.norm(delta) > 0:
                command_delta = delta.copy()
                if control_frame == "local":
                    command_delta = quat_wxyz_to_rot(arm.target_quat_world) @ command_delta
                arm.target_pos_world += command_delta
                lo = arm.start_target_pos_world - MAX_TARGET_OFFSET_FROM_START
                hi = arm.start_target_pos_world + MAX_TARGET_OFFSET_FROM_START
                arm.target_pos_world = np.minimum(np.maximum(arm.target_pos_world, lo), hi)
            for axis, angle in (([1, 0, 0], roll), ([0, 1, 0], pitch), ([0, 0, 1], yaw)):
                if abs(angle) > 1e-12:
                    dq = quat_from_axis_angle(axis, angle)
                    arm.target_quat_world = quat_mul(dq, arm.target_quat_world) if control_frame == "world" else quat_mul(arm.target_quat_world, dq)

        q_cmd = q_all.copy()
        pos_err = ori_err = 0.0
        for arm in selected_arms():
            q_arm = q_cmd[arm.joint_indices]
            q_next, pos_err, ori_err = solve_arm_ik_step(arm, q_arm, base_T_world)
            q_cmd[arm.joint_indices] = q_next
        controller.apply_action(ArticulationAction(joint_positions=q_cmd))
        world.step(render=True)

        log_counter += 1
        if log_counter % 60 == 0:
            view_arm = arms[active_arm_name]
            cur_pos, _ = current_world_pose(view_arm, np.asarray(robot.get_joint_positions(), dtype=np.float64), base_T_world)
            print(
                f"[status] view={active_arm_name.upper()} target={control_target.upper()} frame={control_frame} "
                f"ee={np.round(cur_pos, 4)} target_pos={np.round(view_arm.target_pos_world, 4)} "
                f"pos_err={pos_err:.4f} ori_err={ori_err:.4f}"
            )

    simulation_app.close()


if __name__ == "__main__":
    main()
