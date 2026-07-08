"""
Reusable Isaac Sim keyboard task-space controller for combined mobile dual-arm robots.

The module loads light_ware5.usd, imports a configured URDF when the robot prim is
not already present, and controls the two end effectors with Lula IK.  It omits the
serial/FOB code from dualarm_fob_both_control.py on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import argparse
import numpy as np

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import carb
from isaacsim.core.api import World
from isaacsim.core.utils.stage import get_current_stage, open_stage
from omni.isaac.core.utils.types import ArticulationAction

try:
    from isaacsim.core.api.articulations import Articulation
except Exception:
    from omni.isaac.core.articulations import Articulation

from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver, ArticulationKinematicsSolver

EE_SPEED_MPS = 0.10
ROT_SPEED_RPS = 0.8
MOBILE_JOINT_SPEED_RPS = 0.5
CONTROL_FRAME = "base"  # "base" or "tool"
KP = 1000.0
KD = 1200.0
HOLD_SECONDS = 1.0
FORCE_BASE_WXYZ_SWAP = True
DISABLE_GRAVITY = True

USE_AXIS_AUTO_ALIGN = True
ALIGN_PROBE_DIST = 0.035
ALIGN_SETTLE_STEPS = 70
ALIGN_RESTORE_STEPS = 70
MIN_ALIGN_TO_ENABLE = 0.35
MIN_RESPONSE_NORM = 0.0015
MAX_DELTA_PER_STEP = 0.2
MAX_TARGET_OFFSET_FROM_START = np.array([0.60, 0.60, 0.60], dtype=np.float64)
MOBILE_JOINT_LOWER_RAD = -np.pi
MOBILE_JOINT_UPPER_RAD = np.pi
MOBILE_JOINT_MAX_FORCE = 500.0
MOBILE_JOINT_NAMES = [f"joint{i}_mobile" for i in range(1, 9)]
MOBILE_JOINT_KEY_BINDINGS = [
    ("Q", "A"),
    ("W", "S"),
    ("E", "D"),
    ("R", "F"),
    ("T", "G"),
    ("Y", "H"),
    ("U", "J"),
    ("I", "K"),
]


ROBOT_CONFIGS = {
    "humanoid_base": {
        "stage_path": "./com_hum_light_test.usd",
        "urdf_path": "./humanoid_urdf_assemble/urdf/combined_mobile_humanoid_base.urdf",
        "robot_prim_path": "/World/combined_mobile_humanoid_base",
        "articulation_root_link": "base_mobile",
        "right_desc_yaml": "./combined_mobile_humanoid_base_right_arm_robot_descriptor.yaml",
        "left_desc_yaml": "./combined_mobile_humanoid_base_left_arm_robot_descriptor.yaml",
        "right_ee_frame": "link6_R",
        "left_ee_frame": "link6_L",
    },
    "v4_onlyarm": {
        "stage_path": "./com_onlyarm_light_test.usd",
        "urdf_path": "./humanoid_urdf_assemble/urdf/combined_mobile_v4_onlyarm.urdf",
        "robot_prim_path": "/World/combined_mobile_v4_onlyarm",
        "articulation_root_link": "base_mobile",
        "right_desc_yaml": "./combined_mobile_v4_onlyarm_right_arm_robot_descriptor.yaml",
        "left_desc_yaml": "./combined_mobile_v4_onlyarm_left_arm_robot_descriptor.yaml",
        "right_ee_frame": "link7",
        "left_ee_frame": "link14",
    },
}


def fmt(a):
    return np.array2string(np.asarray(a), precision=5, suppress_small=True)


def as_wxyz_quat(rot):
    r = np.array(rot, dtype=np.float64)
    if r.shape == (4,):
        return r
    if r.shape == (3,):
        roll, pitch, yaw = r
        cr, sr = np.cos(roll / 2), np.sin(roll / 2)
        cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
        cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
        return np.array([
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ], dtype=np.float64)
    if r.shape == (3, 3):
        m = r
        tr = np.trace(m)
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2
            q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
        return q / (np.linalg.norm(q) + 1e-12)
    raise ValueError(f"Unsupported rotation shape: {r.shape}")


def to_wxyz_from_xyzw(q):
    q = np.array(q, dtype=np.float64)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)


def quat_norm(q):
    return q / (np.linalg.norm(q) + 1e-12)


def quat_from_axis_angle(axis, angle):
    axis = np.array(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    half = angle * 0.5
    return np.array([np.cos(half), *(axis * np.sin(half))], dtype=np.float64)


def unit(v):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = np.linalg.norm(v)
    return np.zeros_like(v) if n < 1e-12 else v / n


def alignment_score(actual_delta, desired_delta):
    au, du = unit(actual_delta), unit(desired_delta)
    if np.linalg.norm(au) < 1e-12 or np.linalg.norm(du) < 1e-12:
        return 0.0
    return float(np.dot(au, du))


def clamp_vec(v, max_abs):
    return np.clip(np.asarray(v, dtype=np.float64).copy(), -max_abs, max_abs)



def get_dof_names(robot) -> list[str]:
    if hasattr(robot, "dof_names"):
        return list(robot.dof_names)
    if hasattr(robot, "get_dof_names"):
        return list(robot.get_dof_names())
    return []


def resolve_mobile_joint_indices(robot) -> list[int]:
    dof_names = get_dof_names(robot)
    indices = []
    for name in MOBILE_JOINT_NAMES:
        idx = None
        if hasattr(robot, "get_dof_index"):
            try:
                idx = int(robot.get_dof_index(name))
            except Exception:
                idx = None
        if idx is None and name in dof_names:
            idx = dof_names.index(name)
        if idx is None:
            print(f"[WARN] Mobile joint DOF not found in articulation: {name}")
            continue
        indices.append(idx)
    return indices


def make_mobile_joint_action(mobile_joint_indices, mobile_joint_targets):
    return ArticulationAction(
        joint_positions=np.asarray(mobile_joint_targets, dtype=np.float64),
        joint_velocities=np.zeros(len(mobile_joint_indices), dtype=np.float64),
        joint_indices=np.asarray(mobile_joint_indices, dtype=np.int32),
    )

def set_ik_iters(kin, iters: int):
    if hasattr(kin, "set_max_iterations"):
        kin.set_max_iterations(iters)
    elif hasattr(kin, "ccd_max_iterations"):
        kin.ccd_max_iterations = iters


def iter_world_prims(stage):
    from pxr import Usd

    world = stage.GetPrimAtPath("/World")
    search_root = world if world and world.IsValid() else stage.GetPseudoRoot()
    return Usd.PrimRange(search_root)


def find_prim_by_name(stage, prim_name: str):
    matches = [prim for prim in iter_world_prims(stage) if prim.GetName() == prim_name]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        paths = [str(prim.GetPath()) for prim in matches]
        raise RuntimeError(f"Multiple prims named '{prim_name}' found: {paths}")
    raise RuntimeError(f"Prim named '{prim_name}' not found in opened USD stage")


def configure_mobile_joint_usd_limits(stage) -> None:
    """Relax mobile joint limits/drives in the loaded USD so targets can move immediately."""
    from pxr import UsdPhysics

    configured = 0
    for prim in iter_world_prims(stage):
        if prim.GetName() not in MOBILE_JOINT_NAMES:
            continue
        if prim.HasAPI(UsdPhysics.RevoluteJoint):
            joint = UsdPhysics.RevoluteJoint(prim)
        else:
            joint = UsdPhysics.RevoluteJoint.Apply(prim)
        joint.CreateLowerLimitAttr(MOBILE_JOINT_LOWER_RAD).Set(MOBILE_JOINT_LOWER_RAD)
        joint.CreateUpperLimitAttr(MOBILE_JOINT_UPPER_RAD).Set(MOBILE_JOINT_UPPER_RAD)
        try:
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            if not drive:
                drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            drive.CreateMaxForceAttr(MOBILE_JOINT_MAX_FORCE).Set(MOBILE_JOINT_MAX_FORCE)
        except Exception as exc:
            print(f"[WARN] Could not configure angular drive force for {prim.GetPath()}: {exc}")
        configured += 1

    if configured:
        print(
            f"[INFO] Mobile USD joint limits configured for {configured} joints: "
            f"lower={MOBILE_JOINT_LOWER_RAD:.3f}, upper={MOBILE_JOINT_UPPER_RAD:.3f}, max_force={MOBILE_JOINT_MAX_FORCE:.1f}"
        )
    else:
        print("[WARN] No mobile joint prims found while configuring USD joint limits")


def select_articulation_root_path(stage, preferred_root_link: str) -> str:
    """Select an existing ArticulationRootAPI prim without mutating the authored USD hierarchy."""
    from pxr import UsdPhysics

    articulation_paths = []
    preferred_path = None
    for prim in iter_world_prims(stage):
        if prim.GetName() == preferred_root_link:
            preferred_path = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_paths.append(str(prim.GetPath()))

    if preferred_path in articulation_paths:
        if len(articulation_paths) > 1:
            print(
                f"[WARN] Multiple articulation roots are authored in the stage: {articulation_paths}. "
                f"Using preferred existing root: {preferred_path}"
            )
        else:
            print(f"[INFO] Articulation root active: {preferred_path}")
        return preferred_path

    if len(articulation_paths) == 1:
        print(
            f"[WARN] Preferred articulation root link '{preferred_root_link}' is not authored as an ArticulationRootAPI. "
            f"Using existing root: {articulation_paths[0]}"
        )
        return articulation_paths[0]

    if len(articulation_paths) > 1:
        raise RuntimeError(
            f"Multiple articulation roots are authored in the stage and preferred root '{preferred_root_link}' "
            f"is not one of them: {articulation_paths}"
        )

    raise RuntimeError(
        f"No ArticulationRootAPI found in the opened USD stage. "
        f"Author exactly one articulation root in USD, preferably on '{preferred_root_link}'."
    )


def find_articulation_prim_path(stage, preferred_path: str) -> str:
    """Return preferred_path when valid, otherwise find one articulation in the opened USD stage."""
    preferred = stage.GetPrimAtPath(preferred_path)
    if preferred and preferred.IsValid():
        return preferred_path

    from pxr import UsdPhysics

    articulation_paths = []
    for prim in iter_world_prims(stage):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_paths.append(str(prim.GetPath()))

    if len(articulation_paths) == 1:
        print(f"[WARN] Preferred robot prim not found: {preferred_path}. Using articulation root: {articulation_paths[0]}")
        return articulation_paths[0]
    if len(articulation_paths) > 1:
        raise RuntimeError(
            f"Preferred robot prim not found: {preferred_path}. "
            f"Multiple articulation roots found; set ROBOT_CONFIGS robot_prim_path explicitly: {articulation_paths}"
        )
    raise RuntimeError(f"Robot prim path not found and no articulation root discovered in stage: {preferred_path}")


def disable_robot_gravity(stage_or_world, robot_prim_path: str) -> None:
    """Disable gravity by authoring PhysX USD attributes, without creating RigidPrimView objects."""
    from pxr import Usd, UsdPhysics, PhysxSchema

    stage = stage_or_world.stage if hasattr(stage_or_world, "stage") else stage_or_world
    root = stage.GetPrimAtPath(robot_prim_path)
    if not root or not root.IsValid():
        raise RuntimeError(f"Robot prim path not found: {robot_prim_path}")

    rigid_count = 0
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        physx_rigid = PhysxSchema.PhysxRigidBodyAPI(prim)
        if not physx_rigid:
            physx_rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        physx_rigid.CreateDisableGravityAttr(True).Set(True)
        rigid_count += 1

    if rigid_count == 0:
        print(f"[WARN] No rigid bodies found under {robot_prim_path}; gravity disable skipped")
        return
    print(f"[INFO] Gravity disabled for {rigid_count} rigid bodies under {robot_prim_path}")


def articulation_is_ready(robot) -> bool:
    if hasattr(robot, "is_initialized"):
        try:
            if not robot.is_initialized():
                return False
        except Exception:
            pass
    try:
        joint_positions = robot.get_joint_positions()
    except Exception:
        return False
    return joint_positions is not None


def wait_for_articulation_ready(world: World, robot, max_steps: int = 180) -> None:
    for _ in range(max_steps):
        if articulation_is_ready(robot):
            return
        if hasattr(robot, "initialize"):
            try:
                robot.initialize()
            except Exception:
                pass
        world.step(render=False)
    raise RuntimeError("Robot articulation did not initialize; check the USD ArticulationRootAPI and robot prim path.")


def safe_apply_action(controller, action, fallback_action=None) -> bool:
    selected_action = action if action is not None else fallback_action
    if selected_action is None or selected_action.joint_positions is None:
        return False
    controller.apply_action(selected_action)
    return True


@dataclass
class ArmState:
    name: str
    kin: object
    ik: object
    frame_name: str
    target_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    target_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
    start_target_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    axis_align_enabled: bool = USE_AXIS_AUTO_ALIGN
    axis_sign: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float64))
    axis_scale: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float64))
    axis_enabled: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=bool))


def main(robot_key: str) -> None:
    global CONTROL_FRAME
    cfg = ROBOT_CONFIGS[robot_key]

    open_stage(cfg["stage_path"])
    stage = get_current_stage()
    robot_prim_path = select_articulation_root_path(stage, cfg["articulation_root_link"])
    configure_mobile_joint_usd_limits(stage)
    if DISABLE_GRAVITY:
        disable_robot_gravity(stage, robot_prim_path)

    world = World()
    world.scene.add_default_ground_plane()
    robot = Articulation(robot_prim_path)
    world.scene.add(robot)
    world.reset()
    wait_for_articulation_ready(world, robot)

    controller = robot.get_articulation_controller()
    if hasattr(controller, "set_gains"):
        try:
            controller.set_gains(np.ones(robot.num_dof) * KP, np.ones(robot.num_dof) * KD)
        except Exception as exc:
            print(f"[WARN] controller.set_gains skipped because the articulation physics view is not ready: {exc}")

    jp0 = robot.get_joint_positions()
    if jp0 is None:
        raise RuntimeError("Robot articulation initialized but returned no joint positions.")
    hold_action = ArticulationAction(joint_positions=jp0, joint_velocities=np.zeros_like(jp0))
    dt = 1.0 / 60.0
    for _ in range(int(max(0.0, HOLD_SECONDS) / dt)):
        safe_apply_action(controller, hold_action)
        world.step(render=False)

    right_kin = LulaKinematicsSolver(robot_description_path=cfg["right_desc_yaml"], urdf_path=cfg["urdf_path"])
    left_kin = LulaKinematicsSolver(robot_description_path=cfg["left_desc_yaml"], urdf_path=cfg["urdf_path"])
    set_ik_iters(right_kin, 80)
    set_ik_iters(left_kin, 80)
    right_ik = ArticulationKinematicsSolver(robot, right_kin, cfg["right_ee_frame"])
    left_ik = ArticulationKinematicsSolver(robot, left_kin, cfg["left_ee_frame"])

    base_t, base_q_raw = robot.get_world_pose()
    base_q_use = to_wxyz_from_xyzw(base_q_raw) if FORCE_BASE_WXYZ_SWAP else np.array(base_q_raw, dtype=np.float64)
    right_kin.set_robot_base_pose(base_t, base_q_use)
    left_kin.set_robot_base_pose(base_t, base_q_use)

    arms = {
        "right": ArmState("right", right_kin, right_ik, cfg["right_ee_frame"]),
        "left": ArmState("left", left_kin, left_ik, cfg["left_ee_frame"]),
    }
    active_arm = "right"
    active_control_target = "both"
    control_mode = "arm"  # "arm" or "mobile"
    active_mobile_joint = 0
    mobile_joint_indices = resolve_mobile_joint_indices(robot)
    mobile_joint_targets = robot.get_joint_positions()[mobile_joint_indices].copy() if mobile_joint_indices else np.array([], dtype=np.float64)

    for arm in arms.values():
        pos, rot = arm.ik.compute_end_effector_pose()
        arm.target_pos = np.array(pos, dtype=np.float64)
        arm.target_quat = as_wxyz_quat(rot)
        arm.start_target_pos = arm.target_pos.copy()

    pressed = set()
    last_pressed = set()
    should_quit = False
    input_iface = carb.input.acquire_input_interface()
    try:
        import omni.appwindow
        keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    except Exception:
        keyboard = input_iface.get_keyboard() if hasattr(input_iface, "get_keyboard") else None
    if keyboard is None:
        raise RuntimeError("Keyboard device handle not found.")

    def on_keyboard_event(event, *args, **kwargs):
        nonlocal should_quit
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            pressed.add(event.input)
            if event.input == carb.input.KeyboardInput.ESCAPE:
                should_quit = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            pressed.discard(event.input)

    input_iface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)

    def is_down(key):
        return key in pressed

    def refresh_base_pose():
        base_t_now, _ = robot.get_world_pose()
        right_kin.set_robot_base_pose(base_t_now, base_q_use)
        left_kin.set_robot_base_pose(base_t_now, base_q_use)

    def step_to_target_for_arm(arm: ArmState, goal_pos, goal_quat, steps=ALIGN_SETTLE_STEPS):
        ok_last = False
        for _ in range(steps):
            refresh_base_pose()
            action, ok = arm.ik.compute_inverse_kinematics(goal_pos, goal_quat)
            ok_last = ok
            safe_apply_action(controller, action if ok else None, hold_action)
            world.step(render=False)
        return ok_last

    def probe_axiswise_alignment_for_arm(arm_name: str):
        arm = arms[arm_name]
        print(f"\n=== AXIS-WISE AUTO ALIGN PROBE ({arm.name.upper()} / {arm.frame_name}) ===")
        refresh_base_pose()
        start_pos, start_rot = arm.ik.compute_end_effector_pose()
        start_pos = np.array(start_pos, dtype=np.float64)
        start_quat = as_wxyz_quat(start_rot)
        signs = np.ones(3, dtype=np.float64)
        scales = np.ones(3, dtype=np.float64)
        enabled = np.ones(3, dtype=bool)
        for label, desired, idx in [
            ("X", np.array([ALIGN_PROBE_DIST, 0.0, 0.0]), 0),
            ("Y", np.array([0.0, ALIGN_PROBE_DIST, 0.0]), 1),
            ("Z", np.array([0.0, 0.0, ALIGN_PROBE_DIST]), 2),
        ]:
            ok = step_to_target_for_arm(arm, start_pos + desired, start_quat)
            cur_pos, _ = arm.ik.compute_end_effector_pose()
            actual = np.array(cur_pos, dtype=np.float64) - start_pos
            comp = actual[idx]
            align = alignment_score(actual, desired)
            signs[idx] = 1.0 if comp >= 0.0 else -1.0
            scales[idx] = min(3.0, ALIGN_PROBE_DIST / abs(comp)) if abs(comp) > 1e-9 else 1.0
            enabled[idx] = align >= MIN_ALIGN_TO_ENABLE and np.linalg.norm(actual) >= MIN_RESPONSE_NORM
            print(f"[{arm.name.upper()}] axis {label}: desired={fmt(desired)} actual={fmt(actual)} align={align:+.4f} enabled={enabled[idx]} ok={ok}")
            step_to_target_for_arm(arm, start_pos, start_quat, steps=ALIGN_RESTORE_STEPS)
        arm.axis_sign = signs
        arm.axis_scale = scales
        arm.axis_enabled = enabled
        cur_pos, cur_rot = arm.ik.compute_end_effector_pose()
        arm.target_pos = np.array(cur_pos, dtype=np.float64)
        arm.target_quat = as_wxyz_quat(cur_rot)
        arm.start_target_pos = arm.target_pos.copy()
        print(f"[{arm.name.upper()}] axis_enabled={arm.axis_enabled} axis_sign={fmt(arm.axis_sign)} axis_scale={fmt(arm.axis_scale)}")

    def apply_axiswise_correction(arm: ArmState, desired_delta_base):
        corrected = np.zeros(3, dtype=np.float64)
        for i in range(3):
            corrected[i] = arm.axis_sign[i] * arm.axis_scale[i] * desired_delta_base[i] if arm.axis_enabled[i] else 0.0
        return corrected

    if USE_AXIS_AUTO_ALIGN:
        probe_axiswise_alignment_for_arm("right")
        probe_axiswise_alignment_for_arm("left")

    print("Controls:")
    print("  V: toggle control mode (arm/mobile)")
    print("  [ARM] TAB: switch active arm view  |  1/2/3: control right/left/both arms")
    print("  Arrow keys: translate X/Y. Shift+Up/Down: Z")
    print("  I/K: pitch +/-,  J/L: yaw +/-,  U/O: roll +/-")
    print("  F: toggle CONTROL_FRAME (base/tool)")
    print("  R: reset selected arm target pose")
    print("  M: toggle axis auto alignment on/off for active arm")
    print("  T/G: re-run axis auto alignment for active/both arms")
    print("  [MOBILE] 1-8: select mobile joint")
    print("  [MOBILE] Left/Down: selected joint - | Right/Up: selected joint +")
    print("  [MOBILE] Q/A W/S E/D R/F T/G Y/H U/J I/K: joint1..joint8 +/-")
    print("  ESC: quit")
    print(f"  ROBOT={robot_key}, STAGE={cfg['stage_path']}, PRIM={robot_prim_path}")
    print(f"  MODE={control_mode}, CONTROL_FRAME={CONTROL_FRAME}, ACTIVE_ARM={active_arm}")

    log_counter = 0
    while simulation_app.is_running() and not should_quit:
        world.step(render=True)
        log_counter += 1
        newly_pressed = pressed - last_pressed
        last_pressed = set(pressed)

        if carb.input.KeyboardInput.V in newly_pressed:
            control_mode = "mobile" if control_mode == "arm" else "arm"
            if control_mode == "mobile" and mobile_joint_indices:
                mobile_joint_targets[:] = robot.get_joint_positions()[mobile_joint_indices]
            elif control_mode == "arm":
                for target_arm in arms.values():
                    cur_pos, cur_rot = target_arm.ik.compute_end_effector_pose()
                    target_arm.target_pos = np.array(cur_pos, dtype=np.float64)
                    target_arm.target_quat = as_wxyz_quat(cur_rot)
                    target_arm.start_target_pos = target_arm.target_pos.copy()
            print(f"[INFO] Control mode -> {control_mode.upper()}")

        number_keys = [
            carb.input.KeyboardInput.KEY_1, carb.input.KeyboardInput.KEY_2,
            carb.input.KeyboardInput.KEY_3, carb.input.KeyboardInput.KEY_4,
            carb.input.KeyboardInput.KEY_5, carb.input.KeyboardInput.KEY_6,
            carb.input.KeyboardInput.KEY_7, carb.input.KeyboardInput.KEY_8,
        ]

        if control_mode == "mobile":
            for i, key in enumerate(number_keys):
                if key in newly_pressed and i < len(mobile_joint_indices):
                    active_mobile_joint = i
                    print(f"[INFO] Active mobile joint -> {MOBILE_JOINT_NAMES[i]}")
        else:
            if carb.input.KeyboardInput.TAB in newly_pressed:
                active_arm = "left" if active_arm == "right" else "right"
                print(f"[INFO] Active arm switched to: {active_arm.upper()}")
            if carb.input.KeyboardInput.KEY_1 in newly_pressed:
                active_control_target = "right"; active_arm = "right"; print("[INFO] Control target -> RIGHT")
            if carb.input.KeyboardInput.KEY_2 in newly_pressed:
                active_control_target = "left"; active_arm = "left"; print("[INFO] Control target -> LEFT")
            if carb.input.KeyboardInput.KEY_3 in newly_pressed:
                active_control_target = "both"; print("[INFO] Control target -> BOTH")
            if carb.input.KeyboardInput.F in newly_pressed:
                CONTROL_FRAME = "tool" if CONTROL_FRAME == "base" else "base"
                print(f"[INFO] CONTROL_FRAME switched to: {CONTROL_FRAME}")
            if carb.input.KeyboardInput.M in newly_pressed:
                arms[active_arm].axis_align_enabled = not arms[active_arm].axis_align_enabled
                print(f"[INFO] {active_arm.upper()} AXIS_AUTO_ALIGN -> {arms[active_arm].axis_align_enabled}")
            if carb.input.KeyboardInput.T in newly_pressed:
                probe_axiswise_alignment_for_arm(active_arm)
            if carb.input.KeyboardInput.G in newly_pressed:
                probe_axiswise_alignment_for_arm("right"); probe_axiswise_alignment_for_arm("left")

        refresh_base_pose()
        control_arms = [arms["right"], arms["left"]] if active_control_target == "both" else [arms[active_control_target]]

        if control_mode == "mobile":
            mobile_step = MOBILE_JOINT_SPEED_RPS * dt
            if len(mobile_joint_targets) > 0:
                selected_dir = (
                    is_down(carb.input.KeyboardInput.RIGHT) + is_down(carb.input.KeyboardInput.UP)
                    - is_down(carb.input.KeyboardInput.LEFT) - is_down(carb.input.KeyboardInput.DOWN)
                )
                mobile_joint_targets[active_mobile_joint] += selected_dir * mobile_step
                key_name_to_input = {name: getattr(carb.input.KeyboardInput, name) for pair in MOBILE_JOINT_KEY_BINDINGS for name in pair}
                for i, (plus_key, minus_key) in enumerate(MOBILE_JOINT_KEY_BINDINGS[:len(mobile_joint_targets)]):
                    direction = is_down(key_name_to_input[plus_key]) - is_down(key_name_to_input[minus_key])
                    mobile_joint_targets[i] += direction * mobile_step
                safe_apply_action(controller, make_mobile_joint_action(mobile_joint_indices, mobile_joint_targets))

            if log_counter % 60 == 0:
                if len(mobile_joint_targets) > 0:
                    print(f"[status] mode=MOBILE active={MOBILE_JOINT_NAMES[active_mobile_joint]} mobile_q={fmt(mobile_joint_targets)}")
                else:
                    print("[status] mode=MOBILE no mobile joints found")
            continue

        d = EE_SPEED_MPS * dt
        shift = is_down(carb.input.KeyboardInput.LEFT_SHIFT) or is_down(carb.input.KeyboardInput.RIGHT_SHIFT)
        desired_delta = np.zeros(3, dtype=np.float64)
        if is_down(carb.input.KeyboardInput.UP):
            desired_delta[2 if shift else 0] += d
        if is_down(carb.input.KeyboardInput.DOWN):
            desired_delta[2 if shift else 0] -= d
        if is_down(carb.input.KeyboardInput.LEFT):
            desired_delta[1] += d
        if is_down(carb.input.KeyboardInput.RIGHT):
            desired_delta[1] -= d
        desired_delta = clamp_vec(desired_delta, MAX_DELTA_PER_STEP)

        for target_arm in control_arms:
            command_delta = apply_axiswise_correction(target_arm, desired_delta) if target_arm.axis_align_enabled else desired_delta.copy()
            command_delta = clamp_vec(command_delta, MAX_DELTA_PER_STEP)
            if np.linalg.norm(command_delta) > 0:
                if CONTROL_FRAME == "tool":
                    w, x, y, z = target_arm.target_quat
                    rot = np.array([
                        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
                        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
                    ], dtype=np.float64)
                    target_arm.target_pos += rot @ command_delta
                else:
                    target_arm.target_pos += command_delta
                target_arm.target_pos = np.minimum(
                    np.maximum(target_arm.target_pos, target_arm.start_target_pos - MAX_TARGET_OFFSET_FROM_START),
                    target_arm.start_target_pos + MAX_TARGET_OFFSET_FROM_START,
                )

        if carb.input.KeyboardInput.R in newly_pressed:
            for target_arm in control_arms:
                cur_pos, cur_rot = target_arm.ik.compute_end_effector_pose()
                target_arm.target_pos = np.array(cur_pos, dtype=np.float64)
                target_arm.target_quat = as_wxyz_quat(cur_rot)
                target_arm.start_target_pos = target_arm.target_pos.copy()
            print(f"[INFO] target pose reset for: {active_control_target.upper()}")

        da = ROT_SPEED_RPS * dt
        roll = (is_down(carb.input.KeyboardInput.U) - is_down(carb.input.KeyboardInput.O)) * da
        pitch = (is_down(carb.input.KeyboardInput.I) - is_down(carb.input.KeyboardInput.K)) * da
        yaw = (is_down(carb.input.KeyboardInput.J) - is_down(carb.input.KeyboardInput.L)) * da
        for target_arm in control_arms:
            for axis, angle in [([1, 0, 0], roll), ([0, 1, 0], pitch), ([0, 0, 1], yaw)]:
                if abs(angle) > 0:
                    dq = quat_from_axis_angle(axis, angle)
                    target_arm.target_quat = quat_norm(quat_mul(target_arm.target_quat, dq)) if CONTROL_FRAME == "tool" else quat_norm(quat_mul(dq, target_arm.target_quat))

        ok = True
        for target_arm in control_arms:
            action, arm_ok = target_arm.ik.compute_inverse_kinematics(target_arm.target_pos, target_arm.target_quat)
            ok = ok and arm_ok
            safe_apply_action(controller, action if arm_ok else None, hold_action)

        if log_counter % 60 == 0:
            arm = arms[active_arm]
            cur_pos, _ = arm.ik.compute_end_effector_pose()
            pos_err = np.linalg.norm(np.array(cur_pos, dtype=np.float64) - arm.target_pos)
            print(f"[status] mode=ARM ok={ok} view_arm={arm.name.upper()} control_target={active_control_target.upper()} frame={CONTROL_FRAME.upper()} "
                  f"target={fmt(arm.target_pos)} current={fmt(cur_pos)} err={pos_err:.5f} enabled={arm.axis_enabled}")

    simulation_app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=sorted(ROBOT_CONFIGS), required=True)
    args = parser.parse_args()
    main(args.robot)
