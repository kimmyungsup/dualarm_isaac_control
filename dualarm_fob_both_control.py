"""
Dual-arm task-space controller with keyboard/FOB switching and per-arm axis auto-alignment.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.utils.stage import open_stage
from omni.isaac.core.utils.types import ArticulationAction

try:
    from isaacsim.core.api.articulations import Articulation
except Exception:
    from omni.isaac.core.articulations import Articulation

from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver, ArticulationKinematicsSolver
import carb

try:
    import serial
except Exception:
    serial = None

POINT_CMD = 0x42
POS_ANGLES_CMD = 0x59
RECORD_LEN = 12

@dataclass
class PoseAngles:
    x_in: float
    y_in: float
    z_in: float
    x_cm: float
    y_cm: float
    z_cm: float
    azimuth_deg: float
    elevation_deg: float
    roll_deg: float
    raw_words: tuple

def open_serial(port: str, baud: int, timeout: float):
    if serial is None:
        raise RuntimeError("pyserial이 필요합니다. pip install pyserial")
    return serial.Serial(
        port=port, baudrate=baud, timeout=timeout,
        bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE,
    )

def decode_fob_word(lsb_byte: int, msb_byte: int) -> int:
    word = ((msb_byte & 0x7F) << 9) | ((lsb_byte & 0x7F) << 2)
    if word >= 0x8000:
        word -= 0x10000
    return word

def position_to_inches(raw: int, max_range_in: float) -> float:
    return (raw * max_range_in) / 32768.0

def angle_to_degrees(raw: int) -> float:
    return (raw * 180.0) / 32768.0

def parse_position_angles_record(record: bytes, max_range_in: float) -> PoseAngles:
    words = []
    for i in range(0, RECORD_LEN, 2):
        words.append(decode_fob_word(record[i], record[i + 1]))
    x_raw, y_raw, z_raw, zang_raw, yang_raw, xang_raw = words
    x_in = position_to_inches(x_raw, max_range_in)
    y_in = position_to_inches(y_raw, max_range_in)
    z_in = position_to_inches(z_raw, max_range_in)
    return PoseAngles(
        x_in=x_in, y_in=y_in, z_in=z_in,
        x_cm=x_in * 2.54, y_cm=y_in * 2.54, z_cm=z_in * 2.54,
        azimuth_deg=angle_to_degrees(zang_raw),
        elevation_deg=angle_to_degrees(yang_raw),
        roll_deg=angle_to_degrees(xang_raw),
        raw_words=(x_raw, y_raw, z_raw, zang_raw, yang_raw, xang_raw),
    )

def read_exact(ser, n: int, deadline_s: float) -> bytes:
    out = bytearray()
    t0 = time.time()
    while len(out) < n:
        if time.time() - t0 > deadline_s:
            break
        chunk = ser.read(n - len(out))
        if chunk:
            out.extend(chunk)
    return bytes(out)

def phasing_ok(record: bytes) -> bool:
    if len(record) != RECORD_LEN:
        return False
    if (record[0] & 0x80) == 0:
        return False
    for b in record[1:]:
        if (b & 0x80) != 0:
            return False
    return True

def read_one_position_angles_record(ser, timeout_s: float) -> bytes:
    ser.write(bytes([POINT_CMD]))
    ser.flush()
    record = read_exact(ser, RECORD_LEN, timeout_s)
    if phasing_ok(record):
        return record
    buf = bytearray(record)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        b = ser.read(1)
        if b:
            buf.extend(b)
        while len(buf) >= RECORD_LEN:
            candidate = bytes(buf[:RECORD_LEN])
            if phasing_ok(candidate):
                return candidate
            del buf[0]
    raise TimeoutError("12바이트 POSITION/ANGLES 레코드를 phasing 기준으로 읽지 못했습니다.")

USD_STAGE_PATH = "./light_ware5.usd"
ROBOT_PRIM_PATH = "/World/v4_onlyarm_urdf_1_0"
URDF_PATH = "./only_arm_urdf_new/urdf/v4_onlyarm_urdf_1.0.urdf"
RIGHT_DESC_YAML = "./right_arm_robot_descriptor.yaml"
LEFT_DESC_YAML = "./left_arm_robot_descriptor.yaml"
RIGHT_EE_FRAME = "link7"
LEFT_EE_FRAME = "link14"

EE_SPEED_MPS = 0.10
ROT_SPEED_RPS = 0.8
CONTROL_FRAME = "world"
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

MAX_DELTA_PER_STEP = 0.2 #0.01
MAX_TARGET_OFFSET_FROM_START = np.array([0.60, 0.60, 0.60], dtype=np.float64) # [0.35, 0.35, 0.35]

FOB_ENABLED = True
FOB_PORT = "/dev/ttyUSB0"
FOB_BAUD = 115200
FOB_TIMEOUT = 0.05 #0.05
FOB_RANGE_IN = 36.0
FOB_SEND_SET_POS_ANGLES_ONCE = False
FOB_MODE_DEFAULT = "keyboard"


FOB_AXIS_ORDER = [0, 1, 2]
FOB_AXIS_SIGN = np.array([1.0, 1.0, -1.0], dtype=np.float64)
FOB_AXIS_SCALE_M_PER_CM = np.array([0.05, 0.05, 0.05], dtype=np.float64) # [0.002, 0.002, 0.002]
FOB_MAX_DELTA_FROM_REF = np.array([0.20, 0.20, 0.20], dtype=np.float64) # [0.20, 0.20, 0.20]
FOB_USE_RELATIVE_POSITION = True
FOB_WORLD_OFFSET = np.array([0.0, 0.0, 0.0], dtype=np.float64)
FOB_APPLY_ORIENTATION = False #False
FOB_ORI_SIGN = np.array([1.0, 1.0, 1.0], dtype=np.float64)
FOB_ORI_SCALE_RAD_PER_DEG = np.array([np.pi/180.0, np.pi/180.0, np.pi/180.0], dtype=np.float64)

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
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return np.array([w, x, y, z], dtype=np.float64)
    if r.shape == (3, 3):
        m = r
        tr = np.trace(m)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            w = 0.25 * S
            x = (m[2, 1] - m[1, 2]) / S
            y = (m[0, 2] - m[2, 0]) / S
            z = (m[1, 0] - m[0, 1]) / S
        else:
            if (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
                S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
                w = (m[2, 1] - m[1, 2]) / S
                x = 0.25 * S
                y = (m[0, 1] + m[1, 0]) / S
                z = (m[0, 2] + m[2, 0]) / S
            elif m[1, 1] > m[2, 2]:
                S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
                w = (m[0, 2] - m[2, 0]) / S
                x = (m[0, 1] + m[1, 0]) / S
                y = 0.25 * S
                z = (m[1, 2] + m[2, 1]) / S
            else:
                S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
                w = (m[1, 0] - m[0, 1]) / S
                x = (m[0, 2] + m[2, 0]) / S
                y = (m[1, 2] + m[2, 1]) / S
                z = 0.25 * S
        q = np.array([w, x, y, z], dtype=np.float64)
        q /= np.linalg.norm(q) + 1e-12
        return q
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
    s = np.sin(half)
    return np.array([np.cos(half), axis[0]*s, axis[1]*s, axis[2]*s], dtype=np.float64)

def unit(v):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n

def alignment_score(actual_delta, desired_delta):
    au = unit(actual_delta)
    du = unit(desired_delta)
    if np.linalg.norm(au) < 1e-12 or np.linalg.norm(du) < 1e-12:
        return 0.0
    return float(np.dot(au, du))

def clamp_vec(v, max_abs):
    v = np.asarray(v, dtype=np.float64).copy()
    return np.clip(v, -max_abs, max_abs)

def clamp_vec_per_axis(v, lim):
    v = np.asarray(v, dtype=np.float64).copy()
    lim = np.asarray(lim, dtype=np.float64)
    return np.minimum(np.maximum(v, -lim), lim)

def set_ik_iters(kin, iters: int):
    if hasattr(kin, "set_max_iterations"):
        kin.set_max_iterations(iters)
    elif hasattr(kin, "ccd_max_iterations"):
        kin.ccd_max_iterations = iters

def disable_robot_gravity(world: World, robot_prim_path: str) -> None:
    from pxr import Usd, UsdPhysics
    stage = world.stage
    root = stage.GetPrimAtPath(robot_prim_path)
    if not root or not root.IsValid():
        raise RuntimeError(f"Robot prim path not found: {robot_prim_path}")
    rigid_paths = []
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_paths.append(str(prim.GetPath()))
    if not rigid_paths:
        raise RuntimeError(f"No rigid bodies found under {robot_prim_path}")
    try:
        from isaacsim.core.prims import RigidPrimView
    except Exception:
        from omni.isaac.core.prims import RigidPrimView
    views = []
    for i, path in enumerate(rigid_paths):
        view = RigidPrimView(prim_paths_expr=path, name=f"robot_rigid_view_{i}", reset_xform_properties=False)
        world.scene.add(view)
        views.append(view)
    world.reset()
    for view in views:
        view.disable_gravities()
    print(f"[INFO] Gravity disabled for {len(rigid_paths)} rigid bodies under {robot_prim_path}")

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
    axis_quality: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    axis_response: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    fob_ref_xyz_cm: Optional[np.ndarray] = None
    fob_ref_angles_deg: Optional[np.ndarray] = None

class FOBBridge:
    def __init__(self):
        self.ser = None
        self.last_pose: Optional[PoseAngles] = None
        self.last_read_ok = False
    def open(self):
        if not FOB_ENABLED:
            return
        self.ser = open_serial(FOB_PORT, FOB_BAUD, FOB_TIMEOUT)
        print(f"[INFO] FOB serial opened: port={FOB_PORT}, baud={FOB_BAUD}, timeout={FOB_TIMEOUT}")
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        if FOB_SEND_SET_POS_ANGLES_ONCE:
            self.ser.write(bytes([POS_ANGLES_CMD]))
            self.ser.flush()
            time.sleep(0.05)
            self.ser.reset_input_buffer()
    def close(self):
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None
    def poll(self) -> Optional[PoseAngles]:
        if self.ser is None:
            return None
        try:
            record = read_one_position_angles_record(self.ser, timeout_s=max(0.01, FOB_TIMEOUT * 2)) # (0.2, FOB_TIMEOUT * 4)
            pose = parse_position_angles_record(record, max_range_in=FOB_RANGE_IN)
            self.last_pose = pose
            self.last_read_ok = True
            return pose
        except Exception:
            self.last_read_ok = False
            return self.last_pose
    def pose_to_world_delta(self, pose: PoseAngles, arm: ArmState) -> Optional[np.ndarray]:
        if pose is None:
            return None
        raw_xyz_cm = np.array([pose.x_cm, pose.y_cm, pose.z_cm], dtype=np.float64)
        mapped_cm = raw_xyz_cm[FOB_AXIS_ORDER]
        if arm.fob_ref_xyz_cm is None:
            arm.fob_ref_xyz_cm = mapped_cm.copy()
        delta_cm = mapped_cm - arm.fob_ref_xyz_cm if FOB_USE_RELATIVE_POSITION else mapped_cm.copy()
        delta_m = delta_cm * FOB_AXIS_SCALE_M_PER_CM * FOB_AXIS_SIGN
        delta_m = clamp_vec_per_axis(delta_m, FOB_MAX_DELTA_FROM_REF)
        delta_m = delta_m + FOB_WORLD_OFFSET
        return delta_m
    def pose_to_orientation_delta(self, pose: PoseAngles, arm: ArmState) -> Optional[np.ndarray]:
        if pose is None:
            return None
        raw_angles = np.array([pose.azimuth_deg, pose.elevation_deg, pose.roll_deg], dtype=np.float64)
        if arm.fob_ref_angles_deg is None:
            arm.fob_ref_angles_deg = raw_angles.copy()
        delta_deg = raw_angles - arm.fob_ref_angles_deg
        return delta_deg * FOB_ORI_SCALE_RAD_PER_DEG * FOB_ORI_SIGN


pressed = set()
should_quit = False
last_pressed = set()
input_iface = carb.input.acquire_input_interface()
keyboard = None
try:
    import omni.appwindow
    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard()
except Exception:
    keyboard = None
if keyboard is None and hasattr(input_iface, "get_keyboard"):
    keyboard = input_iface.get_keyboard()
if keyboard is None:
    raise RuntimeError("Keyboard device handle not found.")

def on_keyboard_event(event, *args, **kwargs):
    global should_quit
    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
        pressed.add(event.input)
        if event.input == carb.input.KeyboardInput.ESCAPE:
            should_quit = True
    elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
        pressed.discard(event.input)

sub = input_iface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)
def is_down(key):
    return key in pressed

if USD_STAGE_PATH:
    open_stage(USD_STAGE_PATH)
world = World()
world.scene.add_default_ground_plane()
robot = Articulation(ROBOT_PRIM_PATH)
world.scene.add(robot)
world.reset()

if DISABLE_GRAVITY:
    disable_robot_gravity(world, ROBOT_PRIM_PATH)

controller = robot.get_articulation_controller()
num_dof = robot.num_dof
if hasattr(controller, "set_gains"):
    controller.set_gains(np.ones(num_dof) * KP, np.ones(num_dof) * KD)

jp0 = robot.get_joint_positions()
hold_action = ArticulationAction(joint_positions=jp0, joint_velocities=np.zeros_like(jp0))
dt = 1.0 / 60.0
for _ in range(int(max(0.0, HOLD_SECONDS) / dt)):
    controller.apply_action(hold_action)
    world.step(render=False)

right_kin = LulaKinematicsSolver(robot_description_path=RIGHT_DESC_YAML, urdf_path=URDF_PATH)
left_kin = LulaKinematicsSolver(robot_description_path=LEFT_DESC_YAML, urdf_path=URDF_PATH)
set_ik_iters(right_kin, 80)
set_ik_iters(left_kin, 80)
right_ik = ArticulationKinematicsSolver(robot, right_kin, RIGHT_EE_FRAME)
left_ik = ArticulationKinematicsSolver(robot, left_kin, LEFT_EE_FRAME)

base_t, base_q_raw = robot.get_world_pose()
base_q_raw = np.array(base_q_raw, dtype=np.float64)
base_q_as_is = base_q_raw
base_q_wxyz = to_wxyz_from_xyzw(base_q_raw)
if FORCE_BASE_WXYZ_SWAP:
    base_q_use = base_q_wxyz
    print("[BASE QUAT] forced: wxyz_swap")
else:
    right_kin.set_robot_base_pose(base_t, base_q_as_is)
    right_pos0, right_rot0 = right_ik.compute_end_effector_pose()
    right_quat0 = as_wxyz_quat(right_rot0)
    test_pos = np.array(right_pos0, dtype=np.float64); test_pos[2] += 0.001
    def try_base(kin, ik, q):
        kin.set_robot_base_pose(base_t, q)
        _, ok = ik.compute_inverse_kinematics(test_pos, right_quat0)
        return ok
    ok_as_is = try_base(right_kin, right_ik, base_q_as_is)
    ok_wxyz = try_base(right_kin, right_ik, base_q_wxyz)
    base_q_use = base_q_wxyz if (ok_wxyz and not ok_as_is) else base_q_as_is

right_kin.set_robot_base_pose(base_t, base_q_use)
left_kin.set_robot_base_pose(base_t, base_q_use)

arms = {
    "right": ArmState(name="right", kin=right_kin, ik=right_ik, frame_name=RIGHT_EE_FRAME),
    "left": ArmState(name="left", kin=left_kin, ik=left_ik, frame_name=LEFT_EE_FRAME),
}
active_arm = "right"
active_control_target = "both"  # "right", "left", "both"
control_mode = FOB_MODE_DEFAULT

for arm in arms.values():
    pos, rot = arm.ik.compute_end_effector_pose()
    arm.target_pos = np.array(pos, dtype=np.float64)
    arm.target_quat = as_wxyz_quat(rot)
    arm.start_target_pos = arm.target_pos.copy()


def get_active_state():
    return arms[active_arm]

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
        controller.apply_action(action if ok else hold_action)
        world.step(render=False)
    return ok_last

def restore_pose_for_arm(arm: ArmState, start_pos, start_quat):
    step_to_target_for_arm(arm, start_pos, start_quat, steps=ALIGN_RESTORE_STEPS)

def apply_axiswise_correction(arm: ArmState, desired_delta_world):
    corrected = np.zeros(3, dtype=np.float64)
    for i in range(3):
        corrected[i] = arm.axis_sign[i] * arm.axis_scale[i] * desired_delta_world[i] if arm.axis_enabled[i] else 0.0
    return corrected

def probe_axiswise_alignment_for_arm(arm_name: str):
    arm = arms[arm_name]
    print(f"\n=== AXIS-WISE AUTO ALIGN PROBE ({arm.name.upper()} / {arm.frame_name}) ===")
    refresh_base_pose()
    start_pos, start_rot = arm.ik.compute_end_effector_pose()
    start_pos = np.array(start_pos, dtype=np.float64)
    start_quat = as_wxyz_quat(start_rot)

    responses = np.zeros((3, 3), dtype=np.float64)
    signs = np.ones(3, dtype=np.float64)
    scales = np.ones(3, dtype=np.float64)
    enabled = np.ones(3, dtype=bool)
    qualities = np.zeros(3, dtype=np.float64)

    axes = [("X", np.array([ALIGN_PROBE_DIST, 0.0, 0.0], dtype=np.float64), 0),
            ("Y", np.array([0.0, ALIGN_PROBE_DIST, 0.0], dtype=np.float64), 1),
            ("Z", np.array([0.0, 0.0, ALIGN_PROBE_DIST], dtype=np.float64), 2)]
    for label, desired, idx in axes:
        goal_pos = start_pos + desired
        ok = step_to_target_for_arm(arm, goal_pos, start_quat, steps=ALIGN_SETTLE_STEPS)
        cur_pos, _ = arm.ik.compute_end_effector_pose()
        cur_pos = np.array(cur_pos, dtype=np.float64)
        actual = cur_pos - start_pos
        responses[:, idx] = actual
        comp = actual[idx]
        sign = 1.0 if comp >= 0.0 else -1.0
        mag = abs(comp)
        align = alignment_score(actual, desired)
        enable = (align >= MIN_ALIGN_TO_ENABLE) and (np.linalg.norm(actual) >= MIN_RESPONSE_NORM)
        scale = min(3.0, ALIGN_PROBE_DIST / mag) if mag > 1e-9 else 1.0
        signs[idx] = sign; scales[idx] = scale; enabled[idx] = enable; qualities[idx] = align
        print(f"[{arm.name.upper()}] axis {label}: desired={fmt(desired)} actual={fmt(actual)} "
              f"align={align:+.4f} main_comp={comp:+.5f} sign={sign:+.0f} scale={scale:.4f} enabled={enable} ok={ok}")
        restore_pose_for_arm(arm, start_pos, start_quat)

    arm.axis_sign = signs
    arm.axis_scale = scales
    arm.axis_enabled = enabled
    arm.axis_quality = qualities
    arm.axis_response = responses

    cur_pos, cur_rot = arm.ik.compute_end_effector_pose()
    arm.target_pos = np.array(cur_pos, dtype=np.float64)
    arm.target_quat = as_wxyz_quat(cur_rot)
    arm.start_target_pos = arm.target_pos.copy()
    print(f"[{arm.name.upper()}] axis_enabled={arm.axis_enabled} axis_sign={fmt(arm.axis_sign)} axis_scale={fmt(arm.axis_scale)}")
    print(f"=== {arm.name.upper()} AXIS-WISE AUTO ALIGN DONE ===\n")

def probe_both_arms():
    probe_axiswise_alignment_for_arm("right")
    probe_axiswise_alignment_for_arm("left")

fob = FOBBridge()
#global FOB_ENABLED
if FOB_ENABLED:
    try:
        fob.open()
    except Exception as e:
        print(f"[WARN] FOB open 실패: {e}")
        FOB_ENABLED = False
        control_mode = "keyboard"

if USE_AXIS_AUTO_ALIGN:
    probe_both_arms()

print("Controls:")
print("  TAB: switch active arm view  |  1/2/3: control right/left/both arms")
print("  K: keyboard mode        |  B: FOB mode")
print("  C: capture FOB reference for current arm")
print("  Arrow keys: translate (X/Y). Shift+Up/Down: Z   [keyboard mode]")
print("  I/K: pitch +/-,  J/L: yaw +/-,  U/O: roll +/-   [keyboard mode]")
print("  F: toggle CONTROL_FRAME (world/local)")
print("  R: reset current arm target pose to current EE pose")
print("  M: toggle axis auto alignment on/off for current arm")
print("  T: re-run axis auto alignment for current arm")
print("  G: re-run axis auto alignment for both arms")
print("  ESC: quit")
print(f"  CONTROL_FRAME={CONTROL_FRAME}, ACTIVE_ARM={active_arm}, MODE={control_mode}")
print("  Demo auto motion is embedded. Replace build_demo_dualarm_sequences() with the exact previous .ino path.")
print(f"  FOB_AXIS_ORDER={FOB_AXIS_ORDER}, FOB_AXIS_SIGN={fmt(FOB_AXIS_SIGN)}, FOB_AXIS_SCALE_M_PER_CM={fmt(FOB_AXIS_SCALE_M_PER_CM)}")

t = 0.0
log_counter = 0
while simulation_app.is_running() and not should_quit:
    world.step(render=True)
    t += dt
    log_counter += 1

    newly_pressed = pressed - last_pressed
    last_pressed = set(pressed)

    if carb.input.KeyboardInput.TAB in newly_pressed:
        active_arm = "left" if active_arm == "right" else "right"
        print(f"[INFO] Active arm switched to: {active_arm.upper()}")
    if carb.input.KeyboardInput.KEY_1 in newly_pressed:
        active_arm = "right"; print(f"[INFO] Active arm switched to: RIGHT")
    if carb.input.KeyboardInput.KEY_2 in newly_pressed:
        active_arm = "left"; print(f"[INFO] Active arm switched to: LEFT")

    if carb.input.KeyboardInput.K in newly_pressed:
        control_mode = "keyboard"; print("[INFO] Control mode -> KEYBOARD")
    if carb.input.KeyboardInput.B in newly_pressed:
        if FOB_ENABLED:
            control_mode = "fob"; print("[INFO] Control mode -> FOB")
        else:
            print("[WARN] FOB 비활성 상태")
    if carb.input.KeyboardInput.H in newly_pressed:
        control_mode = "auto_motion"; print("[INFO] Control mode -> AUTO_MOTION")
    if carb.input.KeyboardInput.J in newly_pressed:
        arm = get_active_state()
        set_auto_motion_enabled(arm, not arm.auto_motion_active)
        print(f"[INFO] {arm.name.upper()} auto motion -> {arm.auto_motion_active}")
    if carb.input.KeyboardInput.N in newly_pressed:
        new_state = not (arms["right"].auto_motion_active and arms["left"].auto_motion_active)
        set_auto_motion_enabled(arms["right"], new_state)
        set_auto_motion_enabled(arms["left"], new_state)
        print(f"[INFO] BOTH ARM auto motion -> {new_state}")

    if carb.input.KeyboardInput.C in newly_pressed:
        arm = get_active_state()
        pose = fob.poll() if FOB_ENABLED else None
        if pose is not None:
            arm.fob_ref_xyz_cm = np.array([pose.x_cm, pose.y_cm, pose.z_cm], dtype=np.float64)[FOB_AXIS_ORDER].copy()
            arm.fob_ref_angles_deg = np.array([pose.azimuth_deg, pose.elevation_deg, pose.roll_deg], dtype=np.float64).copy()
            print(f"[INFO] {arm.name.upper()} FOB reference captured | pos_cm={fmt(arm.fob_ref_xyz_cm)} angles_deg={fmt(arm.fob_ref_angles_deg)}")
        else:
            print("[WARN] FOB 데이터를 아직 읽지 못했습니다.")

    if carb.input.KeyboardInput.F in newly_pressed:
        CONTROL_FRAME = "local" if CONTROL_FRAME == "world" else "world"
        print(f"[INFO] CONTROL_FRAME switched to: {CONTROL_FRAME}")

    if carb.input.KeyboardInput.M in newly_pressed:
        arm = get_active_state()
        arm.axis_align_enabled = not arm.axis_align_enabled
        print(f"[INFO] {arm.name.upper()} AXIS_AUTO_ALIGN -> {arm.axis_align_enabled}")

    if carb.input.KeyboardInput.T in newly_pressed:
        probe_axiswise_alignment_for_arm(active_arm)
    if carb.input.KeyboardInput.G in newly_pressed:
        probe_both_arms()

    refresh_base_pose()
    arm = get_active_state()
    
    control_arms = [arms["right"], arms["left"]] if active_control_target == "both" else [arms[active_control_target]]

    desired_delta_world = np.zeros(3, dtype=np.float64)

    if control_mode == "keyboard":
        d = EE_SPEED_MPS * dt
        shift = is_down(carb.input.KeyboardInput.LEFT_SHIFT) or is_down(carb.input.KeyboardInput.RIGHT_SHIFT)
        if is_down(carb.input.KeyboardInput.UP):
            desired_delta_world[2 if shift else 0] += d
        if is_down(carb.input.KeyboardInput.DOWN):
            desired_delta_world[2 if shift else 0] -= d
        if is_down(carb.input.KeyboardInput.LEFT):
            desired_delta_world[1] += d
        if is_down(carb.input.KeyboardInput.RIGHT):
            desired_delta_world[1] -= d
    elif control_mode == "fob" and FOB_ENABLED:
        pose = fob.poll()
        delta_from_fob = fob.pose_to_world_delta(pose, arm)
        if delta_from_fob is not None:
            desired_delta_world = delta_from_fob
        if FOB_APPLY_ORIENTATION and pose is not None:
            delta_rpy = fob.pose_to_orientation_delta(pose, arm)
            if delta_rpy is not None:
                roll, pitch, yaw = delta_rpy[2], delta_rpy[1], delta_rpy[0]
                if abs(roll) > 1e-9:
                    dq = quat_from_axis_angle([1, 0, 0], roll)
                    arm.target_quat = quat_norm(quat_mul(dq, arm.target_quat)) if CONTROL_FRAME == "world" else quat_norm(quat_mul(arm.target_quat, dq))
                if abs(pitch) > 1e-9:
                    dq = quat_from_axis_angle([0, 1, 0], pitch)
                    arm.target_quat = quat_norm(quat_mul(dq, arm.target_quat)) if CONTROL_FRAME == "world" else quat_norm(quat_mul(arm.target_quat, dq))
                if abs(yaw) > 1e-9:
                    dq = quat_from_axis_angle([0, 0, 1], yaw)
                    arm.target_quat = quat_norm(quat_mul(dq, arm.target_quat)) if CONTROL_FRAME == "world" else quat_norm(quat_mul(arm.target_quat, dq))

    #desired_delta_world = clamp_vec(desired_delta_world, MAX_DELTA_PER_STEP)

    #if np.linalg.norm(desired_delta_world) > 0:
    #    command_delta_world = apply_axiswise_correction(arm, desired_delta_world) if arm.axis_align_enabled else desired_delta_world.copy()
    #    command_delta_world = clamp_vec(command_delta_world, MAX_DELTA_PER_STEP)
    
    if control_mode == "keyboard":
    	desired_delta_world = clamp_vec(desired_delta_world, MAX_DELTA_PER_STEP)
    elif control_mode == "fob":
        desired_delta_world = clamp_vec_per_axis(desired_delta_world, FOB_MAX_DELTA_FROM_REF)

    if np.linalg.norm(desired_delta_world) > 0:
        command_delta_world = apply_axiswise_correction(arm, desired_delta_world) if arm.axis_align_enabled else desired_delta_world.copy()

        if control_mode == "keyboard":
            command_delta_world = clamp_vec(command_delta_world, MAX_DELTA_PER_STEP)
        elif control_mode == "fob":
            command_delta_world = clamp_vec_per_axis(command_delta_world, FOB_MAX_DELTA_FROM_REF)
        
        if CONTROL_FRAME == "local":
            w, x, y, z = arm.target_quat
            R_now = np.array([
                [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
                [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
                [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y],
            ], dtype=np.float64)
            arm.target_pos += R_now @ command_delta_world
        else:
            arm.target_pos = arm.start_target_pos + command_delta_world if (control_mode == "fob" and FOB_USE_RELATIVE_POSITION) else arm.target_pos + command_delta_world
        min_pos = arm.start_target_pos - MAX_TARGET_OFFSET_FROM_START
        max_pos = arm.start_target_pos + MAX_TARGET_OFFSET_FROM_START
        arm.target_pos = np.minimum(np.maximum(arm.target_pos, min_pos), max_pos)
    #control_arms = [arms["right"], arms["left"]] if active_control_target == "both" else [arms[active_control_target]]
    if carb.input.KeyboardInput.R in newly_pressed:
        for target_arm in control_arms:
            cur_pos, cur_rot = target_arm.ik.compute_end_effector_pose()
            target_arm.target_pos = np.array(cur_pos, dtype=np.float64)
            target_arm.target_quat = as_wxyz_quat(cur_rot)
            target_arm.start_target_pos = target_arm.target_pos.copy()
        print(f"[INFO] target pose reset for: {active_control_target.upper()}")

    if control_mode == "keyboard":
        da = ROT_SPEED_RPS * dt
        roll = (is_down(carb.input.KeyboardInput.U) - is_down(carb.input.KeyboardInput.O)) * da
        pitch = (is_down(carb.input.KeyboardInput.I) - is_down(carb.input.KeyboardInput.K)) * da
        yaw = (is_down(carb.input.KeyboardInput.J) - is_down(carb.input.KeyboardInput.L)) * da
        for target_arm in control_arms:
            if abs(roll) > 0:
                dq = quat_from_axis_angle([1, 0, 0], roll)
                target_arm.target_quat = quat_norm(quat_mul(dq, target_arm.target_quat)) if CONTROL_FRAME == "world" else quat_norm(quat_mul(target_arm.target_quat, dq))
            if abs(pitch) > 0:
                dq = quat_from_axis_angle([0, 1, 0], pitch)
                target_arm.target_quat = quat_norm(quat_mul(dq, target_arm.target_quat)) if CONTROL_FRAME == "world" else quat_norm(quat_mul(target_arm.target_quat, dq))
            if abs(yaw) > 0:
                dq = quat_from_axis_angle([0, 0, 1], yaw)
                target_arm.target_quat = quat_norm(quat_mul(dq, target_arm.target_quat)) if CONTROL_FRAME == "world" else quat_norm(quat_mul(target_arm.target_quat, dq))

    # Apply IK to the viewed active arm every frame, and also to the other arm when controlling both
    apply_order = [arm]
    if active_control_target == "both":
        apply_order = [arms["right"], arms["left"]]
    for target_arm in apply_order:
        action, ok = target_arm.ik.compute_inverse_kinematics(target_arm.target_pos, target_arm.target_quat)
        controller.apply_action(action if ok else hold_action)

    if log_counter % 60 == 0:
        cur_pos, _ = arm.ik.compute_end_effector_pose()
        cur_pos = np.array(cur_pos, dtype=np.float64)
        pos_err = np.linalg.norm(cur_pos - arm.target_pos)
        if FOB_ENABLED and fob.last_pose is not None:
            raw_xyz = np.array([fob.last_pose.x_cm, fob.last_pose.y_cm, fob.last_pose.z_cm], dtype=np.float64)
            mapped_xyz = raw_xyz[FOB_AXIS_ORDER]
            fob_info = f" FOB_cm={fmt(mapped_xyz)}"
        else:
            fob_info = ""
        print(f"[status] ok={ok} view_arm={arm.name.upper()} control_target={active_control_target.upper()} mode={control_mode.upper()} axis_align={arm.axis_align_enabled} "
              f"target={fmt(arm.target_pos)} current={fmt(cur_pos)} err={pos_err:.5f} enabled={arm.axis_enabled}{fob_info}")

fob.close()
simulation_app.close()
