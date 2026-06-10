import numpy as np

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.utils.stage import open_stage
from omni.isaac.core.utils.types import ArticulationAction

try:
    from isaacsim.core.api.articulations import Articulation
except Exception:
    from omni.isaac.core.articulations import Articulation  # Isaac Sim 4.x fallback

import carb

from rb20_kinematics_py import (
    TRobotKinematics,
    TJointVar,
    TTaskVar,
    robot_init,
    robot_kinematicsf,
    robot_task_contol,
    get_homogeneous,
)

# -------------------------
# User settings
# -------------------------
USD_STAGE_PATH = "./light_ware5.usd"
ROBOT_PRIM_PATH = "/World/v4_onlyarm_urdf_1_0"

# Set to the exact right-arm joint names if available.
# If left empty or not found, the script falls back to the first 6 DOFs.
RIGHT_ARM_JOINT_NAMES = [
    # "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
]

KP = 1000.0
KD = 1200.0
HOLD_SECONDS = 1.0
DEFAULT_DT = 1.0 / 60.0

# Same oscillation logic as the original Teensy/RB20 test.
XV_NEG_Y = np.array([0.0, -0.5, 0.0, 0.0, 0.0, 0.0], dtype=float)
XV_POS_Y = np.array([0.0,  0.5, 0.0, 0.0, 0.0, 0.0], dtype=float)
Y_UPPER_LIMIT = 1.5
Y_LOWER_LIMIT = -1.0

# If the articulation world pose already matches your previous RB20 setup,
# leave this False and the base offset will be read from Isaac.
USE_FIXED_BASE_OFFSET = False
FIXED_BASE_OFFSET = np.array([0.0, 0.3, 1.5, 0.0, 0.0, np.pi], dtype=float)

# -------------------------
# Quaternion / rotation helpers
# -------------------------
def quat_xyzw_to_rotmat(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(q_xyzw, dtype=float)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def rotmat_to_rpy(R: np.ndarray) -> np.ndarray:
    r11_r12 = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    pitch = np.arctan2(-R[2, 0], r11_r12)

    if np.deg2rad(89.5) < abs(pitch) < np.deg2rad(90.5):
        if pitch > 0:
            pitch = np.deg2rad(90.0)
            yaw = 0.0
            roll = np.arctan2(R[0, 1], R[1, 1])
        else:
            pitch = np.deg2rad(-90.0)
            yaw = 0.0
            roll = -np.arctan2(R[0, 1], R[1, 1])
    else:
        cb = np.cos(pitch)
        yaw = np.arctan2(R[1, 0] / cb, R[0, 0] / cb)
        roll = np.arctan2(R[2, 1] / cb, R[2, 2] / cb)

    return np.array([roll, pitch, yaw], dtype=float)


def pose_to_base_offset(position: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    rot = quat_xyzw_to_rotmat(quat_xyzw)
    rpy = rotmat_to_rpy(rot)
    return np.array([position[0], position[1], position[2], rpy[0], rpy[1], rpy[2]], dtype=float)


# -------------------------
# Joint index resolution
# -------------------------
def _safe_get_dof_names(articulation):
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
    return None


def resolve_right_arm_indices(articulation):
    names = _safe_get_dof_names(articulation)
    num_dof = articulation.num_dof

    if names:
        print("[INFO] Isaac DOF names:", names)

    if names and RIGHT_ARM_JOINT_NAMES:
        missing = [name for name in RIGHT_ARM_JOINT_NAMES if name not in names]
        if not missing:
            indices = [names.index(name) for name in RIGHT_ARM_JOINT_NAMES]
            print("[INFO] Using configured RIGHT_ARM_JOINT_NAMES ->", indices)
            return indices
        print("[WARN] Some configured joint names were not found:", missing)

    if num_dof >= 6:
        print("[WARN] Falling back to the first 6 DOFs as the RB20 arm indices.")
        return list(range(6))

    raise RuntimeError(f"Articulation DOF count is {num_dof}, which is smaller than 6.")


# -------------------------
# Keyboard input (ESC only)
# -------------------------
should_quit = False
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
    raise RuntimeError("Keyboard device handle not found. (Isaac Sim input API mismatch)")


def on_keyboard_event(event, *args, **kwargs):
    global should_quit
    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
        if event.input == carb.input.KeyboardInput.ESCAPE:
            should_quit = True


sub = input_iface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)


# -------------------------
# Stage + World
# -------------------------
if USD_STAGE_PATH:
    open_stage(USD_STAGE_PATH)

world = World()
world.scene.add_default_ground_plane()

robot = Articulation(ROBOT_PRIM_PATH)
world.scene.add(robot)
world.reset()

controller = robot.get_articulation_controller()
num_dof = robot.num_dof

kps = np.ones(num_dof) * KP
kds = np.ones(num_dof) * KD
if hasattr(controller, "set_gains"):
    controller.set_gains(kps, kds)
else:
    print("[WARN] controller.set_gains not found (skip gains)")

jp0 = np.asarray(robot.get_joint_positions(), dtype=float)
hold_action = ArticulationAction(
    joint_positions=jp0,
    joint_velocities=np.zeros_like(jp0),
)

hold_steps = int(max(0.0, HOLD_SECONDS) / DEFAULT_DT)
for _ in range(hold_steps):
    controller.apply_action(hold_action)
    world.step(render=False)

right_indices = resolve_right_arm_indices(robot)
print("[INFO] Right-arm joint indices:", right_indices)

# -------------------------
# RB20 controller init
# -------------------------
rb20_robot = TRobotKinematics()
qv = TJointVar()

base_t, base_q = robot.get_world_pose()
base_t = np.asarray(base_t, dtype=float)
base_q = np.asarray(base_q, dtype=float)

if USE_FIXED_BASE_OFFSET:
    base_offset = FIXED_BASE_OFFSET.copy()
    print("[INFO] Using FIXED_BASE_OFFSET:", base_offset)
else:
    base_offset = pose_to_base_offset(base_t, base_q)
    print("[INFO] Using articulation world pose as RB20 base offset:", base_offset)

robot_init(rb20_robot, base_offset)
current_xv = TTaskVar(XV_NEG_Y.copy())

print("[INFO] Running RB20 task-space oscillation test.")
print("       ESC to quit.")

# -------------------------
# Main loop
# -------------------------
step_count = 0
while simulation_app.is_running() and not should_quit:
    world.step(render=True)
    step_count += 1

    # Read current state from Isaac
    q_all = np.asarray(robot.get_joint_positions(), dtype=float)
    q_right = q_all[right_indices].copy()

    # Compute FK with the current Isaac state
    robot_kinematicsf(rb20_robot, q_right)

    # Keep the original Teensy test logic: flip Y command at bounds
    ee_y = rb20_robot.Position[5].y
    if ee_y >= Y_UPPER_LIMIT:
        current_xv = TTaskVar(XV_NEG_Y.copy())
    elif ee_y <= Y_LOWER_LIMIT:
        current_xv = TTaskVar(XV_POS_Y.copy())

    ok = robot_task_contol(rb20_robot, qv, current_xv)
    if not ok and step_count % 60 == 0:
        print("[WARN] robot_task_contol failed; fallback value is being used.")

    # Use Isaac state as source of truth, then apply the next position target only to the RB20 arm joints.
    q_right_cmd = q_right + qv.vq[:6] * DEFAULT_DT
    arm_action = ArticulationAction(
        joint_positions=q_right_cmd,
        joint_indices=right_indices,
    )
    controller.apply_action(arm_action)

    if step_count % 60 == 0:
        ee_pose = get_homogeneous(rb20_robot.T).vx
        print(
            f"[STEP {step_count:05d}] "
            f"ee_pose={np.round(ee_pose, 4)} "
            f"xv={np.round(current_xv.vx, 3)} "
            f"dq={np.round(qv.vq[:6], 5)}"
        )

simulation_app.close()
