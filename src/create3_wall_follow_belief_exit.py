# ============================================================
# create3_wall_follow_belief_exit.py
# ------------------------------------------------------------
# iRobot Create 3 wall-follow with Bayesian "door" detection.
# Uses CPTs learned by compute_cpts.py via belief_module.BeliefEngine.
#
# Behavior:
#  - Undock, follow wall slowly using IR+PID
#  - Each tick, compute belief P(DoorPassed)
#  - If sustained high probability, turn INTO the gap (exit) and drive in
#  - Resume wall-follow in the new corridor
#
# Safety:
#  - Gentle speeds (cm/s), clamp wheels
#  - Bumper triggers brief backoff & recovery
#  - BLE and sensor reads wrapped in try/except
# ============================================================

import asyncio, csv, os, time
from collections import deque

from irobot_edu_sdk.backend.bluetooth import Bluetooth
from irobot_edu_sdk.robots import Create3, event
from irobot_edu_sdk.music import Note

from belief_module import BeliefEngine, BeliefConfig

# -------------------- CONFIG --------------------
ROBOT_BT = 'iRobot_25'  # or '00:16:A4:4B:5D:77'
RAW_LOG = "C:/Users/pyppr/Create3_BayesianFusion/data/raw/ir_pid_belief_exit_log.csv"
CPT_JSON = "C:/Users/pyppr/Create3_BayesianFusion/data/processed/cpt_tables.json"

# PID constants (same scale as earlier)
Kp, Ki, Kd = 0.4, 0.02, 0.1
TARGET_DIST_M = 0.06

# Wheel speed limits (cm/s)
BASE_CM = 1.6
MAX_WHEEL_CM = 5.0
MAX_DELTA_CM = 3.0

# IR indices (use your confirmed mapping)
IR_LEFT_IDX = 0
IR_RIGHT_IDX = 4

# Door decision logic
P_DOOR_THRESH = 0.65       # trigger threshold
SUSTAIN_COUNT = 4          # require this many consecutive ticks above threshold
COOLDOWN_SEC = 5.0         # minimum time between exits
EXIT_TURN_DEG = 85         # turn angle into doorway
EXIT_ADVANCE_M = 0.45      # drive forward into doorway
REALIGN_WAIT_S = 0.5

# ------------------------------------------------
# Helpers
# ------------------------------------------------
def clamp(v, lo, hi): return max(lo, min(hi, v))

# ------------------------------------------------
# Robot + belief engine
# ------------------------------------------------
robot = Create3(Bluetooth(ROBOT_BT))
eng = BeliefEngine(BeliefConfig(cpt_json=CPT_JSON))

last_error = 0.0
integral = 0.0

# Door trigger smoothing
pdoor_hist = deque(maxlen=SUSTAIN_COUNT)
last_exit_time = 0.0

# ------------------------------------------------
# Main behavior
# ------------------------------------------------
@event(robot.when_play)
async def start(r: Create3):
    global last_error, integral, last_exit_time

    os.makedirs(os.path.dirname(RAW_LOG), exist_ok=True)
    print("🔵 Connecting...")
    await r.set_lights_on_rgb(0, 0, 255)
    await asyncio.sleep(0.5)
    print("✅ Connected.")
    await r.play_note(Note.C5, 0.25)
    await r.set_lights_on_rgb(0, 255, 0)

    # Undock if possible
    try:
        await r.undock()
        await asyncio.sleep(2)
    except Exception:
        pass

    print("🤖 Wall-follow with Bayesian exit detection starting...")
    await r.set_lights_on_rgb(0, 255, 0)

    with open(RAW_LOG, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Time","IR_Left","IR_Right",
            "dist_m","Trend","Gap","P_Door",
            "P_W_near","P_W_ok","P_W_far",
            "bumped","state"
        ])

        t0 = time.time()
        state = "FOLLOW"

        # ~ 60 seconds (@ 0.2 s step) — adjust as needed
        for step in range(300):
            await asyncio.sleep(0.2)

            # ----------------- sense -----------------
            try:
                ir = await r.get_ir_proximity()
                s = getattr(ir, "sensors", None)
                if not s or len(s) <= max(IR_LEFT_IDX, IR_RIGHT_IDX):
                    print("⚠️ IR read failed, skipping")
                    continue

                ir_left  = float(s[IR_LEFT_IDX])
                ir_right = float(s[IR_RIGHT_IDX])

                bump = await r.get_bumpers()
                bumped = bool(getattr(bump, "left", False) or getattr(bump, "right", False))
            except Exception as e:
                print("⚠️ sensor error:", e)
                continue

            # ----------------- belief -----------------
            b = eng.belief({"IR_Left": ir_left, "IR_Right": ir_right, "Bumper": bumped})
            dist_m   = b["features"]["dist_m"]
            trend    = b["features"]["Trend"]
            gap      = b["features"]["GapDetected"]
            p_door   = b["P(DoorPassed)"]
            p_w_near = b["P(WallDist)"].get("near", 0.0)
            p_w_ok   = b["P(WallDist)"].get("ok", 0.0)
            p_w_far  = b["P(WallDist)"].get("far", 0.0)

            # smoothing
            pdoor_hist.append(p_door)
            sustained = len(pdoor_hist) == SUSTAIN_COUNT and all(p >= P_DOOR_THRESH for p in pdoor_hist)

            # ----------------- state machine -----------------
            if state == "FOLLOW":
                # PID wall-follow (slow and gentle)
                error = TARGET_DIST_M - dist_m
                integral += error
                derivative = error - last_error
                last_error = error

                ctrl_cm = (Kp*error + Ki*integral + Kd*derivative) * 100.0
                ctrl_cm = clamp(ctrl_cm, -MAX_DELTA_CM, MAX_DELTA_CM)

                left_cm  = clamp(BASE_CM - ctrl_cm, -MAX_WHEEL_CM, MAX_WHEEL_CM)
                right_cm = clamp(BASE_CM + ctrl_cm, -MAX_WHEEL_CM, MAX_WHEEL_CM)

                if bumped:
                    # backoff and keep following
                    await r.set_lights_blink_rgb(255, 115, 0)
                    await r.set_wheel_speeds(0, 0)
                    await r.move(-0.08)
                    await r.play_note(Note.C4, 0.2)
                else:
                    await r.set_lights_on_rgb(0, 255, 0)
                    await r.set_wheel_speeds(left_cm, right_cm)

                # Decide to take exit if sustained and cooldown passed
                now = time.time()
                if sustained and (now - last_exit_time) > COOLDOWN_SEC:
                    # Choose side with larger opening (bigger IR → farther)
                    turn_right = (ir_right > ir_left)
                    await r.set_wheel_speeds(0, 0)
                    await r.play_note(Note.A4, 0.2)
                    await r.set_lights_on_rgb(255, 255, 0)  # yellow = commit exit
                    if turn_right:
                        await r.turn_right(EXIT_TURN_DEG)
                    else:
                        await r.turn_left(EXIT_TURN_DEG)
                    state = "EXIT_COMMIT"
                    continue

            elif state == "EXIT_COMMIT":
                try:
                    await r.set_wheel_speeds(0, 0)
                    await r.move(EXIT_ADVANCE_M)     # drive into doorway
                    await asyncio.sleep(REALIGN_WAIT_S)
                    last_exit_time = time.time()
                    await r.play_note(Note.C6, 0.25)
                    await r.set_lights_on_rgb(0, 200, 255)  # cyan = inside exit
                finally:
                    state = "FOLLOW"  # resume following in new corridor

            # ----------------- log -----------------
            w.writerow([
                round(time.time()-t0, 2),
                round(ir_left, 3), round(ir_right, 3),
                round(dist_m, 4), trend, gap, round(p_door, 4),
                round(p_w_near, 4), round(p_w_ok, 4), round(p_w_far, 4),
                bumped, state
            ])

        # stop at end
        await r.set_wheel_speeds(0, 0)
        await r.set_lights_on_rgb(255, 255, 255)
        await r.play_note(Note.A4, 0.3)
        await asyncio.sleep(0.5)
        await r.set_lights_off()
        print(f"✅ Log saved: {RAW_LOG}")

robot.play()
