# measure_logger.py — reliable measurement & logging loop for Create 3
# - Reads IR sensors via .sensors[]
# - Polls bumpers
# - Logs in the schema your CPT learner expects

import asyncio, csv, time, os
from irobot_edu_sdk.backend.bluetooth import Bluetooth
from irobot_edu_sdk.robots import Create3, event
from irobot_edu_sdk.music import Note

ROBOT_BT = 'iRobot_25'  # or '00:16:A4:4B:5D:77'
DATA_FILE = "C:/Users/pyppr/Create3_BayesianFusion/data/raw/ir_pid_bumper_log.csv"

# PID (keep same constants as your main script to preserve columns)
Kp, Ki, Kd = 0.4, 0.02, 0.1
TARGET_DIST_M = 0.06

# Very gentle speeds (cm/s)
BASE_CM = 1.5
MAX_WHEEL_CM = 4.0

os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
robot = Create3(Bluetooth(ROBOT_BT))

last_err = 0.0
integ = 0.0

def clamp(v, lo, hi): return max(lo, min(hi, v))

@event(robot.when_play)
async def start(r):
    global last_err, integ
    print("🔵 Connecting...")
    await r.set_lights_on_rgb(0, 0, 255)
    await asyncio.sleep(0.5)
    print("✅ Connected.")
    await r.play_note(Note.C5, 0.25)
    await r.set_lights_on_rgb(0, 255, 0)

    # Try to undock; ignore if already off-dock
    try:
        await r.undock()
        await asyncio.sleep(2)
    except Exception:
        pass

    print("📏 Starting measurement loop...")
    await r.set_lights_on_rgb(0, 255, 0)

    with open(DATA_FILE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Time","IR_Left","IR_Right","PID_P","PID_I","PID_D","Bumper","Location"])
        t0 = time.time()

        for step in range(120):  # ~24 s @ 0.2s
            try:
                ir = await r.get_ir_proximity()
                s = getattr(ir, "sensors", None)
                if not s or len(s) < 5:
                    print("⚠️ IR read failed")
                    await asyncio.sleep(0.2)
                    continue

                # Use front-left (0) and front-right (4); change if your mapping differs
                ir_left, ir_right = float(s[0]), float(s[4])
                bump = await r.get_bumpers()
                bumped = bool(getattr(bump, "left", False) or getattr(bump, "right", False))

                # Simple “distance proxy”: average & scale to meters (if your IR is in cm)
                dist_m = (ir_left + ir_right) / 2.0 / 100.0

                # PID (keep identical math as your wall-follow script to keep columns meaningful)
                err = TARGET_DIST_M - dist_m
                integ += err
                deriv = err - last_err
                last_err = err

                # Optional: gentle drive to keep the robot moving similarly to your main run
                ctrl_cm = (Kp*err + Ki*integ + Kd*deriv) * 100.0
                left_cm  = clamp(BASE_CM - ctrl_cm, -MAX_WHEEL_CM, MAX_WHEEL_CM)
                right_cm = clamp(BASE_CM + ctrl_cm, -MAX_WHEEL_CM, MAX_WHEEL_CM)

                if bumped:
                    await r.set_lights_blink_rgb(255, 115, 0)
                    # small backoff: distances are in meters for move()
                    await r.move(-0.05)
                    await asyncio.sleep(0.2)
                    label = "Door_Start"
                else:
                    await r.set_lights_on_rgb(0, 255, 0)
                    label = "Wall"

                await r.set_wheel_speeds(left_cm, right_cm)

                # Log
                w.writerow([
                    round(time.time()-t0, 2),
                    round(ir_left, 3), round(ir_right, 3),
                    round(Kp*err, 4), round(Ki*integ, 4), round(Kd*deriv, 4),
                    bumped, label
                ])
            except Exception as e:
                print("⚠️ loop error:", e)

            await asyncio.sleep(0.2)

    # stop + dock
    await r.set_wheel_speeds(0, 0)
    await r.set_lights_on_rgb(255, 255, 255)
    await r.play_note(Note.A4, 0.3)
    try:
        await r.dock()
        await asyncio.sleep(2)
    except Exception:
        pass
    await r.set_lights_off()
    print(f"✅ Saved: {DATA_FILE}")

robot.play()
