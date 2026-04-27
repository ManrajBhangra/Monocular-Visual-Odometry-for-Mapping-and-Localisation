import cv2
import numpy as np
import time
import keyboard
import sys
import threading
import SensorPoller
import Driver
import PositionTracker
import FrameLogger
import Mapper

ip_address = "192.168.0.9"
cam_url = f"http://{ip_address}/cam0"
omnidrive_url = f"http://{ip_address}/data/omnidrive"
odo_url = f"http://{ip_address}/data/odometry"
features_file = "robotino_features.pkl"
frames_dir = "robotino_frames"
frames_file = "robotino_frame_log.pkl"

mov_spe = 0.1   # 0.1m/s
turn_spe = 0.25   # 0.25 rad/s
dist_tol = 0.005  # metre
ang_tol = 0.03
mov_dis = 0.40   # distance between stations
odo_pause_time = 3 

#Found through camera calibration
K = np.array([[617.38495106,   0.0,         313.36845276],
              [  0.0,         616.65160065, 249.14195857],
              [  0.0,           0.0,           1.0      ]], dtype=np.float32)

dist_coeffs = np.array([[ 1.40800417e-01, -4.46755128e-01,
                           2.66829535e-03,  2.96997280e-04,
                           7.94140375e-01]], dtype=np.float32)
tilt = np.radians(2)

def make_cam_to_robot(pitch_deg):
    p  = np.radians(pitch_deg)
    Rx = np.array([[1,      0,       0     ],
                   [0,  np.cos(p), np.sin(p)],
                   [0, -np.sin(p), np.cos(p)]], dtype=np.float64)
    P  = np.array([[ 0,  0,  1],
                   [-1,  0,  0],
                   [ 0, -1,  0]], dtype=np.float64)
    return np.dot(P, Rx)

#factor in camera tilt
cam_cal_fin = make_cam_to_robot(tilt)

#Emergency Stop
STOP_EVENT = threading.Event()
driver = None

def wait_for_emergency():
    keyboard.wait('space')
    STOP_EVENT.set()
    if driver is not None:
        driver.full_stop()
    print("\nEMERGENCY STOP CALLED!")

def drive(x, y, angle):
    if driver is not None:
        driver.drive(x, y, angle)
        
live_orb = cv2.ORB_create(nfeatures=500)

def live_view(img):
    if img is None:
        return np.zeros((360, 640, 3), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp = live_orb.detect(gray, None)
    return cv2.drawKeypoints(img, kp, None, color=(0, 255, 0),flags=0)

def odo_pause(poller):
    drive(0, 0, 0)
    odo_fails, _ = poller.get_fails()
    print("Odometry Connection Lost")
    print("Attempting to reconnect")
    while not STOP_EVENT.is_set():
        time.sleep(0.1)
        odo_fails, _ = poller.get_fails()
        if odo_fails < odo_pause_time:
            print(f"Odometry Resumed")
            return True
    return False

def turn_to_heading(target_phi, tracker, logger, poller, label="Turning", log_frames=True):
    tracker.mode = "rotating"
    print(f"\n  [{label}] → {np.degrees(target_phi):.1f}"
          + ("" if log_frames else "no frame logging"))

    while not STOP_EVENT.is_set():
        x, y, phi, img = poller.get()
        if x is None:
            time.sleep(0.05)
            continue

        odo_f, _ = poller.get_fails()
        if odo_f >= odo_pause_time:
            if not odo_pause(poller):
                return False
            continue

        tracker.update(x, y, phi)
        logger.update_pose(tracker.x, tracker.y, tracker.phi, tracker.mode, odo_x=x, odo_y=y)
        if log_frames and img is not None:
            logger.update_map(img)
        cv2.imshow("Robotino Mapping", live_view(img))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            STOP_EVENT.set()
            break

        error = (target_phi - phi + np.pi) % (2 * np.pi) - np.pi
        sys.stdout.write(f"\rheading error: {np.degrees(error):+.1f}   ")
        sys.stdout.flush()

        if abs(error) < ang_tol:
            drive(0, 0, 0)
            print(f"\ndone.")
            return True

        if abs(error) < 0.1:
            cmd = np.clip(0.5 * error, -0.15, 0.15)
        else:
            cmd = np.clip(1.5 * error, -turn_spe, turn_spe)
        drive(0, 0, cmd)
        time.sleep(0.05)

    drive(0, 0, 0)
    return False


def turn_by(delta_phi, tracker, logger, poller, label="Turning", log_frames=True):
    _, _, phi, _ = poller.get()
    if phi is None or STOP_EVENT.is_set():
        return False
    target = (phi + delta_phi + np.pi) % (2 * np.pi) - np.pi
    return turn_to_heading(target, tracker, logger, poller, label=label, log_frames=log_frames)


def move_forward(distance, tracker, logger, poller, label="Moving Fwd"):
    tracker.mode = "moving"
    MIN_FRAME_DIST = 0.02   # metres between logged frames

    # Wait for first valid odometry
    while not STOP_EVENT.is_set():
        x, y, phi, img = poller.get()
        if x is not None:
            break
        time.sleep(0.05)

    if STOP_EVENT.is_set():
        return False

    start_x, start_y = x, y
    tracker.update(x, y, phi)

    last_logged_wx = start_x   # raw encoder coords for spacing gate
    last_logged_wy = start_y

    while not STOP_EVENT.is_set():
        x, y, phi, img = poller.get()
        if x is None:
            time.sleep(0.05)
            continue

        odo_f, _ = poller.get_fails()
        if odo_f >= odo_pause_time:
            if not odo_pause(poller):
                return False
            continue

        tracker.update(x, y, phi)
        wx, wy = tracker.x, tracker.y
        logger.update_pose(wx, wy, tracker.phi, tracker.mode,
                           odo_x=x, odo_y=y)

        dist_since_last = np.hypot(x - last_logged_wx, y - last_logged_wy)
        if img is not None and dist_since_last >= MIN_FRAME_DIST:
            logger.update_map(img)
            last_logged_wx = x
            last_logged_wy = y

        cv2.imshow("Robotino Mapping", live_view(img))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            STOP_EVENT.set()
            break

        travelled = np.hypot(x - start_x, y - start_y)
        remaining = distance - travelled
        sys.stdout.flush()

        if remaining <= dist_tol:
            drive(0, 0, 0)
            return True

        drive(np.clip(2.0 * remaining, 0.05, mov_spe), 0, 0)
        time.sleep(0.05)

    drive(0, 0, 0)
    return False


def scan_360(tracker, logger, poller, label="360 Scan"):
    x, y, phi, img = poller.get()
    if phi is None or STOP_EVENT.is_set():
        return False
    print(f"\n [{label}] starting: {np.degrees(phi):.1f}")
    logger.log_keyframe(img, x, y, phi)
    for step_label in ["Clockwise 180 - front", "Clockwise 180 - rear"]:
        if not turn_by(-np.pi, tracker, logger, poller,
                       label=f"{label} > {step_label}", log_frames=False):
            return False
    print(f"{label} complete full 360 scanned.")
    return True

def run_square_exploration(tracker, logger, poller):
    
    _, _, start_phi, _ = poller.get()
    if start_phi is None:
        print("Error could not find initial direction")
        return False

    print("Exploration Beginning")
    print("SPACE = emergency stop")

    steps = [
        ("Station 1", None, None, True),
        ("Leg 1", mov_dis, None, True),
        ("Station 2",  None, None, True),
        ("Turn → S3", None, -np.pi / 2, False),
        ("Leg 2", mov_dis, None, True),
        ("Station 3", None, None, True),
        ("Turn → S4", None, -np.pi / 2, False),
        ("Leg 3", mov_dis, None, True),
        ("Station 4", None, None, True),
        ("Turn → return", None, -np.pi / 2, False),
        ("Leg 4", mov_dis, None, True),
        ("Station 1 (return)", None, None, True),
    ]

    for label, fwd, turn, log in steps:
        if STOP_EVENT.is_set():
            return False
        if fwd is not None:
            print(f"\n─── {label} ───")
            if not move_forward(fwd, tracker, logger, poller, label=label):
                return False
            time.sleep(0.5)
        elif turn is not None:
            if not turn_by(turn, tracker, logger, poller, label=label,
                           log_frames=log):
                return False
            time.sleep(0.5)
        else:
            print(f"\n═══ {label} ═══")
            if not scan_360(tracker, logger, poller, label=label):
                return False

    if not turn_to_heading(start_phi, tracker, logger, poller, label="Reset heading", log_frames=False):
        return False
    return True

def main():
    global driver
    driver = Driver.Driver(omnidrive_url)

    poller = SensorPoller.SensorPoller(odo_url, cam_url)
    logger = FrameLogger.FrameLogger(frames_dir, frames_file)
    threading.Thread(target=wait_for_emergency, daemon=True).start()

    print("Press SPACE in emergency")

    print("Waiting for first sensor readings")
    deadline = time.time() + 5.0
    while time.time() < deadline:
        x, y, phi, _ = poller.get()
        if x is not None and y is not None and phi is not None:
            break
        time.sleep(0.05)
    else:
        print("ERROR: Could not connect")
        poller.stop()
        return

    print("Successfully connected")

    tracker = PositionTracker.PositionTracker()

    time.sleep(0.5)

    try:
        success = run_square_exploration(tracker, logger, poller)
        if success:
            print("Exploration complete.")
        else:
            print("Mapping Failed")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        drive(0, 0, 0)
        poller.stop()
        logger.save()
        driver.full_stop()
        driver.shutdown()
        cv2.destroyAllWindows()
        if STOP_EVENT.is_set():
            print("Emergency Stop Executed")
    
    print("Creating map")
    Mapper.Mapper(K, dist_coeffs, features_file, cam_cal_fin).build_from_log(frames_file)
    print("Map Created")


if __name__ == "__main__":
    main()