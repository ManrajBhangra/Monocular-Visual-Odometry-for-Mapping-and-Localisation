import cv2
import numpy as np
import time
import keyboard
import sys
import threading
import Localiser
import SensorPoller
import Driver
import PositionTracker

ip_address = "192.168.0.9"
cam_url  = f"http://{ip_address}/cam0"
track_url = f"http://{ip_address}/data/omnidrive"
odo_url = f"http://{ip_address}/data/odometry"

mov_spe = 0.1
turn_spe = 0.25

STOP_EVENT = threading.Event()
driver = None


def wait_for_emergency():
    keyboard.wait('space')
    STOP_EVENT.set()
    if driver is not None:
        driver.full_stop()
    print("\nEMERGENCY STOP CALLED!")

def drive(vx, vy, angle):
    if driver is not None:
        driver.drive(vx, vy, angle)

def draw_hud(frame, tracker, odo_f, cam_f, keys_active):
    canvas = frame.copy() if frame is not None else np.zeros((360, 640, 3), dtype=np.uint8)

    def put(text, y, color=(0, 220, 100)):
        cv2.putText(canvas, text, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)

    put(f"Pose  x:{(tracker.x*100):+.1f}cm  y:{(tracker.y*100):+.1f}cm " f"phi:{np.degrees(tracker.phi):+.1f}deg", 22)
    put("Arrows=move  Z/X=rotate  SPACE=e-stop  Q=quit",
        canvas.shape[0] - 10, color=(180, 180, 180))
    return canvas

def main():
    global driver
    driver = Driver.Driver(track_url)

    poller = SensorPoller.SensorPoller(odo_url, cam_url)
    localiser = Localiser.Localiser()
    threading.Thread(target=wait_for_emergency, daemon=True).start()

    print("Robotino Keyboard Control")
    print("Arrows=move  Z/X=rotate  SPACE=e-stop  Q=quit")

    print("Waiting for first sensor readings...")
    deadline = time.time() + 5.0
    while time.time() < deadline:
        rx, ry, phi, _ = poller.get()
        if phi is not None:
            break
        time.sleep(0.05)
    else:
        print("ERROR: Failed to connect.")
        poller.stop()
        return

    print(f"Odometry online: x={rx:.1f}  y={ry:.1f}  phi={np.degrees(phi):.1f}")

    tracker = PositionTracker.PositionTracker()
    deadline = time.time() + 8.0
    while time.time() < deadline:
        _, _, phi_now, img = poller.get()
        if img is not None and phi_now is not None:
            loc = localiser.localise_best(img)
            if loc is not None:
                lx, ly, lphi = loc
                print(f"Start Position: x={lx:.1f}  y={ly:.1f}  "
                      f"phi={np.degrees(lphi):.1f}")
                tracker = PositionTracker.PositionTracker(lx, ly, lphi)
            else:
                print("Start Position (0,0)")
            break
        time.sleep(0.1)

    print(f"Starting pose: x={tracker.x:.1f}  y={tracker.y:.1f} phi={np.degrees(tracker.phi):.1f}\n")
    try:
        while True:
            if STOP_EVENT.is_set():
                break
            if keyboard.is_pressed('q') or (cv2.waitKey(1) & 0xFF == ord('q')):
                break

            rx, ry, phi, img = poller.get()
            if rx is None:
                time.sleep(0.05)
                continue

            tracker.update(rx, ry, phi)

            vx, vy, angle = 0.0, 0.0, 0.0
            keys = []
            if keyboard.is_pressed('up'): vx = mov_spe; keys.append("fwd")
            if keyboard.is_pressed('down'): vx = -mov_spe; keys.append("back")
            if keyboard.is_pressed('left'): vy =  mov_spe; keys.append("strafe-L")
            if keyboard.is_pressed('right'): vy = -mov_spe; keys.append("strafe-R")
            if keyboard.is_pressed('z'): angle =  turn_spe; keys.append("rot-L")
            if keyboard.is_pressed('x'): angle = -turn_spe; keys.append("rot-R")

            drive(vx, vy, angle)

            odo_f, cam_f = poller.get_fails()
            hud  = draw_hud(img, tracker, odo_f, cam_f, " + ".join(keys) if keys else "")
            cv2.imshow("Robotino Keyboard Control", hud)
            sys.stdout.write(
                f"\r  x:{tracker.x:+.1f}  y:{tracker.y:+.1f}"
                f"phi:{np.degrees(tracker.phi):+.1f}"
            )
            sys.stdout.flush()
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        drive(0, 0, 0)
        print("\nCalculating final position")
        _, _, phi_final, img_final = poller.get()
        if img_final is not None and phi_final is not None:
            loc = localiser.localise_best(img_final)
            if loc is not None:
                lx, ly, lphi = loc
                print(f"End position: x={lx:.1f}  y={ly:.1f} phi={lphi:.1f}")
                tracker.set_position(lx, ly, lphi)
        else:
            print("No camera frame available.")

        poller.stop()
        driver.full_stop()
        driver.shutdown()
        cv2.destroyAllWindows()
        print("Keyboard session ended.")


if __name__ == "__main__":
    main()