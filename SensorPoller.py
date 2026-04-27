import threading
import requests
import time
import cv2
import numpy as np

class SensorPoller:

    def __init__(self, odo_url, cam_url):
        self.lock = threading.Lock()
        self.x = self.y = self.phi = self.img = None
        self.odo_fails = self.camera_fails = 0
        self.running = True
        self.odo_url = odo_url
        self.cam_url = cam_url
        threading.Thread(target=self.poll_odo, daemon=True).start()
        threading.Thread(target=self.poll_cam, daemon=True).start()

    def poll_odo(self):
        session = requests.Session()
        while self.running:
            t0 = time.time()
            try:
                odo = session.get(self.odo_url, timeout=0.1).json()
                with self.lock:
                    self.x, self.y, self.phi = (float(odo[0]), float(odo[1]), float(odo[2]))
                    self.odo_fails = 0
            except Exception:
                with self.lock:
                    self.odo_fails += 1
            time.sleep(max(0.0, 0.1 - (time.time() - t0)))

    def poll_cam(self):
        session = requests.Session()
        while self.running:
            t0 = time.time()
            try:
                resp = session.get(self.cam_url, timeout=0.1)
                img  = cv2.imdecode(np.frombuffer(resp.content, np.uint8),cv2.IMREAD_COLOR)
                with self.lock:
                    self.img = img
                    self.camera_fails = 0
            except Exception:
                with self.lock:
                    self.camera_fails += 1
            time.sleep(max(0.0, 0.1 - (time.time() - t0)))

    def get(self):
        with self.lock:
            return self.x, self.y, self.phi, self.img

    def get_fails(self):
        with self.lock:
            return self.odo_fails, self.camera_fails

    def stop(self):
        self.running = False