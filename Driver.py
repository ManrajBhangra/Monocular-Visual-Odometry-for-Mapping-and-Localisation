import threading
import queue
import requests

class Driver:
    timeout = 0.3
    stop_timeout = 1.0

    def __init__(self, url):
        self.url = url
        self.queue = queue.Queue(maxsize=1)
        self.running = True
        self.session = requests.Session()
        threading.Thread(target=self.send_move, daemon=True).start()

    def drive(self, x, y, angle):
        try:
            self.queue.put_nowait([float(x), float(y), float(angle)])
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait([float(x), float(y), float(angle)])
            except queue.Full:
                pass

    def full_stop(self):
        try:
            self.session.put(self.url, json=[0.0, 0.0, 0.0], timeout=self.stop_timeout)
        except Exception as e:
            print(f"\nEmergency Stop Error {e}")

    def shutdown(self):
        self.running = False
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass

    def send_move(self):
        while self.running:
            try:
                cmd = self.queue.get(timeout=0.5)
                if cmd is None:
                    break
                self.session.put(self.url, json=cmd, timeout=self.timeout)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"\nDriver error {e}")
                