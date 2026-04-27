import os
import cv2
import pickle
import numpy as np
class FrameLogger:

    def __init__(self, frames_dir, frames_file):
        self.pose_x = self.pose_y = self.pose_phi = 0.0
        self.mode = "moving"
        self.frame_log = []
        self.keyframes = []
        self.frame_index = 0
        self.last_image = None
        self.log_x = 0.0
        self.log_y = 0.0
        self.frames_dir = frames_dir
        self.frames_file = frames_file
        os.makedirs(self.frames_dir, exist_ok=True)

    def update_pose(self, x, y, phi, mode=None, odo_x=None, odo_y=None):
        self.pose_x = x
        self.pose_y = y
        self.pose_phi = phi
        if mode is not None:
            self.mode = mode
        self.log_x = odo_x if odo_x is not None else x
        self.log_y = odo_y if odo_y is not None else y

    def update_map(self, frame):
        if frame is None:
            return
        image_hash = int(frame[::8, ::8].mean() * 1000)
        if image_hash == self.last_image:
            return
        self.last_image = image_hash
        fname = os.path.join(self.frames_dir, f"frame_{self.frame_index:05d}.jpg")
        cv2.imwrite(fname, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        self.frame_log.append({
            "path": fname,
            "pose": (self.log_x, self.log_y, self.pose_phi),
            "mode": self.mode,
        })
        self.frame_index += 1

    def log_keyframe(self, frame, odo_x, odo_y, phi):
        if frame is None:
            return
        fname = os.path.join(self.frames_dir, f"keyframe_{len(self.keyframes):03d}.jpg")
        cv2.imwrite(fname, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        self.keyframes.append({
            "path": fname,
            "pose": (odo_x, odo_y, phi),
        })
        print(f"Keyframe logged at enc=({odo_x:.3f},{odo_y:.3f}) phi={np.degrees(phi):.1f}deg")

    def save(self):
        with open(self.frames_file, "wb") as f:
            pickle.dump({"frame_log": self.frame_log, "keyframes": self.keyframes}, f)
        print(f"Saved {len(self.frame_log)} frames  {len(self.keyframes)} keyframes")
