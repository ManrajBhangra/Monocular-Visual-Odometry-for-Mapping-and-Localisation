import os
import pickle
import sys
import time
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

K = np.array([[617.38495106,   0.0,         313.36845276],
              [  0.0,         616.65160065, 249.14195857],
              [  0.0,           0.0,           1.0      ]], dtype=np.float32)

dist_coeffs = np.array([[ 1.40800417e-01, -4.46755128e-01,
                           2.66829535e-03,  2.96997280e-04,
                           7.94140375e-01]], dtype=np.float32)

features = "robotino_features.pkl"

class Localiser:

    bf_threshold = 80
    min_match = 10
    dbscan_eps = 0.5
    dbscan_min = 3
    pnp_min = 20
    max_features = 5000

    def __init__(self, features_file: str = features, cam_K: np.ndarray = K, cam_dist: np.ndarray = dist_coeffs):
        self.K = cam_K
        self.dist_coeffs = cam_dist
        self.db = []
        self.orb = cv2.ORB_create(nfeatures=1500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.ready = False

        if not os.path.exists(features_file):
            print(f"No features file found at '{features_file}'")
            return
        try:
            with open(features_file, "rb") as f:
                self.db = pickle.load(f)
            if len(self.db) > self.max_features:
                indices = np.random.choice(len(self.db), self.max_features, replace=False)
                self.db = [self.db[i] for i in indices]
            print(f"Loaded {len(self.db)} features from '{features_file}'")
            self.ready = len(self.db) >= self.min_match
        except Exception as e:
            print(f"Failed to load features: {e}")

    def localise_best(self, frame: np.ndarray, required: int = 10, min_inliers: int = 10):
        results = []
        deadline = time.time() + 10

        while len(results) < required:
            if time.time() > deadline:
                print(f"Timeout reached with only {len(results)}/{required} valid results.")
                break

            result = self.localise_with_inliers(frame)
            if result is not None:
                est_x, est_y, est_phi, inlier_count = result
                if inlier_count >= min_inliers:
                    results.append((est_x, est_y, est_phi, inlier_count))

        if not results:
            print("No valid localisation results obtained.")
            return None

        xs = [r[0] for r in results]
        ys = [r[1] for r in results]
        phis = [r[2] for r in results]
        est_x = float(np.median(xs))
        est_y = float(np.median(ys))
        est_phi = float(np.median(phis))
        avg_inliers = float(np.mean([r[3] for r in results]))
        print(f"best ({len(results)}/{required} samples): x={est_x:.1f} cm  y={est_y:.1f} cm  phi={(np.degrees(est_phi))%360:.1f}°")
        print(f"inliers={avg_inliers:.1f}")
        return (est_x, est_y, est_phi)

    def localise_with_inliers(self, frame: np.ndarray):
        if not self.ready or frame is None:
            return None

        frame_u = cv2.undistort(frame, self.K, self.dist_coeffs)
        gray    = cv2.cvtColor(frame_u, cv2.COLOR_BGR2GRAY)
        kps, des = self.orb.detectAndCompute(gray, None)

        if des is None or len(des) < self.pnp_min:
            return None

        db_3d = [lm for lm in self.db if "world_xyz" in lm and "des" in lm]
        if len(db_3d) < self.min_match:
            return None

        if len(db_3d) > self.max_features:
            indices = np.random.choice(len(db_3d), self.max_features, replace=False)
            db_3d = [db_3d[i] for i in indices]

        db_des = np.vstack([lm["des"] for lm in db_3d])
        db_xyz = np.array([lm["world_xyz"] for lm in db_3d], dtype=np.float32)
        matches = self.bf.match(des, db_des)
        matches = [m for m in matches if m.distance < self.bf_threshold]
        if len(matches) < self.min_match:
            return None

        matched_img_pts = np.array([kps[m.queryIdx].pt for m in matches], dtype=np.float32)
        matched_world_pts = db_xyz[[m.trainIdx for m in matches]]

        labels = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min).fit(matched_world_pts).labels_
        unique_labels = set(labels) - {-1}
        if not unique_labels:
            return None

        best_label = max(unique_labels, key=lambda l: (labels == l).sum())
        mask = labels == best_label

        if mask.sum() < self.pnp_min:
            return None

        obj_pts = matched_world_pts[mask]
        img_pts = matched_img_pts[mask]

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts, img_pts, self.K, self.dist_coeffs,
            iterationsCount=200,
            reprojectionError=6.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_EPNP
        )

        if not success or inliers is None:
            return None

        R, _ = cv2.Rodrigues(rvec)
        cam_world = (-R.T @ tvec).flatten()

        est_x = float(cam_world[0]) * 100
        est_y = float(cam_world[1]) * 100
        est_phi = float(np.arctan2(R[1, 0], R[0, 0]))

        return (est_x, est_y, est_phi, len(inliers))


if __name__ == "__main__":
    import requests

    ip_address = "192.168.0.9"
    snap_url   = f"http://{ip_address}/cam0"
    odo_url    = f"http://{ip_address}/data/odometry"

    loc = Localiser()

    if not loc.ready:
        print("Localiser not ready — check features file path.")
        sys.exit(1)

    print("Connecting to Robotino")
    deadline = time.time() + 5.0
    phi_now = None
    while time.time() < deadline:
        try:
            odo = requests.get(odo_url, timeout=0.5).json()
            phi_now = float(odo[2])
            break
        except Exception:
            time.sleep(0.1)

    if phi_now is None:
        print("ERROR: Could not reach odometry endpoint.")
        sys.exit(1)

    print("Localising")
    session = requests.Session()

    frame = None
    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            odo = session.get(odo_url, timeout=0.1).json()
            phi_now = float(odo[2])
        except Exception:
            pass
        try:
            resp  = session.get(snap_url, timeout=0.1)
            frame = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                break
        except Exception:
            pass
        time.sleep(0.1)

    if frame is None:
        print("Could not get camera frame.")
        sys.exit(1)

    result = loc.localise_best(frame)

    display = frame.copy()
    if result is not None:
        x, y, phi = result
        label = f"x={x:.1f}cm  y={y:.1f}cm  phi={(np.degrees(phi))%360:.1f}deg"
        color = (0, 220, 100)
        print(f"Final position: x={x:.1f}cm  y={y:.1f}cm  phi={(np.degrees(phi))%360:.1f}degrees")
    else:
        label = "Localisation failed"
        color = (0, 0, 220)
        print("Localisation failed.")

    cv2.putText(display, label, (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cv2.imshow("Localiser Result", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()