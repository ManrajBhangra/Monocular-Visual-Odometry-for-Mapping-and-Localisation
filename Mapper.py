import sys
import numpy as np
import cv2
import pickle
import os

class Mapper:

    def __init__(self, K, dist_coeffs, features_file, cam_cal_fin):
        self.orb = cv2.ORB_create(nfeatures=1500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.last_kp = self.last_des = self.last_gray = self.last_pose = None
        self.mode = "moving"
        self.features = []
        self.xyz = []
        self.grid = {}
        self.K = K
        self.dist_coeffs = dist_coeffs
        self.features_file = features_file
        self.cam_cal_fin = cam_cal_fin
        self.dedup_radius = 0.08

    def grid_key(self, lm_x, lm_y):
        cell = int(lm_x / self.dedup_radius), int(lm_y / self.dedup_radius)
        return cell

    def is_duplicate(self, lm_x, lm_y):
        cx, cy = self.grid_key(lm_x, lm_y)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if (cx + dx, cy + dy) in self.grid:
                    return True
        return False

    def register(self, lm_x, lm_y, lm_z, des_row):
        key = self.grid_key(lm_x, lm_y)
        self.grid[key] = True
        self.xyz.append((lm_x, lm_y, lm_z))
        self.features.append({
            "des": des_row.reshape(1, -1),
            "world_xyz": (lm_x, lm_y, lm_z)
        })

    def process_frame(self, frame, pose, mode):
        enc_x, enc_y, enc_phi = pose
        self.mode = mode

        frame_u = cv2.undistort(frame, self.K, self.dist_coeffs)
        gray = cv2.cvtColor(frame_u, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)

        if self.last_des is None or des is None or len(des) < 8 or self.last_pose is None:
            self.last_kp = kp; self.last_des = des
            self.last_gray = gray; self.last_pose = pose
            return

        matches = self.bf.match(self.last_des, des)
        matches = sorted(matches, key=lambda m: m.distance)
        matches = [m for m in matches if m.distance < 60]

        if len(matches) < 10:
            self.last_kp = kp; self.last_des = des
            self.last_gray = gray; self.last_pose = pose
            return

        pts_prev = np.float32([self.last_kp[m.queryIdx].pt for m in matches])
        pts_curr = np.float32([kp[m.trainIdx].pt for m in matches])

        try:
            F, mask_f = cv2.findFundamentalMat(pts_prev, pts_curr, cv2.FM_RANSAC, ransacReprojThreshold=1.0, confidence=0.995)
        except cv2.error:
            self.last_kp = kp; self.last_des = des
            self.last_gray = gray; self.last_pose = pose
            return

        if F is None or mask_f is None or F.shape != (3, 3):
            self.last_kp = kp; self.last_des = des
            self.last_gray = gray; self.last_pose = pose
            return

        mask_f = mask_f.ravel().astype(bool)
        if mask_f.shape[0] != pts_prev.shape[0]:
            self.last_kp = kp; self.last_des = des
            self.last_gray = gray; self.last_pose = pose
            return

        pts_prev_in = pts_prev[mask_f]
        pts_curr_in = pts_curr[mask_f]
        matches_in  = [m for m, ok in zip(matches, mask_f) if ok]

        if len(pts_prev_in) < 8 or len(pts_prev_in) / len(matches) < 0.25:
            self.last_kp = kp; self.last_des = des
            self.last_gray = gray; self.last_pose = pose
            return

        E = np.dot(self.K.T, np.dot(F, self.K))

        try:
            n_inliers, R, t_unit, _ = cv2.recoverPose(E, pts_prev_in, pts_curr_in, self.K)
        except cv2.error:
            self.last_kp = kp; self.last_des = des
            self.last_gray = gray; self.last_pose = pose
            return

        if n_inliers < 6:
            self.last_kp = kp; self.last_des = des
            self.last_gray = gray; self.last_pose = pose
            return

        lx, ly, _ = self.last_pose
        enc_baseline = np.hypot(enc_x - lx, enc_y - ly)

        if mode == "moving" and enc_baseline > 0.02:
            p1 = np.dot(self.K, np.hstack([np.eye(3), np.zeros((3, 1))]))
            p2 = np.dot(self.K, np.hstack([R, t_unit * enc_baseline]))
            pts4d = cv2.triangulatePoints(p1, p2, pts_prev_in.T, pts_curr_in.T)
            w = pts4d[3]
            valid = np.abs(w) > 1e-6
            pts3d = np.zeros((pts4d.shape[1], 3))
            pts3d[valid] = (pts4d[:3, valid] / w[valid]).T
            pts3d_h = np.hstack([pts3d, np.ones((pts3d.shape[0], 1))])
            proj0_h = np.dot(np.dot(self.K, np.hstack([np.eye(3), np.zeros((3, 1))])), pts3d_h.T).T
            proj0_z = proj0_h[:, 2:3]
            reproj_err = np.full(len(pts3d), np.inf)
            valid_proj = valid & (np.abs(proj0_z.ravel()) > 1e-6)
            proj0_2d = proj0_h[valid_proj, :2] / proj0_z[valid_proj]
            reproj_err[valid_proj] = np.linalg.norm(proj0_2d - pts_prev_in[valid_proj], axis=1)

            for i, (pt3, m) in enumerate(zip(pts3d, matches_in)):
                if not valid[i] or reproj_err[i] > 2.0:
                    continue
                pt_robot = self.cam_cal_fin @ pt3.reshape(3, 1)
                xr = float(pt_robot[0].item())
                yr = float(pt_robot[1].item())
                zr = float(pt_robot[2].item())
                if not (0.3 < xr < 6.0) or abs(zr) > 2.0:
                    continue
                lm_x = enc_x + (xr * np.cos(enc_phi) - yr * np.sin(enc_phi))
                lm_y = enc_y + (xr * np.sin(enc_phi) + yr * np.cos(enc_phi))
                lm_z = max(0.0, zr)
                if self.is_duplicate(lm_x, lm_y):
                    continue
                self.register(lm_x, lm_y, lm_z, des[m.trainIdx])

        elif mode == "rotating":
            for m in matches_in:
                lm_x, lm_y, lm_z = enc_x, enc_y, 0.0
                if self.is_duplicate(lm_x, lm_y):
                    continue
                self.register(lm_x, lm_y, lm_z, des[m.trainIdx])

        self.last_kp = kp; self.last_des = des
        self.last_gray = gray; self.last_pose = pose

    def build_from_log(self, log):
        print(f"\n{'='*65}")
        print("Mapper: building map from saved frames")
        print(f"{'='*65}")

        if not os.path.exists(log):
            print(f"ERROR: frame log '{log}' not found.")
            return False

        with open(log, "rb") as f:
            saved = pickle.load(f)

        frame_log = saved["frame_log"] if isinstance(saved, dict) else saved
        keyframes = saved["keyframes"] if isinstance(saved, dict) else []

        total = len(frame_log)
        print(f"Replaying {total} logged frames...\n")

        for i, entry in enumerate(frame_log):
            frame = cv2.imread(entry["path"])
            if frame is None:
                print(f"WARNING: could not read {entry['path']}")
                continue
            self.process_frame(frame, entry["pose"], entry["mode"])
            sys.stdout.write(f"\r  [{i+1}/{total}]  enc:({entry['pose'][0]:.2f},{entry['pose'][1]:.2f})m  mode:{entry['mode']}   ")
            sys.stdout.flush()

        if len(keyframes) >= 2:
            print(f"\nCross-station pass: {len(keyframes)} keyframes")
            self.cross_station_pass(keyframes)

        print(f"\nMapping complete {len(self.features)} features added.")
        self.save()
        return True

    def cross_station_pass(self, keyframes):
        phi_tol = np.radians(15)
        loaded = []
        for kf in keyframes:
            frame = cv2.imread(kf["path"])
            if frame is None:
                continue
            frame_u = cv2.undistort(frame, self.K, self.dist_coeffs)
            gray = cv2.cvtColor(frame_u, cv2.COLOR_BGR2GRAY)
            kp, des = self.orb.detectAndCompute(gray, None)
            if des is not None and len(des) >= 8:
                loaded.append({"pose": kf["pose"], "kp": kp, "des": des})

        pairs_tried = 0
        pairs_ok = 0
        for i in range(len(loaded)):
            for j in range(i + 1, len(loaded)):
                a, b = loaded[i], loaded[j]
                ax, ay, aphi = a["pose"]
                bx, by, bphi = b["pose"]
                enc_baseline = np.hypot(bx - ax, by - ay)
                if enc_baseline < 0.10:
                    continue
                dphi = abs((aphi - bphi + np.pi) % (2 * np.pi) - np.pi)
                if dphi > phi_tol:
                    continue
                pairs_tried += 1
                matches = self.bf.match(a["des"], b["des"])
                matches = [m for m in matches if m.distance < 60]
                if len(matches) < 10:
                    continue
                pts_a = np.float32([a["kp"][m.queryIdx].pt for m in matches])
                pts_b = np.float32([b["kp"][m.trainIdx].pt for m in matches])
                try:
                    F, mask_f = cv2.findFundamentalMat(pts_a, pts_b, cv2.FM_RANSAC,
                                                        ransacReprojThreshold=1.0, confidence=0.995)
                except cv2.error:
                    continue
                if F is None or mask_f is None or F.shape != (3, 3):
                    continue
                mask_f = mask_f.ravel().astype(bool)
                if mask_f.shape[0] != pts_a.shape[0]:
                    continue
                pts_a_in = pts_a[mask_f]
                pts_b_in = pts_b[mask_f]
                matches_in = [m for m, ok in zip(matches, mask_f) if ok]
                if len(pts_a_in) < 8:
                    continue
                E = np.dot(self.K.T, np.dot(F, self.K))
                try:
                    n_inliers, R, t_unit, _ = cv2.recoverPose(E, pts_a_in, pts_b_in, self.K)
                except cv2.error:
                    continue
                if n_inliers < 6:
                    continue
                p1 = np.dot(self.K, np.hstack([np.eye(3), np.zeros((3, 1))]))
                p2 = np.dot(self.K, np.hstack([R, t_unit * enc_baseline]))
                pts4d = cv2.triangulatePoints(p1, p2, pts_a_in.T, pts_b_in.T)
                w = pts4d[3]
                valid = np.abs(w) > 1e-6
                pts3d = np.zeros((pts4d.shape[1], 3))
                pts3d[valid] = (pts4d[:3, valid] / w[valid]).T
                pts3d_h = np.hstack([pts3d, np.ones((pts3d.shape[0], 1))])
                proj0_h = np.dot(np.dot(self.K, np.hstack([np.eye(3), np.zeros((3, 1))])), pts3d_h.T).T
                proj0_z = proj0_h[:, 2:3]
                reproj_err = np.full(len(pts3d), np.inf)
                valid_proj = valid & (np.abs(proj0_z.ravel()) > 1e-6)
                proj0_2d = proj0_h[valid_proj, :2] / proj0_z[valid_proj]
                reproj_err[valid_proj] = np.linalg.norm(proj0_2d - pts_a_in[valid_proj], axis=1)
                added = 0
                for i2, (pt3, m) in enumerate(zip(pts3d, matches_in)):
                    if not valid[i2] or reproj_err[i2] > 2.0:
                        continue
                    pt_robot = self.cam_cal_fin @ pt3.reshape(3, 1)
                    xr = float(pt_robot[0].item())
                    yr = float(pt_robot[1].item())
                    zr = float(pt_robot[2].item())
                    if not (0.3 < xr < 6.0) or abs(zr) > 2.0:
                        continue
                    lm_x = ax + (xr * np.cos(aphi) - yr * np.sin(aphi))
                    lm_y = ay + (xr * np.sin(aphi) + yr * np.cos(aphi))
                    lm_z = max(0.0, zr)
                    if self.is_duplicate(lm_x, lm_y):
                        continue
                    self.register(lm_x, lm_y, lm_z, b["des"][m.trainIdx])
                    added += 1
                if added > 0:
                    pairs_ok += 1
                    sys.stdout.write(f"\r  pair ({i},{j}) baseline={enc_baseline:.2f}m dphi={np.degrees(dphi):.1f}deg +{added} features")
                    sys.stdout.flush()
        print(f"Cross-station pass done: {pairs_ok}/{pairs_tried} pairs contributed features.")

    def save(self):
        with open(self.features_file, "wb") as f:
            pickle.dump(self.features, f)
        if self.xyz:
            pc_array = np.array(self.xyz, dtype=np.float32)
            np.save("robotino_xyz.npy", pc_array)
            print(f"Saved: robotino_xyz.npy  ({len(self.xyz)} points)")
        print(f"Saved: {self.features_file}  ({len(self.features)} features)")
