"""
Approximate Pose Estimation (APE) with fallback to ICT (Iterative Corner Tracking).

Pipeline:
- APE (ArUco + PnP)
- ICT fallback (Optical Flow tracking)
- Kalman Filter smoothing
- ROI optimization

Usage:
    python src/pose_estimation/ape_with_ict.py \
        --calib_file configs/camera_calibration.npz \
        --marker_file outputs/dodeca_calibration/marker_object_poses.json
"""

import cv2
import numpy as np
import json
import pyrealsense2 as rs
import argparse


# ================================
# LOAD CAMERA & GEOMETRY
# ================================
def load_inputs(calib_file, marker_file):
    calib = np.load(calib_file)
    K = calib["camera_matrix"]
    D = calib["dist_coeffs"]

    with open(marker_file, "r") as f:
        marker_geometry = json.load(f)

    for key in marker_geometry:
        marker_geometry[key]["R"] = np.array(marker_geometry[key]["R"], dtype=np.float32)
        marker_geometry[key]["t"] = np.array(marker_geometry[key]["t"], dtype=np.float32)

    return K, D, marker_geometry


# ==============================
# SETTINGS & KALMAN INIT
# ==============================
def init_kalman():
    kf = cv2.KalmanFilter(12, 6)

    kf.transitionMatrix = np.eye(12, dtype=np.float32)
    for i in range(6):
        kf.transitionMatrix[i, i+6] = 1.0

    kf.measurementMatrix = np.zeros((6, 12), np.float32)
    for i in range(6):
        kf.measurementMatrix[i, i] = 1.0

    kf.processNoiseCov = np.eye(12, dtype=np.float32) * 0.01
    kf.measurementNoiseCov = np.eye(6, dtype=np.float32) * 0.1
    kf.errorCovPost = np.eye(12, dtype=np.float32)

    return kf


# ==============================
# GLOBAL SETTINGS
# ==============================
REPROJECTION_THRESHOLD = 8.0
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

aruco_detector = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250),
    cv2.aruco.DetectorParameters()
)


# ==============================
# CORE FUNCTIONS
# ==============================
def build_correspondences(corners_dict, marker_geometry, marker_size=20.0):
    obj_pts, img_pts = [], []
    s = marker_size / 2.0

    lc = np.array([
        [-s, -s, 0],
        [ s, -s, 0],
        [ s,  s, 0],
        [-s,  s, 0]
    ], dtype=np.float32)

    for m_id, img_corners in corners_dict.items():
        m_str = str(m_id)
        if m_str not in marker_geometry:
            continue

        R_m = marker_geometry[m_str]["R"]
        t_m = marker_geometry[m_str]["t"]

        for i in range(4):
            obj_pts.append(R_m @ lc[i] + t_m)
            img_pts.append(img_corners[i])

    return np.array(obj_pts, np.float32), np.array(img_pts, np.float32), len(obj_pts) >= 8


def run_ICT(prev_gray, curr_gray, prev_corners_dict):
    tracked_corners = {}
    if not prev_corners_dict:
        return tracked_corners

    marker_ids = list(prev_corners_dict.keys())
    prev_centers = np.array(
        [np.mean(prev_corners_dict[mid], axis=0) for mid in marker_ids],
        dtype=np.float32
    ).reshape(-1, 1, 2)

    next_centers, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_centers, None, **LK_PARAMS
    )

    if next_centers is None or not any(status):
        return tracked_corners

    valid_idx = np.where(status.flatten() == 1)[0]
    if len(valid_idx) < 1:
        return tracked_corners

    vel_centers = (next_centers[valid_idx] - prev_centers[valid_idx]).reshape(-1, 2)
    vel_mags = np.linalg.norm(vel_centers, axis=1)

    mean_v, std_v = np.mean(vel_mags), np.std(vel_mags)

    for i in valid_idx:
        m_id = marker_ids[i]

        if abs(vel_mags[np.where(valid_idx == i)[0][0]] - mean_v) > 3 * std_v and len(valid_idx) > 2:
            continue

        p_corners = prev_corners_dict[m_id].reshape(-1, 1, 2).astype(np.float32)

        n_corners, c_status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, p_corners, None, **LK_PARAMS
        )

        if n_corners is not None and all(c_status):
            c_vel = (n_corners - p_corners).reshape(-1, 2)
            c_vel_mag = np.linalg.norm(c_vel, axis=1)

            mean_c, std_c = np.mean(c_vel_mag), np.std(c_vel_mag)
            mask = np.abs(c_vel_mag - mean_c) <= (3 * std_c + 0.1)

            if np.all(mask):
                tracked_corners[m_id] = n_corners.reshape(4, 2)

    return tracked_corners


def get_bbox(rvec, tvec, marker_geometry, K, D):
    all_pts = []

    for k in marker_geometry:
        s = 10.0
        lc = np.array([[-s,-s,0],[s,-s,0],[s,s,0],[-s,s,0]], dtype=np.float32)
        for i in range(4):
            all_pts.append(marker_geometry[k]["R"] @ lc[i] + marker_geometry[k]["t"])

    proj, _ = cv2.projectPoints(np.array(all_pts), rvec, tvec, K, D)
    proj = proj.reshape(-1, 2)

    return int(np.min(proj[:,0])), int(np.min(proj[:,1])), int(np.max(proj[:,0])), int(np.max(proj[:,1]))


# ==============================
# MAIN
# ==============================
def main(args):
    K, D, marker_geometry = load_inputs(args.calib_file, args.marker_file)
    kf = init_kalman()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)

    initialized = False
    prev_gray = None
    prev_corners = {}

    print("--- Starting Dodecahedron Tracking ---")
    print(f"Reprojection Threshold: {REPROJECTION_THRESHOLD}px")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frame = np.asanyarray(frames.get_color_frame().get_data())
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            h, w = frame.shape[:2]

            prediction = kf.predict()
            pred_rvec, pred_tvec = prediction[:3], prediction[3:6]

            if initialized:
                try:
                    xmin, ymin, xmax, ymax = get_bbox(pred_rvec, pred_tvec, marker_geometry, K, D)
                    bw, bh = xmax - xmin, ymax - ymin
                    xmin, xmax = max(0, xmin - bw), min(w, xmax + bw)
                    ymin, ymax = max(0, ymin - bh), min(h, ymax + bh)
                    roi_gray = gray[ymin:ymax, xmin:xmax]
                except:
                    roi_gray, xmin, ymin = gray, 0, 0
            else:
                roi_gray, xmin, ymin = gray, 0, 0

            corners, ids, _ = aruco_detector.detectMarkers(roi_gray)

            current_corners_candidate = {}
            pose_found = False

            # --- APE ---
            if ids is not None:
                ids_list = ids.flatten().tolist()
                current_corners_candidate = {
                    ids_list[i]: corners[i][0] + [xmin, ymin]
                    for i in range(len(ids_list))
                }

                obj_p, img_p, ok = build_correspondences(current_corners_candidate, marker_geometry)

                if ok:
                    succ, rvec, tvec = cv2.solvePnP(
                        obj_p, img_p, K, D,
                        rvec=pred_rvec.copy(),
                        tvec=pred_tvec.copy(),
                        useExtrinsicGuess=initialized,
                        flags=cv2.SOLVEPNP_SQPNP
                    )

                    if succ:
                        proj_p, _ = cv2.projectPoints(obj_p, rvec, tvec, K, D)
                        err = np.mean(np.linalg.norm(img_p - proj_p.reshape(-1, 2), axis=1))

                        if err < REPROJECTION_THRESHOLD:
                            pose_found = True
                            mode_text = "APE (ArUco)"
                            print(f"[APE] SUCCESS: Error = {err:.3f} px | Markers: {len(ids)}")
                        else:
                            print(f"[APE] REJECTED: Error too high ({err:.3f} px)")

            # --- ICT ---
            if not pose_found and initialized and prev_gray is not None:
                tracked_dict = run_ICT(prev_gray, gray, prev_corners)

                obj_p, img_p, ok = build_correspondences(tracked_dict, marker_geometry)

                if ok:
                    succ, rvec, tvec = cv2.solvePnP(
                        obj_p, img_p, K, D,
                        rvec=pred_rvec.copy(),
                        tvec=pred_tvec.copy(),
                        useExtrinsicGuess=True,
                        flags=cv2.SOLVEPNP_SQPNP
                    )

                    if succ:
                        proj_p, _ = cv2.projectPoints(obj_p, rvec, tvec, K, D)
                        err = np.mean(np.linalg.norm(img_p - proj_p.reshape(-1, 2), axis=1))

                        if err < REPROJECTION_THRESHOLD:
                            current_corners_candidate = tracked_dict
                            pose_found = True
                            mode_text = "ICT (Optical Flow)"
                            print(f"[ICT] SUCCESS: Error = {err:.3f} px | Markers: {len(tracked_dict)}")
                        else:
                            print(f"[ICT] REJECTED: Error too high ({err:.3f} px)")

            # --- UPDATE ---
            if pose_found:
                measurement = np.concatenate((rvec, tvec)).astype(np.float32)

                if not initialized:
                    kf.statePost[:6] = measurement
                    initialized = True

                estimated = kf.correct(measurement)
                draw_r, draw_t = estimated[:3], estimated[3:6]

                cv2.drawFrameAxes(frame, K, D, draw_r, draw_t, 40)

                prev_corners = current_corners_candidate

                cv2.putText(frame, f"Mode: {mode_text}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Err: {err:.2f}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            else:
                if initialized:
                    print("[LOG] Tracking Lost - Predicting via Kalman Only")

            prev_gray = gray.copy()

            cv2.imshow("Dodecahedron Tracking (APE + ICT)", frame)

            if cv2.waitKey(1) == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APE + ICT Tracking")

    parser.add_argument("--calib_file", required=True)
    parser.add_argument("--marker_file", required=True)

    args = parser.parse_args()

    main(args)