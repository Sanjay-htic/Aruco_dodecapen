"""
Dodecahedron tracking with pen-tip projection and drawing.

Pipeline:
- APE (ArUco + RANSAC PnP)
- ICT fallback (Optical Flow)
- Kalman Filter smoothing
- Pen-tip projection & trajectory drawing

Controls:
    ESC → Exit
    C   → Clear drawing

Usage:
    python src/pose_estimation/pen_tip_tracking.py \
        --calib_file configs/camera_calibration.npz \
        --marker_file outputs/dodeca_calibration/marker_object_poses.json
"""

import cv2
import numpy as np
import json
import pyrealsense2 as rs
import argparse


# ================================
# LOAD INPUTS
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


# ================================
# KALMAN FILTER
# ================================
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


# ================================
# GLOBAL SETTINGS
# ================================
REPROJECTION_THRESHOLD = 4.0
MAX_DRAW_DISTANCE = 120

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=4,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

# 🔴 PEN TIP (from calibration)
PEN_TIP_L = np.array([0.30466388, -1.39366957, 184.99224307], dtype=np.float32)


# ================================
# ARUCO SETUP
# ================================
def create_aruco_detector():
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 7
    params.cornerRefinementMaxIterations = 30
    params.cornerRefinementMinAccuracy = 0.1

    return cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250),
        params
    )


# ================================
# CORE FUNCTIONS
# ================================
def build_correspondences(corners_dict, marker_geometry, marker_size=20.0):
    obj_pts, img_pts = [], []
    s = marker_size / 2.0

    lc = np.array([[-s,-s,0],[s,-s,0],[s,s,0],[-s,s,0]], dtype=np.float32)

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
    ).reshape(-1,1,2)

    next_centers, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_centers, None, **LK_PARAMS
    )

    if next_centers is None or not any(status):
        return tracked_corners

    valid_idx = np.where(status.flatten() == 1)[0]
    if len(valid_idx) < 1:
        return tracked_corners

    vel = (next_centers[valid_idx] - prev_centers[valid_idx]).reshape(-1, 2)
    vel_mag = np.linalg.norm(vel, axis=1)

    mean_v, std_v = np.mean(vel_mag), np.std(vel_mag)

    for i in valid_idx:
        m_id = marker_ids[i]

        if abs(vel_mag[np.where(valid_idx==i)[0][0]] - mean_v) > 3 * std_v and len(valid_idx) > 2:
            continue

        p_corners = prev_corners_dict[m_id].reshape(-1,1,2).astype(np.float32)

        n_corners, c_status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, p_corners, None, **LK_PARAMS
        )

        if n_corners is not None and all(c_status):
            tracked_corners[m_id] = n_corners.reshape(4,2)

    return tracked_corners


def get_bbox(rvec, tvec, marker_geometry, K, D):
    pts = []

    for k in marker_geometry:
        s = 10.0
        lc = np.array([[-s,-s,0],[s,-s,0],[s,s,0],[-s,s,0]], dtype=np.float32)
        for i in range(4):
            pts.append(marker_geometry[k]["R"] @ lc[i] + marker_geometry[k]["t"])

    proj, _ = cv2.projectPoints(np.array(pts), rvec, tvec, K, D)
    proj = proj.reshape(-1,2)

    return int(np.min(proj[:,0])), int(np.min(proj[:,1])), int(np.max(proj[:,0])), int(np.max(proj[:,1]))


# ================================
# MAIN LOOP
# ================================
def main(args):
    K, D, marker_geometry = load_inputs(args.calib_file, args.marker_file)
    kf = init_kalman()
    aruco_detector = create_aruco_detector()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)

    drawing_points = []
    initialized, prev_gray, prev_corners = False, None, {}

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frame = np.asanyarray(frames.get_color_frame().get_data())
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            prediction = kf.predict()
            pred_rvec, pred_tvec = prediction[:3], prediction[3:6]

            # --- ROI ---
            if initialized:
                try:
                    xmin, ymin, xmax, ymax = get_bbox(pred_rvec, pred_tvec, marker_geometry, K, D)
                    roi_gray = gray[ymin:ymax, xmin:xmax]
                except:
                    roi_gray, xmin, ymin = gray, 0, 0
            else:
                roi_gray, xmin, ymin = gray, 0, 0

            corners, ids, _ = aruco_detector.detectMarkers(roi_gray)

            current_corners, pose_found = {}, False

            # --- APE ---
            if ids is not None:
                ids_list = ids.flatten().tolist()
                current_corners = {ids_list[i]: corners[i][0] + [xmin, ymin] for i in range(len(ids_list))}

                obj_p, img_p, ok = build_correspondences(current_corners, marker_geometry)

                if ok:
                    succ, rvec, tvec, _ = cv2.solvePnPRansac(
                        obj_p, img_p, K, D,
                        rvec=pred_rvec.copy(),
                        tvec=pred_tvec.copy(),
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )

                    if succ:
                        pose_found = True

            # --- ICT ---
            if not pose_found and initialized and prev_gray is not None:
                tracked = run_ICT(prev_gray, gray, prev_corners)
                obj_p, img_p, ok = build_correspondences(tracked, marker_geometry)

                if ok:
                    succ, rvec, tvec, _ = cv2.solvePnPRansac(
                        obj_p, img_p, K, D,
                        rvec=pred_rvec.copy(),
                        tvec=pred_tvec.copy(),
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )

                    if succ:
                        current_corners = tracked
                        pose_found = True

            # --- UPDATE ---
            if pose_found:
                measurement = np.concatenate((rvec, tvec)).astype(np.float32)

                if not initialized:
                    kf.statePost[:6] = measurement
                    initialized = True

                estimated = kf.correct(measurement)
                draw_r, draw_t = estimated[:3], estimated[3:6]

                tip_2d, _ = cv2.projectPoints(PEN_TIP_L.reshape(1,3), draw_r, draw_t, K, D)
                tip_px = tuple(tip_2d[0].flatten().astype(int))

                if not drawing_points or np.linalg.norm(np.array(tip_px) - np.array(drawing_points[-1])) < MAX_DRAW_DISTANCE:
                    drawing_points.append(tip_px)

                prev_corners = current_corners

                cv2.circle(frame, tip_px, 5, (0,0,255), -1)

            else:
                if drawing_points:
                    drawing_points.append(None)

            # --- DRAW TRAIL ---
            for i in range(1, len(drawing_points)):
                if drawing_points[i-1] is None or drawing_points[i] is None:
                    continue
                cv2.line(frame, drawing_points[i-1], drawing_points[i], (255,0,255), 2)

            prev_gray = gray.copy()

            cv2.imshow("Pen Tip Drawing", frame)

            key = cv2.waitKey(1)
            if key == 27:
                break
            elif key == ord('c'):
                drawing_points = []

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--calib_file", required=True)
    parser.add_argument("--marker_file", required=True)

    args = parser.parse_args()

    main(args)