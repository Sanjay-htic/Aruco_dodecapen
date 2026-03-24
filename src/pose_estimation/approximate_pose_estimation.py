"""
Real-time Approximate Pose Estimation (APE) using:
- ArUco markers
- PnP (with prediction)
- Kalman Filter smoothing
- ROI-based detection

Usage:
    python src/pose_estimation/approximate_pose_estimation.py \
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
# KALMAN FILTER INIT
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
# HELPERS
# ==============================
aruco_detector = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250),
    cv2.aruco.DetectorParameters()
)

def build_correspondences(detected_ids, detected_corners, marker_geometry, marker_size=20.0):
    obj_pts, img_pts = [], []
    s = marker_size / 2.0

    lc = np.array([
        [-s, -s, 0],
        [ s, -s, 0],
        [ s,  s, 0],
        [-s,  s, 0]
    ], dtype=np.float32)

    for m_id in detected_ids:
        m_str = str(m_id)
        if m_str not in marker_geometry:
            continue

        R_m = marker_geometry[m_str]["R"]
        t_m = marker_geometry[m_str]["t"]

        for i in range(4):
            obj_pts.append(R_m @ lc[i] + t_m)
            img_pts.append(detected_corners[m_id][i])

    return np.array(obj_pts, np.float32), np.array(img_pts, np.float32), len(obj_pts) >= 4


def get_bbox(rvec, tvec, marker_geometry, K, D):
    all_pts = []
    s = 10.0

    lc = np.array([
        [-s, -s, 0],
        [ s, -s, 0],
        [ s,  s, 0],
        [-s,  s, 0]
    ], dtype=np.float32)

    for k in marker_geometry:
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
    REPROJECTION_THRESHOLD = 5.0

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)

    initialized = False

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frame = np.asanyarray(frames.get_color_frame().get_data())
            h, w = frame.shape[:2]

            # --- PREDICT ---
            prediction = kf.predict()
            pred_rvec = prediction[:3]
            pred_tvec = prediction[3:6]

            # --- ROI ---
            if initialized:
                try:
                    xmin, ymin, xmax, ymax = get_bbox(pred_rvec, pred_tvec, marker_geometry, K, D)
                    bw, bh = xmax - xmin, ymax - ymin
                    xmin, xmax = max(0, xmin - bw), min(w, xmax + bw)
                    ymin, ymax = max(0, ymin - bh), min(h, ymax + bh)
                    roi = frame[ymin:ymax, xmin:xmax]
                except:
                    roi, xmin, ymin = frame, 0, 0
            else:
                roi, xmin, ymin = frame, 0, 0

            # --- DETECT ---
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = aruco_detector.detectMarkers(gray)

            if ids is not None:
                ids_list = ids.flatten().tolist()
                corners_dict = {
                    ids_list[i]: corners[i][0] + [xmin, ymin]
                    for i in range(len(ids_list))
                }

                obj_p, img_p, ok = build_correspondences(ids_list, corners_dict, marker_geometry)

                if ok:
                    succ, rvec, tvec = cv2.solvePnP(
                        obj_p, img_p, K, D,
                        rvec=pred_rvec,
                        tvec=pred_tvec,
                        useExtrinsicGuess=initialized,
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )

                    if succ:
                        proj_p, _ = cv2.projectPoints(obj_p, rvec, tvec, K, D)
                        mean_error = np.mean(np.linalg.norm(img_p - proj_p.reshape(-1, 2), axis=1))

                        if mean_error < REPROJECTION_THRESHOLD:
                            measurement = np.concatenate((rvec, tvec)).astype(np.float32)

                            if not initialized:
                                kf.statePost[:6] = measurement
                                initialized = True

                            estimated = kf.correct(measurement)
                            draw_r, draw_t = estimated[:3], estimated[3:6]

                            cv2.drawFrameAxes(frame, K, D, draw_r, draw_t, 40)
                            print(f"Pose accepted ✅ Error: {mean_error:.2f}")
                        else:
                            print(f"Pose rejected ❌ High Error: {mean_error:.2f}")
                    else:
                        initialized = False
            else:
                initialized = False

            cv2.imshow("Filtered APE Tracking", frame)

            if cv2.waitKey(1) == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Approximate Pose Estimation")

    parser.add_argument("--calib_file", required=True)
    parser.add_argument("--marker_file", required=True)

    args = parser.parse_args()

    main(args)