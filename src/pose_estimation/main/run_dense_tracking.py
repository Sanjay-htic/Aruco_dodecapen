import cv2
import numpy as np
import json
import glob
import os
import argparse
import pyrealsense2 as rs

from tracking.ape import build_correspondences
from tracking.ict import run_ICT
from tracking.dense_refinement import refine_pose_dense_gauss_newton
from utils.geometry import get_bbox
from utils.projection import compute_pen_tip_mm, reprojection_px_to_mm


# ================================
# CLI ARGUMENTS
# ================================
parser = argparse.ArgumentParser(description="Dense Pose Tracking with Pen Tip")

parser.add_argument("--calib", required=True, help="Path to camera calibration .npz")
parser.add_argument("--geometry", required=True, help="Path to marker geometry JSON")
parser.add_argument("--mipmaps", required=True, help="Folder containing masked mipmaps")
parser.add_argument("--pen_tip", nargs=3, type=float, help="Pen tip offset (x y z) in mm")

args = parser.parse_args()


# ================================
# LOAD CALIBRATION
# ================================
calib = np.load(args.calib)
K = calib["camera_matrix"]
D = calib["dist_coeffs"]

# ================================
# LOAD PEN TIP
# ================================
if args.pen_tip is not None:
    PEN_TIP_L = np.array(args.pen_tip, dtype=np.float32)
else:
    raise ValueError("❌ Please provide --pen_tip x y z")

print(f"[INFO] Pen tip offset: {PEN_TIP_L}")


# ================================
# LOAD MARKER GEOMETRY
# ================================
with open(args.geometry, "r") as f:
    marker_geometry = json.load(f)

for key in marker_geometry:
    marker_geometry[key]["R"] = np.array(marker_geometry[key]["R"], dtype=np.float32)
    marker_geometry[key]["t"] = np.array(marker_geometry[key]["t"], dtype=np.float32)

print(f"[INFO] Loaded marker geometry: {len(marker_geometry)} markers")


# ================================
# LOAD MIPMAPS
# ================================
marker_templates = {}

print("[INFO] Loading mipmaps...")

for npz_path in glob.glob(os.path.join(args.mipmaps, "*.npz")):
    try:
        base = os.path.basename(npz_path)
        marker_id = int(base.split('_')[1])

        data = np.load(npz_path)

        marker_templates[marker_id] = {
            'images': [data[f'img_level_{i}'] for i in range(6)],
            'masks':  [data[f'mask_level_{i}'] for i in range(6)]
        }

        print(f"  Loaded marker {marker_id}")

    except Exception as e:
        print(f"  Failed: {npz_path} → {e}")

print(f"[INFO] Total templates loaded: {len(marker_templates)}")


# ================================
# INIT KALMAN
# ================================
kf = cv2.KalmanFilter(12, 6)

kf.transitionMatrix = np.eye(12, dtype=np.float32)
for i in range(6):
    kf.transitionMatrix[i, i+6] = 1.0

kf.measurementMatrix = np.zeros((6, 12), np.float32)
for i in range(6):
    kf.measurementMatrix[i, i] = 1.0

kf.processNoiseCov = np.eye(12, dtype=np.float32) * 0.01
kf.measurementNoiseCov = np.eye(6, dtype=np.float32) * 0.01
kf.errorCovPost = np.eye(12, dtype=np.float32)


# ================================
# ARUCO DETECTOR
# ================================
aruco_detector = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250),
    cv2.aruco.DetectorParameters()
)


# ================================
# REALSENSE INIT
# ================================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)


initialized = False
prev_gray = None
prev_corners = {}
drawing_points = []


print("\n[INFO] Starting tracking...\n")


# ================================
# MAIN LOOP
# ================================
try:
    while True:
        frames = pipeline.wait_for_frames()
        frame = np.asanyarray(frames.get_color_frame().get_data())

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_norm = (gray.astype(np.float32) / 127.5) - 1.0

        prediction = kf.predict()
        pred_rvec, pred_tvec = prediction[:3], prediction[3:6]

        corners, ids, _ = aruco_detector.detectMarkers(gray)

        pose_found = False
        current_corners = {}

        # ================= APE =================
        if ids is not None:
            ids_list = ids.flatten().tolist()
            current_corners = {
                ids_list[i]: corners[i][0]
                for i in range(len(ids))
            }

            obj_p, img_p, ok = build_correspondences(current_corners, marker_geometry)

            if ok:
                success, rvec, tvec = cv2.solvePnP(
                    obj_p, img_p, K, D,
                    rvec=pred_rvec.copy(),
                    tvec=pred_tvec.copy(),
                    useExtrinsicGuess=initialized
                )

                if success:
                    pose_found = True
                    print(f"[APE] Markers: {len(ids)}")

        # ================= ICT =================
        if not pose_found and initialized and prev_gray is not None:
            tracked = run_ICT(prev_gray, gray, prev_corners)

            obj_p, img_p, ok = build_correspondences(tracked, marker_geometry)

            if ok:
                success, rvec, tvec = cv2.solvePnP(
                    obj_p, img_p, K, D,
                    rvec=pred_rvec.copy(),
                    tvec=pred_tvec.copy(),
                    useExtrinsicGuess=True
                )

                if success:
                    pose_found = True
                    current_corners = tracked
                    print(f"[ICT] Tracked markers: {len(tracked)}")

        # ================= DENSE =================
        if pose_found:
            used_ids = sorted(current_corners.keys())

            rvec, tvec = refine_pose_dense_gauss_newton(
                rvec, tvec, used_ids,
                marker_templates, marker_geometry,
                image_norm, K, D, gray.shape
            )

            measurement = np.concatenate((rvec, tvec)).astype(np.float32)

            if not initialized:
                kf.statePost[:6] = measurement
                initialized = True

            estimated = kf.correct(measurement)

            draw_r, draw_t = estimated[:3], estimated[3:6]

            # Pen tip projection
            tip_2d, _ = cv2.projectPoints(
                PEN_TIP_L.reshape(1,3), draw_r, draw_t, K, D
            )

            tip_px = tuple(tip_2d[0,0].astype(int))

            drawing_points.append(tip_px)

            # Draw trail
            for i in range(1, len(drawing_points)):
                cv2.line(frame, drawing_points[i-1], drawing_points[i], (255,0,255), 2)

            cv2.circle(frame, tip_px, 5, (0,0,255), -1)

            prev_corners = current_corners

        prev_gray = gray.copy()

        cv2.imshow("Dense Tracking", frame)

        if cv2.waitKey(1) == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    
# python run_dense_tracking.py \
#   --calib realsense_color_intrinsics_1280x720.npz \
#   --geometry optimized_marker_object_poses_ITERATIVE.json \
#   --mipmaps aruco_masked_mipmaps \
#   --pen_tip 0.30 -1.39 184.99