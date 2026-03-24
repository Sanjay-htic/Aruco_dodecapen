"""
dodecahead_calibration.py

Performs Dodecahedron calibration using ArUco markers and
Bundle Adjustment (BA).

Pipeline:
1. Load camera intrinsics
2. Detect ArUco markers
3. Estimate marker poses (PnP)
4. Collect observations
5. Initialize object + marker poses
6. Run bundle adjustment (least squares)
7. Save optimized poses

Usage:
    python dodecahead_calibration.py \
        --image_dir datasets/raw/dodecapen \
        --calib_file configs/camera_calibration.npz \
        --output_dir outputs/dodeca_calibration
"""

import cv2
import numpy as np
import glob
import json
import os
import argparse
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares


# --------------------------------------------------
# CAMERA
# --------------------------------------------------
def load_camera_calibration(calib_path):
    with np.load(calib_path) as f:
        calib = dict(f)

    K = np.array(calib["camera_matrix"], dtype=np.float64)
    D = np.array(calib["dist_coeffs"], dtype=np.float64)

    print("[INFO] Loaded camera calibration")
    return K, D


# --------------------------------------------------
# ARUCO DETECTION
# --------------------------------------------------
def detect_aruco_markers(image, dict_type=cv2.aruco.DICT_6X6_250):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    corners, ids, _ = detector.detectMarkers(gray)
    return corners, ids


# --------------------------------------------------
# GEOMETRY
# --------------------------------------------------
def marker_3d_points(marker_size=20.0):
    s = marker_size / 2
    return np.array([
        [-s, -s, 0],
        [ s, -s, 0],
        [ s,  s, 0],
        [-s,  s, 0]
    ], dtype=np.float64)


def estimate_marker_pose(corners, K, D, marker_size=20.0):
    obj_pts = marker_3d_points(marker_size)
    img_pts = corners.reshape(-1, 2)

    success, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, K, D,
        flags=cv2.SOLVEPNP_SQPNP
    )

    if not success:
        return None

    R_cm, _ = cv2.Rodrigues(rvec)
    return R_cm, tvec.reshape(3)


# --------------------------------------------------
# OBSERVATIONS
# --------------------------------------------------
def collect_marker_observations(image_paths, K, D):
    observations = []

    for img_id, path in enumerate(image_paths):
        img = cv2.imread(path)
        corners, ids = detect_aruco_markers(img)

        if ids is None:
            continue

        for i, marker_id in enumerate(ids.flatten()):
            pose = estimate_marker_pose(corners[i], K, D)
            if pose is None:
                continue

            observations.append({
                "image_id": img_id,
                "marker_id": int(marker_id),
                "corners_2d": corners[i].reshape(-1, 2),
                "R_cam_marker": pose[0],
                "t_cam_marker": pose[1]
            })

    print(f"[INFO] Collected {len(observations)} observations")
    return observations


# --------------------------------------------------
# INITIALIZATION
# --------------------------------------------------
def initialize_marker_object_poses(marker_ids):
    return {mid: {"R": np.eye(3), "t": np.zeros(3)} for mid in marker_ids}


def initialize_object_camera_poses(observations, marker_obj_poses):
    obj_cam_poses = {}

    for obs in observations:
        img_id = obs["image_id"]
        mid = obs["marker_id"]

        if img_id in obj_cam_poses:
            continue

        R_cm = obs["R_cam_marker"]
        t_cm = obs["t_cam_marker"]

        R_om = marker_obj_poses[mid]["R"]
        t_om = marker_obj_poses[mid]["t"]

        R_co = R_cm @ R_om.T
        t_co = t_cm - R_co @ t_om

        obj_cam_poses[img_id] = {"R": R_co, "t": t_co}

    return obj_cam_poses


# --------------------------------------------------
# PARAM PACKING
# --------------------------------------------------
def pack_params(marker_obj_poses, obj_cam_poses, fixed_marker_id):
    params = []
    index = {}
    i = 0

    for mid, pose in marker_obj_poses.items():
        if mid == fixed_marker_id:
            continue
        rvec = R.from_matrix(pose["R"]).as_rotvec()
        params.extend(rvec)
        params.extend(pose["t"])
        index[f"marker_{mid}"] = i
        i += 6

    for img_id, pose in obj_cam_poses.items():
        rvec = R.from_matrix(pose["R"]).as_rotvec()
        params.extend(rvec)
        params.extend(pose["t"])
        index[f"image_{img_id}"] = i
        i += 6

    return np.array(params), index


# --------------------------------------------------
# OPTIMIZATION
# --------------------------------------------------
def reprojection_residuals(params, observations, K, D,
                          marker_obj_poses, obj_cam_poses,
                          index, fixed_marker_id):

    residuals = []
    obj_pts = marker_3d_points()

    for obs in observations:
        mid = obs["marker_id"]
        img_id = obs["image_id"]

        if mid == fixed_marker_id:
            R_om = marker_obj_poses[mid]["R"]
            t_om = marker_obj_poses[mid]["t"]
        else:
            idx = index[f"marker_{mid}"]
            rvec = params[idx:idx+3]
            tvec = params[idx+3:idx+6]
            R_om = R.from_rotvec(rvec).as_matrix()
            t_om = tvec

        idx = index[f"image_{img_id}"]
        rvec = params[idx:idx+3]
        tvec = params[idx+3:idx+6]
        R_co = R.from_rotvec(rvec).as_matrix()
        t_co = tvec

        pts_cam = (R_co @ (R_om @ obj_pts.T + t_om[:, None])).T + t_co
        proj, _ = cv2.projectPoints(pts_cam, np.zeros(3), np.zeros(3), K, D)

        residuals.extend((proj.reshape(-1, 2) - obs["corners_2d"]).ravel())

    return np.array(residuals)


# --------------------------------------------------
# SAVE
# --------------------------------------------------
def save_poses(data, path):
    out = {
        k: {"R": v["R"].tolist(), "t": v["t"].tolist()}
        for k, v in data.items()
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    K, D = load_camera_calibration(args.calib_file)

    image_paths = sorted(glob.glob(os.path.join(args.image_dir, "*.jpg")))
    print(f"[INFO] Found {len(image_paths)} images")

    observations = collect_marker_observations(image_paths, K, D)

    marker_ids = sorted(set(obs["marker_id"] for obs in observations))

    marker_obj_poses = initialize_marker_object_poses(marker_ids)
    obj_cam_poses = initialize_object_camera_poses(observations, marker_obj_poses)

    fixed_marker_id = marker_ids[0]

    x0, index = pack_params(marker_obj_poses, obj_cam_poses, fixed_marker_id)

    print("[INFO] Running bundle adjustment...")
    result = least_squares(
        reprojection_residuals,
        x0,
        method='lm',
        max_nfev=5000,
        args=(observations, K, D,
              marker_obj_poses, obj_cam_poses,
              index, fixed_marker_id)
    )

    print("[INFO] Final RMS:", np.sqrt(np.mean(result.fun**2)))

    # Unpack
    opt_marker_poses = {}
    opt_obj_cam_poses = {}

    for mid in marker_obj_poses:
        if mid == fixed_marker_id:
            opt_marker_poses[mid] = marker_obj_poses[mid]
        else:
            idx = index[f"marker_{mid}"]
            rvec = result.x[idx:idx+3]
            tvec = result.x[idx+3:idx+6]
            opt_marker_poses[mid] = {
                "R": R.from_rotvec(rvec).as_matrix(),
                "t": tvec
            }

    for img_id in obj_cam_poses:
        idx = index[f"image_{img_id}"]
        rvec = result.x[idx:idx+3]
        tvec = result.x[idx+3:idx+6]
        opt_obj_cam_poses[img_id] = {
            "R": R.from_rotvec(rvec).as_matrix(),
            "t": tvec
        }

    save_poses(opt_marker_poses,
               os.path.join(args.output_dir, "marker_object_poses.json"))

    save_poses(opt_obj_cam_poses,
               os.path.join(args.output_dir, "object_camera_poses.json"))

    print("[INFO] Calibration completed and saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dodecahead Calibration (BA)")

    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--calib_file", required=True)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    main(args)
    
# python src/calibration/dodecahead_calibration.py \
#     --image_dir datasets/raw/dodecapen \
#     --calib_file configs/camera_calibration.npz \
#     --output_dir outputs/dodeca_calibration