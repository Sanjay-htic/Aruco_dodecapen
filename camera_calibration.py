"""
camera_calibration.py

Performs camera calibration using checkerboard images.

Outputs:
    - Camera intrinsic matrix
    - Distortion coefficients
    - Reprojection error metrics
    - Saved calibration file (.npz)

Usage:
    python camera_calibration.py \
        --image_dir datasets/raw/checkerboard \
        --output_file configs/camera_calibration.npz
"""

import cv2
import numpy as np
import glob
import os
import argparse


def calibrate_camera(image_dir, chessboard_size=(9, 6), square_size=0.035):
    """
    Perform camera calibration.

    Args:
        image_dir (str): Path to checkerboard images
        chessboard_size (tuple): (cols, rows)
        square_size (float): Size of square in meters

    Returns:
        camera_matrix, dist_coeffs, rvecs, tvecs, reprojection_error
    """

    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints = []

    images = glob.glob(os.path.join(image_dir, "*.jpg"))

    if len(images) == 0:
        raise ValueError(f"No images found in {image_dir}")

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    print("\n[INFO] Camera Matrix:\n", camera_matrix)
    print("\n[INFO] Distortion Coefficients:\n", dist_coeffs)
    print(f"\n[INFO] RMS Reprojection Error: {ret:.4f}")

    # Per-image error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
        print(f"[INFO] Image {i}: reprojection error = {error:.4f} px")

    mean_error /= len(objpoints)
    print(f"\n[INFO] Mean Reprojection Error: {mean_error:.4f} px")

    return camera_matrix, dist_coeffs


def save_calibration(camera_matrix, dist_coeffs, output_file):
    """
    Save calibration parameters.

    Args:
        output_file (str): Path to save .npz file
    """

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    np.savez(
        output_file,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs
    )

    print(f"[INFO] Calibration saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera Calibration")

    parser.add_argument("--image_dir", type=str, required=True,
                        help="Path to checkerboard images")

    parser.add_argument("--output_file", type=str, required=True,
                        help="Output .npz file")

    args = parser.parse_args()

    cam_mtx, dist = calibrate_camera(args.image_dir)
    save_calibration(cam_mtx, dist, args.output_file)
    
    
"""# Collect dodecapen images
python datasets/scripts/collect_dodecapen_data.py \
    --output_dir datasets/raw/dodecapen

# Run calibration
python src/calibration/camera_calibration.py \
    --image_dir datasets/raw/checkerboard \
    --output_file configs/camera_calibration.npz
"""