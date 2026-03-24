"""
generate_aruco_masks.py

Creates gradient-based masks from normalized ArUco markers.

Usage:
    python generate_aruco_masks.py \
        --input_dir datasets/processed/aruco_normalized \
        --output_dir datasets/processed/aruco_masks
"""

import cv2
import numpy as np
import os
import glob
import argparse


def generate_masks(input_dir, output_dir, threshold=0.05):
    os.makedirs(output_dir, exist_ok=True)

    paths = glob.glob(os.path.join(input_dir, "*.npy"))

    for path in paths:
        img = np.load(path)

        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

        grad = np.sqrt(gx**2 + gy**2)
        mask = grad > threshold

        name = os.path.basename(path)
        np.save(os.path.join(output_dir, name), mask.astype(np.uint8))

    print(f"[INFO] Masks generated → {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ArUco masks")

    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.05)

    args = parser.parse_args()

    generate_masks(args.input_dir, args.output_dir, args.threshold)