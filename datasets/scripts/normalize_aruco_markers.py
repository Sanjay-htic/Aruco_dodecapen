"""
normalize_aruco_markers.py

Converts ArUco marker images into normalized numpy arrays.

Output:
    Saves .npy files with values in [-1, 1]

Usage:
    python normalize_aruco_markers.py \
        --input_dir datasets/raw/aruco_markers \
        --output_dir datasets/processed/aruco_normalized
"""

import cv2
import numpy as np
import os
import glob
import argparse


def normalize_markers(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    paths = glob.glob(os.path.join(input_dir, "*.png"))

    for path in paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

        img_norm = (img / 127.5) - 1.0

        name = os.path.basename(path).replace(".png", ".npy")
        np.save(os.path.join(output_dir, name), img_norm)

    print(f"[INFO] Normalized {len(paths)} markers → {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize ArUco markers")

    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    normalize_markers(args.input_dir, args.output_dir)