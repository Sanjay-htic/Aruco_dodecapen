"""
create_aruco_mipmaps.py

Creates image pyramids (mipmaps) from normalized ArUco markers.

Usage:
    python create_aruco_mipmaps.py \
        --input_dir datasets/processed/aruco_normalized \
        --output_dir datasets/processed/aruco_mipmaps
"""

import cv2
import numpy as np
import os
import glob
import argparse


def create_mipmaps(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    paths = glob.glob(os.path.join(input_dir, "*.npy"))

    for path in paths:
        img = np.load(path)

        pyramid = []
        current = img

        while min(current.shape) > 16:
            pyramid.append(current)
            current = cv2.pyrDown(current)

        base_name = os.path.splitext(os.path.basename(path))[0]

        save_dict = {
            f"level{i}": level for i, level in enumerate(pyramid)
        }

        np.savez(
            os.path.join(output_dir, f"{base_name}_mipmaps.npz"),
            **save_dict
        )

    print(f"[INFO] Mipmaps created → {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create mipmaps")

    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    create_mipmaps(args.input_dir, args.output_dir)