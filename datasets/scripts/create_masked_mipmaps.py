"""
create_masked_mipmaps.py

Creates mipmaps along with gradient masks at each level.
Used for dense pose refinement.

Usage:
    python create_masked_mipmaps.py \
        --input_dir datasets/processed/aruco_normalized \
        --output_dir datasets/processed/aruco_masked_mipmaps
"""

import cv2
import numpy as np
import os
import glob
import argparse


def create_masked_mipmaps(input_dir, output_dir, threshold=0.05):
    os.makedirs(output_dir, exist_ok=True)

    paths = glob.glob(os.path.join(input_dir, "*.npy"))

    for path in paths:
        img = np.load(path)

        pyramid = []
        masks = []

        current = img

        while min(current.shape) > 16:
            gx = cv2.Sobel(current, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(current, cv2.CV_32F, 0, 1, ksize=3)

            grad = np.sqrt(gx**2 + gy**2)
            mask = grad > threshold

            pyramid.append(current)
            masks.append(mask)

            current = cv2.pyrDown(current)

        base_name = os.path.splitext(os.path.basename(path))[0]

        save_dict = {}

        for i, level in enumerate(pyramid):
            save_dict[f"img_level_{i}"] = level

        for i, mask in enumerate(masks):
            save_dict[f"mask_level_{i}"] = mask

        np.savez(
            os.path.join(output_dir, f"{base_name}_masked_mipmaps.npz"),
            **save_dict
        )

        print(f"[INFO] Saved {len(pyramid)} levels → {base_name}")

    print("[INFO] Masked mipmaps ready")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create masked mipmaps")

    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.05)

    args = parser.parse_args()

    create_masked_mipmaps(args.input_dir, args.output_dir, args.threshold)