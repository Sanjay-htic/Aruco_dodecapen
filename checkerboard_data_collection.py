"""
collect_checkerboard_data.py

Captures checkerboard images using Intel RealSense camera
for camera calibration.

Controls:
    Press 's' → Save image
    Press 'q' → Quit

Usage:
    python collect_checkerboard_data.py --output_dir datasets/raw/checkerboard
"""

import os
import cv2
import numpy as np
import pyrealsense2 as rs
import argparse


def collect_images(output_dir, width=1280, height=720, fps=30):
    """
    Capture images from RealSense camera.

    Args:
        output_dir (str): Directory to save images
        width (int): Frame width
        height (int): Frame height
        fps (int): Frames per second
    """

    os.makedirs(output_dir, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    pipeline.start(config)

    image_count = 0

    print("[INFO] Press 's' to save image | 'q' to quit")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            cv2.imshow("Checkerboard Capture", color_image)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                filename = os.path.join(output_dir, f"calib_image_{image_count:03d}.jpg")
                cv2.imwrite(filename, color_image)
                print(f"[INFO] Saved: {filename}")
                image_count += 1

            elif key == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    print(f"[INFO] Total images captured: {image_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect checkerboard calibration images")

    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save captured images")

    args = parser.parse_args()

    collect_images(output_dir=args.output_dir)