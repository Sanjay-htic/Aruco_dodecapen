"""
collect_dodecapen_data.py

Captures images of the dodecahedron tool using Intel RealSense.

Controls:
    SPACE → Save image
    ESC   → Quit

Usage:
    python collect_dodecapen_data.py --output_dir datasets/raw/dodecapen
"""

import os
import cv2
import numpy as np
import pyrealsense2 as rs
import argparse


def collect_dodecapen_data(output_dir, width=1280, height=720, fps=30):
    """
    Capture dodecapen dataset images.

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

    count = 0
    print("[INFO] Press SPACE to save image | ESC to quit")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            cv2.imshow("DodecaPen Capture", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break

            elif key == 32:  # SPACE
                filename = os.path.join(output_dir, f"dodeca_{count:04d}.jpg")
                cv2.imwrite(filename, frame)
                print(f"[INFO] Saved: {filename}")
                count += 1

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    print(f"[INFO] Total images captured: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect DodecaPen dataset images")

    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save captured images")

    args = parser.parse_args()

    collect_dodecapen_data(output_dir=args.output_dir)