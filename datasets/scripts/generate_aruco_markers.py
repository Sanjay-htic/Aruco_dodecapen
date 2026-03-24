"""
generate_aruco_markers.py

Generates ArUco markers for dodecahedron faces.

Usage:
    python generate_aruco_markers.py --output_dir datasets/raw/aruco_markers --num_markers 12

Description:
- Uses OpenCV ArUco module
- Generates marker images and saves them as PNG
"""

import cv2
import os
import argparse


def generate_markers(output_dir, num_markers=12, marker_size=600, dictionary_type=cv2.aruco.DICT_6X6_250):
    """
    Generate ArUco markers.

    Args:
        output_dir (str): Directory to save markers
        num_markers (int): Number of markers to generate
        marker_size (int): Image size in pixels
        dictionary_type: OpenCV ArUco dictionary
    """

    os.makedirs(output_dir, exist_ok=True)

    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_type)

    for marker_id in range(num_markers):
        marker_img = cv2.aruco.generateImageMarker(
            aruco_dict,
            marker_id,
            marker_size
        )

        filename = os.path.join(output_dir, f"marker_{marker_id}.png")
        cv2.imwrite(filename, marker_img)

    print(f"[INFO] Generated {num_markers} markers in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ArUco markers")

    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for markers")
    parser.add_argument("--num_markers", type=int, default=12,
                        help="Number of markers")
    parser.add_argument("--marker_size", type=int, default=600,
                        help="Marker image size in pixels")

    args = parser.parse_args()

    generate_markers(
        output_dir=args.output_dir,
        num_markers=args.num_markers,
        marker_size=args.marker_size
    )