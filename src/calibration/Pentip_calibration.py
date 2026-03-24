"""
pentip_calibration.py

Estimates the pen-tip position relative to the dodecahedron frame
using multiple observed object poses.

Method:
- Uses pairwise linear constraints:
    (Ri - Rj) * c = tj - ti
- Solves using least squares

Outputs:
- Pen-tip position in dodecahedron frame
- Diagnostic metrics (rank, condition number, residuals)

Usage:
    python pentip_calibration.py \
        --input_file outputs/dodeca_calibration/object_camera_poses.json \
        --output_file outputs/pentip_calibration/pentip_pose.json
"""

import json
import numpy as np
import argparse
import os
import itertools


# --------------------------------------------------
# LOAD POSES
# --------------------------------------------------
def load_object_poses(input_file):
    """
    Load object-camera poses from JSON.

    Returns:
        R_list: (N, 3, 3)
        t_list: (N, 3, 1)
    """
    with open(input_file, "r") as f:
        poses_dict = json.load(f)

    R_list = []
    t_list = []

    for key in poses_dict:
        pose = poses_dict[key]
        R_list.append(np.array(pose["R"], dtype=np.float64))
        t_list.append(np.array(pose["t"], dtype=np.float64).reshape(3, 1))

    R_list = np.array(R_list)
    t_list = np.array(t_list)

    print(f"[INFO] Loaded {len(R_list)} poses")

    return R_list, t_list


# --------------------------------------------------
# SOLVE PEN-TIP
# --------------------------------------------------
def estimate_pentip_position(R_list, t_list):
    """
    Solve for pen-tip position using least squares.
    """

    m = len(R_list)
    A_rows = []
    b_rows = []

    for i, j in itertools.combinations(range(m), 2):
        Ri, Rj = R_list[i], R_list[j]
        ti, tj = t_list[i], t_list[j]

        A_rows.append(Ri - Rj)
        b_rows.append(tj - ti)

    A = np.vstack(A_rows)
    b = np.vstack(b_rows)

    c, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

    print("\n[INFO] Estimated pen-tip (dodeca frame):")
    print(c.flatten())

    return c, A, b


# --------------------------------------------------
# DIAGNOSTICS
# --------------------------------------------------
def compute_diagnostics(c, R_list, t_list, A, b):
    """
    Compute validation metrics.
    """

    z_values = [(Rk @ c + tk)[2] for Rk, tk in zip(R_list, t_list)]

    print("\n[INFO] Diagnostics:")
    print("Z-values:", z_values)
    print("Z std deviation:", np.std(z_values))
    print("Rank of A:", np.linalg.matrix_rank(A))
    print("Condition number:", np.linalg.cond(A))

    residual_norm = np.linalg.norm(A @ c - b)
    print("Residual norm:", residual_norm)

    pen_length = np.linalg.norm(c)
    print("Pen length (mm):", pen_length)

    return {
        "z_std": float(np.std(z_values)),
        "rank": int(np.linalg.matrix_rank(A)),
        "condition_number": float(np.linalg.cond(A)),
        "residual_norm": float(residual_norm),
        "pen_length": float(pen_length)
    }


# --------------------------------------------------
# SAVE
# --------------------------------------------------
def save_result(c, diagnostics, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    result = {
        "pentip_position": c.flatten().tolist(),
        "diagnostics": diagnostics
    }

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[INFO] Saved result → {output_file}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main(args):
    R_list, t_list = load_object_poses(args.input_file)

    c, A, b = estimate_pentip_position(R_list, t_list)

    diagnostics = compute_diagnostics(c, R_list, t_list, A, b)

    save_result(c, diagnostics, args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pen-tip Calibration")

    parser.add_argument("--input_file", required=True,
                        help="Object-camera poses JSON file")

    parser.add_argument("--output_file", required=True,
                        help="Output JSON file")

    args = parser.parse_args()

    main(args)
