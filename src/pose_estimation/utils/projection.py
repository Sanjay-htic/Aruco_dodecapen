import numpy as np
import cv2

def compute_pen_tip_mm(rvec, tvec, pen_tip_offset):
    R, _ = cv2.Rodrigues(rvec)
    return (R @ pen_tip_offset.reshape(3,1) + tvec.reshape(3,1)).flatten()

def reprojection_px_to_mm(px_err, tvec, K):
    depth = float(tvec[2,0])
    fx = float(K[0,0])
    return px_err * (depth / fx)