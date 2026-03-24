import numpy as np
import cv2

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=4,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

def run_ICT(prev_gray, curr_gray, prev_corners_dict):
    tracked_corners = {}
    if not prev_corners_dict:
        return tracked_corners

    marker_ids = list(prev_corners_dict.keys())

    prev_centers = np.array([
        np.mean(prev_corners_dict[mid], axis=0)
        for mid in marker_ids
    ], dtype=np.float32).reshape(-1,1,2)

    next_centers, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_centers, None, **LK_PARAMS
    )

    if next_centers is None or not any(status):
        return tracked_corners

    valid_idx = np.where(status.flatten() == 1)[0]
    if len(valid_idx) < 1:
        return tracked_corners

    vel_centers = (next_centers[valid_idx] - prev_centers[valid_idx]).reshape(-1, 2)
    vel_mags = np.linalg.norm(vel_centers, axis=1)

    mean_v, std_v = np.mean(vel_mags), np.std(vel_mags)

    for i in valid_idx:
        m_id = marker_ids[i]

        if abs(vel_mags[np.where(valid_idx==i)[0][0]] - mean_v) > 3 * std_v and len(valid_idx) > 2:
            continue

        p_corners = prev_corners_dict[m_id].reshape(-1, 1, 2).astype(np.float32)

        n_corners, c_status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, p_corners, None, **LK_PARAMS
        )

        if n_corners is not None and all(c_status):
            c_vel = (n_corners - p_corners).reshape(-1, 2)
            c_vel_mag = np.linalg.norm(c_vel, axis=1)

            mean_c, std_c = np.mean(c_vel_mag), np.std(c_vel_mag)

            if np.all(np.abs(c_vel_mag - mean_c) <= (3 * std_c + 0.1)):
                tracked_corners[m_id] = n_corners.reshape(4, 2)

    return tracked_corners