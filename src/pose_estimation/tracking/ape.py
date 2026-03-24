import numpy as np

def build_correspondences(corners_dict, marker_geometry, marker_size=20.0):
    obj_pts, img_pts = [], []
    s = marker_size / 2.0

    lc = np.array([
        [-s,-s,0],[s,-s,0],
        [s,s,0],[-s,s,0]
    ], dtype=np.float32)

    for m_id, img_corners in corners_dict.items():
        m_str = str(m_id)
        if m_str not in marker_geometry:
            continue

        R_m = marker_geometry[m_str]["R"]
        t_m = marker_geometry[m_str]["t"]

        for i in range(4):
            obj_pts.append(R_m @ lc[i] + t_m)
            img_pts.append(img_corners[i])

    return np.array(obj_pts, np.float32), np.array(img_pts, np.float32), len(obj_pts) >= 8