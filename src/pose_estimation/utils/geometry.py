import numpy as np
import cv2

def get_bbox(rvec, tvec, marker_geometry, K, D):
    all_pts = []

    for k in marker_geometry:
        s = 10.0
        lc = np.array([
            [-s,-s,0],[s,-s,0],
            [s,s,0],[-s,s,0]
        ], dtype=np.float32)

        for i in range(4):
            all_pts.append(marker_geometry[k]["R"] @ lc[i] + marker_geometry[k]["t"])

    proj, _ = cv2.projectPoints(np.array(all_pts), rvec, tvec, K, D)
    proj = proj.reshape(-1, 2)

    return (
        int(np.min(proj[:,0])),
        int(np.min(proj[:,1])),
        int(np.max(proj[:,0])),
        int(np.max(proj[:,1]))
    )