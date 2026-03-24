import cv2
import numpy as np
import json
import glob
import os

def sample_active_marker_pixels(
    marker_id: int,
    level: int,
    marker_templates: dict,
    marker_geometry: dict,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    image_shape: tuple,
    physical_marker_size_mm: float = 20.0,
    target_points_per_marker: int = 800,     # desired order of magnitude after masking
    min_active_points: int = 80,
    max_active_points: int = 2000
):
    if marker_id not in marker_templates:
        return None, None, None

    data = marker_templates[marker_id]
    if level < 0 or level >= len(data['images']):
        return None, None, None

    template_mask = data['masks'][level]      # bool
    h, w = template_mask.shape

    if h < 16 or w < 16:
        return None, None, None

    half = physical_marker_size_mm / 2.0

    # Decide grid resolution so that we aim for ~target_points after masking
    # Assume ~30–60% of pixels are active (typical for edge-heavy ArUco)
    approx_active_ratio = 0.4
    target_grid_points = int(target_points_per_marker / approx_active_ratio)
    side = int(np.sqrt(target_grid_points)) + 1   # roughly square grid
    side = max(12, min(80, side))                 # clamp: no less than 12×12, no more than 80×80

    u, v = np.meshgrid(
        np.linspace(-half, half, side),
        np.linspace(-half, half, side),
        indexing='xy'
    )
    pts_local_3d = np.column_stack((u.ravel(), v.ravel(), np.zeros_like(u.ravel())))

    # Resize mask to match our chosen grid
    mask_resized = cv2.resize(
        template_mask.astype(np.uint8),
        (side, side),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    active = mask_resized.ravel()
    active_count_before_cull = np.sum(active)

    if active_count_before_cull < min_active_points:
        return None, None, None

    pts_local_3d = pts_local_3d[active]

    # Marker → dodecahedron
    geo = marker_geometry.get(str(marker_id))
    if geo is None:
        return None, None, None

    R_m = geo["R"]
    t_m = geo["t"]
    pts_dodec = (pts_local_3d @ R_m.T) + t_m

    # Project
    pts_image, _ = cv2.projectPoints(pts_dodec, rvec, tvec, K, D)
    pts_image = pts_image.reshape(-1, 2)

    # Visibility culling
    h_img, w_img = image_shape
    valid = (
        (pts_image[:,0] >= 0) & (pts_image[:,0] < w_img) &
        (pts_image[:,1] >= 0) & (pts_image[:,1] < h_img)
    )

    valid_count = np.sum(valid)
    if valid_count < min_active_points:
        return None, None, None

    # Optional: if way too many, thin out (but with side≤80 should be fine)
    if valid_count > max_active_points:
        keep_idx = np.random.choice(valid_count, max_active_points, replace=False)
        valid_idx = np.where(valid)[0][keep_idx]
        pts_image = pts_image[valid_idx]
        pts_local_3d = pts_local_3d[valid_idx]
        valid_count = max_active_points

    return pts_local_3d, pts_image, valid_count
       
# ── NEW: Sub-pixel intensity sampling from image ───────────────────────────────
def sample_image_intensity(
    image_norm: np.ndarray,  # normalized grayscale [-1,1], shape (H,W)
    uv_coords: np.ndarray,   # (N,2) sub-pixel coordinates (u,v) in [0,W-1], [0,H-1]
    interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """
    Sample intensities at sub-pixel positions using bilinear interpolation.
    Returns (N,) array of Ic(ui)
    """
    N = len(uv_coords)
    if N == 0:
        return np.array([])

    map_x = uv_coords[:, 0].astype(np.float32)  # u coords
    map_y = uv_coords[:, 1].astype(np.float32)  # v coords

    intensities = cv2.remap(
        image_norm,
        map_x,
        map_y,
        interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0  # neutral for [-1,1] range
    )

    return intensities

# ── NEW: Compute dense residuals for one marker ────────────────────────────────
def compute_dense_residuals(
    marker_id: int,
    level: int,
    pts_local_3d: np.ndarray,      # (N,3) local 3D points on marker plane
    pts_image: np.ndarray,         # (N,2) projected sub-pixel coords in image
    image_norm: np.ndarray,        # current frame normalized [-1,1]
    marker_templates: dict,
    physical_marker_size_mm: float = 20.0
) -> tuple[np.ndarray, float, int] | None:
    """
    Computes residuals r = Ic(ui) - Ot(xi) for one marker.
    Returns: (residuals vector (M,), mean |r|, num_valid) or None if failed
    """
    if marker_id not in marker_templates:
        return None, 0.0, 0

    data = marker_templates[marker_id]
    if level < 0 or level >= len(data['images']):
        return None, 0.0, 0

    tpl_img = data['images'][level]  # (h,w) float32 [-1,1]

    N = len(pts_image)
    if N == 0:
        return None, 0.0, 0

    # 1. Sample image intensities Ic(ui) with bilinear interp
    Ic_vals = sample_image_intensity(image_norm, pts_image)

    # 2. Sample template intensities Ot(xi)
    half = physical_marker_size_mm / 2.0
    uv_local = pts_local_3d[:, :2]  # (N,2) in [-half, half]
    uv_norm = (uv_local + half) / (2 * half)  # normalize to [0,1]

    # Scale to template pixel coords [0, w-1], [0, h-1]
    # Note: assuming y-down coord system (OpenCV style) — flip v if needed
    map_u_tpl = uv_norm[:, 0] * (tpl_img.shape[1] - 1)
    map_v_tpl = uv_norm[:, 1] * (tpl_img.shape[0] - 1)

    Ot_vals = cv2.remap(
        tpl_img,
        map_u_tpl.astype(np.float32),
        map_v_tpl.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0
    )

    # 3. Residuals r = Ic - Ot
    residuals = Ic_vals - Ot_vals

    # 4. Filter valid (finite, reasonable range)
    valid = np.isfinite(residuals) & (np.abs(residuals) < 2.0)  # loose bound for [-1,1]
    num_valid = np.sum(valid)

    if num_valid < 50:  # too few for meaningful error
        return None, 0.0, num_valid

    valid_residuals = residuals[valid]
    mean_abs_error = np.mean(np.abs(valid_residuals))

    return valid_residuals, mean_abs_error, num_valid

def refine_pose_dense_gauss_newton(
        rvec,
        tvec,
        marker_ids,
        marker_templates,
        marker_geometry,
        image_norm,
        K,
        D,
        image_shape,
        iterations=3):

    rvec = rvec.copy()          # ensure we don't modify input
    tvec = tvec.copy()
    
    # Save original for final comparison
    rvec_init = rvec.copy()
    tvec_init = tvec.copy()

    converged = False
    last_ea = None
    
    max_rot = 0.008        # radians  (~0.45 deg)
    max_trans = 0.003      # meters   (3 mm
    print("--------------------------------------------------")
    print("Dense Pose Refinement (Gauss-Newton)")
    print("--------------------------------------------------")


    for iter_idx in range(iterations):

        residuals_all = []
        J_all = []

        for mid in marker_ids:
            mid_str = str(mid)
            if mid_str not in marker_geometry:
                print(f"[GN iter {iter_idx}] Skipping unknown marker {mid} — not in calibrated geometry")
                continue
            geo = marker_geometry[mid_str]
            # ── mipmap level selection ───────────────────────────────────────
            s = 10.0
            lc = np.array([[-s,-s,0],[s,-s,0],[s,s,0],[-s,s,0]], np.float32)

            #geo = marker_geometry[str(mid)]
            pts = (lc @ geo["R"].T + geo["t"])

            proj, _ = cv2.projectPoints(pts, rvec, tvec, K, D)
            proj = proj.reshape(-1, 2)

            diag = np.linalg.norm(proj[0] - proj[2])

            if diag < 1:
                continue

            level = min(5, max(0, int(np.log2(600 / diag))))

            pts_3d, pts_2d, count = sample_active_marker_pixels(
                mid, level, marker_templates, marker_geometry,
                rvec, tvec, K, D, image_shape
            )

            if pts_3d is None:
                continue

            res_vec, _, _ = compute_dense_residuals(
                mid, level, pts_3d, pts_2d, image_norm, marker_templates
            )

            if res_vec is None:
                continue

            residuals_all.append(res_vec)

            # ── numerical Jacobian ────────────────────────────────────────────
            eps = 1e-4
            Ic1 = sample_image_intensity(image_norm, pts_2d)
            J_marker = []

            for k in range(6):
                dr = np.zeros(3, dtype=np.float32)
                dt = np.zeros(3, dtype=np.float32)

                if k < 3:
                    dr[k] = eps
                else:
                    dt[k-3] = eps

                r2 = rvec + dr.reshape(3,1)
                t2 = tvec + dt.reshape(3,1)

                proj2, _ = cv2.projectPoints(
                    (pts_3d @ geo["R"].T + geo["t"]),
                    r2, t2, K, D
                )
                proj2 = proj2.reshape(-1, 2)

                Ic2 = sample_image_intensity(image_norm, proj2)

                deriv = (Ic2 - Ic1) / eps
                J_marker.append(deriv.ravel())   # ensure 1D

            J_marker = np.stack(J_marker, axis=1)  # (N, 6)
            J_all.append(J_marker)

        if len(residuals_all) == 0:
            print(f"[GN iter {iter_idx}] No valid residuals → aborting refinement")
            break

        r = np.concatenate(residuals_all)
        J = np.vstack(J_all)

        current_ea = np.mean(np.square(r))
        print(f"[GN ITER {iter_idx+1}]")
        print(f"  Residual count : {len(r)}")
        print(f"  Ea(p)          : {current_ea:.6f}")
        #print(f"[GN iter {iter_idx}] {len(r)} residuals | Ea(p) = {current_ea:.4f}")

        if last_ea is not None and current_ea >= last_ea:
            print(f"[GN iter {iter_idx}] Ea(p) did not decrease → stopping early")
            break

        last_ea = current_ea

        H = J.T @ J
        g = J.T @ r

        delta = -np.linalg.solve(H + 1e-6 * np.eye(6), g)

        delta_rot = delta[:3].ravel()
        delta_trans = delta[3:].ravel()
        
        ################################
                # ---- clamp rotation ----
        rot_norm = np.linalg.norm(delta_rot)

        if rot_norm > max_rot:
            delta_rot *= max_rot/rot_norm

        # ---- clamp translation ----
        trans_norm = np.linalg.norm(delta_trans)

        if trans_norm > max_trans:
            delta_trans *= max_trans/trans_norm

        # recombine
        delta[:3] = delta_rot
        delta[3:] = delta_trans
        ###################################
        delta_norm = np.linalg.norm(delta)
        
        rot_deg = np.degrees(delta_rot)
        trans_mm = delta_trans * 1000

        print("  Update step")
        print(f"    Δrot (deg)  : [{rot_deg[0]:+.4f}, {rot_deg[1]:+.4f}, {rot_deg[2]:+.4f}]")
        print(f"    Δtrans (mm) : [{trans_mm[0]:+.3f}, {trans_mm[1]:+.3f}, {trans_mm[2]:+.3f}]")
        print(f"    Step norm   : {delta_norm:.6f}")

        #print(f"[GN iter {iter_idx}] Δrot = [{delta_rot[0]:+.6f}, {delta_rot[1]:+.6f}, {delta_rot[2]:+.6f}] rad")
        #print(f"Δtrans= [{delta_trans[0]*1000:+.3f}, {delta_trans[1]*1000:+.3f}, {delta_trans[2]*1000:+.3f}] mm  | norm = {delta_norm:.6f}")

        # Apply update (no need to reshape again if already column vectors)
        rvec += delta[:3].reshape(3, 1)
        tvec += delta[3:].reshape(3, 1)

        if delta_norm < 1e-5:
            print(f"[GN iter {iter_idx}] Converged ")
            converged = True
            break

    # ── Final summary (only once, after loop) ────────────────────────────────
    #total_rot_change = np.linalg.norm(rvec - rvec_init)
    #total_trans_change = np.linalg.norm(tvec - tvec_init) *1000
    total_rot_change_deg = np.degrees(np.linalg.norm(rvec - rvec_init))
    total_trans_change_mm = np.linalg.norm(tvec - tvec_init) * 1000

    iters_used = iter_idx + 1 if 'iter_idx' in locals() else 0

    #print(f"[GN SUMMARY] Final Ea(p) ≈ {current_ea:.4f}  |  "
    #      f"Total Δrot = {total_rot_change:.6f} rad  |  "
    #      f"Total Δtrans = {total_trans_change*1000:.3f} mm  |  "
    #      f"Iterations: {iters_used}{' (converged)' if converged else ''}")
    
    
    print("--------------------------------------------------")
    print("Gauss-Newton Summary")
    print("--------------------------------------------------")
    print(f"Iterations used  : {iters_used}")
    print(f"Final Ea(p)      : {current_ea:.6f}")
    print(f"Total Δrot       : {total_rot_change_deg:.4f} deg")
    print(f"Total Δtrans     : {total_trans_change_mm:.3f} mm")
    print(f"Converged        : {converged}")
    print("--------------------------------------------------")

    return rvec, tvec