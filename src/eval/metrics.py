import numpy as np
import torch
from scipy.spatial import ConvexHull


def _point_in_obb(pts: np.ndarray, center: np.ndarray, axes: np.ndarray, half: np.ndarray) -> np.ndarray:
    diff = pts - center  # (N, 3)
    inside = np.ones(len(pts), dtype=bool)
    for i in range(3):
        proj = diff @ axes[:, i]
        inside &= np.abs(proj) <= half[i] + 1e-6
    return inside


def _edge_face_intersections(
    corners: np.ndarray,
    other_center: np.ndarray,
    other_axes: np.ndarray,
    other_half: np.ndarray,
) -> list:
    EDGES = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]
    pts = []
    for e in EDGES:
        p0, p1 = corners[e[0]], corners[e[1]]
        d = p1 - p0
        for i in range(3):
            for sign in [-1.0, 1.0]:
                n = other_axes[:, i]
                face_pt = other_center + sign * other_half[i] * n
                denom = np.dot(d, n)
                if abs(denom) < 1e-10:
                    continue
                t = (np.dot(face_pt, n) - np.dot(p0, n)) / denom
                if not (0.0 <= t <= 1.0):
                    continue
                pt = p0 + t * d
                diff = pt - other_center
                if all(abs(np.dot(diff, other_axes[:, j])) <= other_half[j] + 1e-6 for j in range(3)):
                    pts.append(pt)
    return pts


def _obb_intersection_volume(
    corners_a: np.ndarray, rot_a: np.ndarray, tr_a: np.ndarray, half_a: np.ndarray,
    corners_b: np.ndarray, rot_b: np.ndarray, tr_b: np.ndarray, half_b: np.ndarray,
) -> float:
    candidates = []

    mask = _point_in_obb(corners_a, tr_b, rot_b, half_b)
    candidates.extend(corners_a[mask])

    mask = _point_in_obb(corners_b, tr_a, rot_a, half_a)
    candidates.extend(corners_b[mask])

    candidates.extend(_edge_face_intersections(corners_a, tr_b, rot_b, half_b))
    candidates.extend(_edge_face_intersections(corners_b, tr_a, rot_a, half_a))

    if len(candidates) < 4:
        return 0.0

    try:
        return ConvexHull(np.array(candidates)).volume
    except Exception:
        return 0.0


def _obb_params_from_corners(corners: np.ndarray):
    """Extract center, axes (3,3), half-extents (3,) from 8 OBB corners."""
    center = corners.mean(axis=0)
    e0 = corners[1] - corners[0]
    e1 = corners[3] - corners[0]
    e2 = corners[4] - corners[0]
    half = np.array([np.linalg.norm(e0), np.linalg.norm(e1), np.linalg.norm(e2)]) / 2
    axes = np.stack([e0, e1, e2], axis=1) / (2 * half)  # (3,3) columns are unit axes
    vol = float(8 * np.prod(half))
    return center, axes, half, vol


def iou3d(pred_corners: torch.Tensor, gt_corners: torch.Tensor) -> float:
    """
    Mean 3D IoU over batch for oriented bounding boxes.
    pred_corners, gt_corners: (B, 8, 3)
    Returns scalar mean IoU in [0, 1].
    """
    pc = pred_corners.detach().cpu().numpy()
    gc = gt_corners.detach().cpu().numpy()

    ious = []
    for i in range(pc.shape[0]):
        tr_p, rot_p, half_p, vol_p = _obb_params_from_corners(pc[i])
        tr_g, rot_g, half_g, vol_g = _obb_params_from_corners(gc[i])
        inter = _obb_intersection_volume(
            pc[i], rot_p, tr_p, half_p,
            gc[i], rot_g, tr_g, half_g,
        )
        union = vol_p + vol_g - inter
        ious.append(inter / union if union > 1e-10 else 0.0)

    return float(np.mean(ious))
