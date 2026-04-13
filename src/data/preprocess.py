import math
import random
import numpy as np
import torch
import open3d as o3d
import itertools
from typing import Literal

BBOX3D_CORNERS = np.array(
    [
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ],
    dtype=np.float32,
)


def create_o3d_pcd(pts: np.ndarray, cols: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
    return pcd


def statistical_outlier_rejection(
    pcd: o3d.geometry.PointCloud,
    nb_neighbors: int = 10,
    std_ratio: float = 2.0,
) -> o3d.geometry.PointCloud:
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    return pcd


def resample(pcd: o3d.geometry.PointCloud, N: int) -> o3d.geometry.PointCloud:
    n = len(pcd.points)
    if n == 0:
        return pcd
    if n <= N:
        idx = np.random.choice(n, N, replace=True)
        pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[idx])
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[idx])
    else:
        pcd = pcd.farthest_point_down_sample(N)
    return pcd


def get_obb_frame(points: torch.Tensor) -> dict:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.numpy())
    obb = pcd.get_minimal_oriented_bounding_box(robust=True)

    rotation = torch.tensor(
        np.asarray(obb.R), dtype=torch.float32
    )  # o3d rotation should be det=+1

    # but still make sure
    if torch.det(rotation) < 0:
        rotation[:, 2] *= -1
    
    # fix sign 
    for i in range(3):
        if rotation[i, i] < 0:
            rotation[:, i] *= -1

    translation = torch.tensor(np.asarray(obb.center), dtype=torch.float32)
    lwh = torch.tensor(obb.extent, dtype=torch.float32)
    canonical = (points - translation) @ rotation

    bbox = torch.from_numpy(BBOX3D_CORNERS) * lwh

    return dict(
        rotation=rotation,
        translation=translation,
        lwh=lwh,
        points=canonical,
        bbox=bbox,
    )


def get_pca_frame(points: torch.Tensor) -> dict:
    centroid = points.mean(dim=0)
    centered = points - centroid
    eig_values, eig_vectors = torch.linalg.eigh(centered.T @ centered)
    sorted_idx = torch.argsort(eig_values, descending=True)
    rotation = eig_vectors[:, sorted_idx]

    # fix sign (make axes point in +x, +y, +z)
    for i in range(3):
        if rotation[i, i] < 0:
            rotation[i] *= -1

    # check right-handedness
    if torch.det(rotation) < 0:
        rotation[:, -1] *= -1

    # rotate points to PCA frame and compute bbox center in that frame
    rotated_pc = centered @ rotation
    mins = rotated_pc.min(dim=0).values
    maxs = rotated_pc.max(dim=0).values
    bbox_center_rot_frame = (mins + maxs) / 2

    # translate points to bbox-centered PCA frame i.e the canonical frame
    canonical = rotated_pc - bbox_center_rot_frame

    # get translation from canonical frame to world frame
    translation = bbox_center_rot_frame @ rotation.T + centroid

    lwh = (
        maxs - mins
    )  # lwh corresponds to longest, medium, shortest axes due to sorting of eigenvalues

    bbox = torch.from_numpy(BBOX3D_CORNERS) * lwh

    return dict(
        rotation=rotation.float(),
        translation=translation.float(),
        points=canonical.float(),
        lwh=lwh.float(),
        bbox=bbox.float(),
    )


def extract_instance_points(
    pc_full: np.ndarray,  # (3, H, W)
    rgb_full: np.ndarray,  # (H, W, 3) uint8
    mask_i: np.ndarray,  # (H, W) bool
):
    pts = pc_full[:, mask_i].T.astype(np.float32)  # (K, 3)
    cols = rgb_full[mask_i].astype(np.float32) / 255.0  # (K, 3)
    return pts, cols


def transform_gt_to_canonical(
    bbox_world: torch.Tensor,  # (8, 3)
    rotation: torch.Tensor,  # (3, 3)
    translation: torch.Tensor,  # (3,)
) -> torch.Tensor:
    return (bbox_world - translation) @ rotation  # (8, 3)

def get_aligned_lwh_from_bbox(bbox_canonical: torch.Tensor):
    """
    bbox_canonical: (8, 3) corners of a perfect cuboid

    Returns:
        lwh: (3,) aligned with canonical axes (x, y, z)
        rotation: (3, 3) rotation from canonical frame → box frame
    """

    # Center the box
    translation = bbox_canonical.mean(dim=0)
    Pc = bbox_canonical - translation

    ref = Pc[0]
    vecs = Pc[1:] - ref  # (7, 3) vectors to all other corners

    # d1 - shortest vector from any corner is always a true edge.
    dists = torch.norm(vecs, dim=1)
    i1 = torch.argmin(dists)
    d1 = vecs[i1] / dists[i1]

    # d2 ensure orthogonality to d1 and pick the smallest to ensure edge
    perp = vecs - (vecs @ d1).unsqueeze(1) * d1  # (7, 3)
    perp_dists = torch.norm(perp, dim=1)
    perp_dists[i1] = float('inf')  # exclude e1 (its perp is zero)
    d2 = perp[torch.argmin(perp_dists)] # a perpendicular face diagonal can creep in but removed due to argmin
    d2 = d2 / torch.norm(d2)

    # d3 cross product to guaranteed orthogonal to both
    d3 = torch.linalg.cross(d1, d2)
    d3 = d3 / torch.norm(d3)

    axes = torch.stack([d1, d2, d3], dim=0)  # (3, 3)

    # ensure right-handed system
    if torch.det(axes) < 0:
        axes[2] *= -1

    # Compute dimensions via projection 
    Q = Pc @ axes.T
    lwh = Q.max(dim=0).values - Q.min(dim=0).values  # (3,)

    # Align axes to canonical axes 
    canonical = torch.eye(3, device=bbox_canonical.device)

    score = torch.abs(axes @ canonical.T)

    best_perm = None
    best_score = -1
    for perm in itertools.permutations(range(3)):
        s = (score[perm[0], 0] + score[perm[1], 1] + score[perm[2], 2]).item()
        if s > best_score:
            best_score = s
            best_perm = perm

    axes = axes[list(best_perm)]
    lwh = lwh[list(best_perm)]

    # Fix axis directions
    for i in range(3):
        if torch.dot(axes[i], canonical[i]) < 0:
            axes[i] *= -1

    rotation = axes.T 

    return lwh, translation, rotation

def normalize(
    input: torch.Tensor,
    scale: float,
):
    return input / scale


def augment(
    pts: torch.Tensor,  # (N, 3)
    cols: torch.Tensor,  # (N, 3)
    rotation: torch.Tensor,  # (3, 3)
    gt_rotation: torch.Tensor,
    gt_translation: torch.Tensor,  # (3,)
    augment_rotation:bool = False
):
    if augment_rotation:
        rotation = rotation.clone()
        gt_rotation = gt_rotation.clone()
        gt_translation = gt_translation.clone()

        # random X-flip
        if random.random() < 0.5:
            pts = pts.clone()
            pts[:, 0] *= -1
            rotation[:, 0] *= -1
            gt_rotation[0, :] *= -1
            gt_translation[0] *= -1

        # random Z-rotation ±15°
        angle = random.uniform(-15, 15) * math.pi / 180
        c, s = math.cos(angle), math.sin(angle)
        R_z = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float32)
        pts = pts @ R_z.T
        rotation = rotation @ R_z.T
        gt_rotation = R_z @ gt_rotation
        gt_translation = gt_translation @ R_z.T

    # point jitter
    N = pts.shape[0]
    ratio = 0.3

    mask = torch.rand(N) < ratio  
    noise = torch.randn_like(pts) * 0.002

    pts[mask] += noise[mask]

    # color jitter
    cols = cols + torch.randn_like(cols) * 0.02
    cols = cols.clamp(0.0, 1.0)

    return pts, cols, rotation, gt_rotation, gt_translation


def preprocess_pointcloud(
    pts: np.ndarray,
    cols: np.ndarray,
    N: int = 1024,
    min_pts: int = 100,
    max_dim: float = 0.5,
    scale: float = 1.0,
    canonical_frame: Literal["pca", "obb"] = "pca"
):
    pcd = create_o3d_pcd(pts, cols)
    pcd = statistical_outlier_rejection(pcd)
    pts = np.asarray(pcd.points, dtype=np.float32)
    cols = np.asarray(pcd.colors, dtype=np.float32)

    if len(pts) < min_pts:
        return None

    if canonical_frame == "pca":
        frame = get_pca_frame(torch.from_numpy(pts))
    elif canonical_frame == "obb":
        frame = get_obb_frame(torch.from_numpy(pts))
    else:
        raise KeyError('canonical_frame arg should be either "pca" or "obb"')
    
    if frame["lwh"].max().item() > max_dim:
        return None #length greater than set value means some problem in dataset

    pts_norm = normalize(frame["points"], 1.0)

    pcd = create_o3d_pcd(pts_norm.numpy(), cols)
    pcd = resample(pcd, N)

    return dict(
        points=torch.from_numpy(np.asarray(pcd.points, dtype=np.float32)),  # (N, 3)
        cols=torch.from_numpy(np.asarray(pcd.colors, dtype=np.float32)),    # (N, 3)
        rotation=frame["rotation"].float(),    # (3, 3)
        translation=frame["translation"].float(),  # (3,)
        scale=torch.tensor(scale),
        lwh=frame["lwh"].float(),              # (3,)
    )


def preprocess_bbox(
    bbox_world: torch.Tensor,  # (8, 3)
    rotation: torch.Tensor,    # (3, 3)
    translation: torch.Tensor, # (3,)
    scale: float = 1.0,
):
    bbox_canonical = transform_gt_to_canonical(bbox_world, rotation, translation)  # (8, 3)
    bbox_canonical = normalize(bbox_canonical, scale)
    lwh, translation, rotation = get_aligned_lwh_from_bbox(bbox_canonical)
    return dict(
        lwh=lwh.float(),
        translation=translation.float(),
        rotation=rotation.float(),
    )

