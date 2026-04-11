import math
import random
import numpy as np
import torch
import open3d as o3d


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
    
    rotation = torch.tensor(np.asarray(obb.R), dtype=torch.float32) #o3d rotation should be det=+1
    translation = torch.tensor(np.asarray(obb.center), dtype=torch.float32)
    lwh = torch.tensor(obb.extent, dtype=torch.float32)
    canonical = (points - translation) @ rotation

    lx, w, h = (lwh / 2).tolist()
    bbox = torch.tensor(
        [
            [-lx, -w, -h],
            [lx, -w, -h],
            [lx, w, -h],
            [-lx, w, -h],
            [-lx, -w, h],
            [lx, -w, h],
            [lx, w, h],
            [-lx, w, h],
        ],
        dtype=torch.float32,
    )

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

    lwh = maxs - mins
    l_half, w_half, h_half = lwh / 2
    bbox = torch.tensor(
        [
            [-l_half, -w_half, -h_half],
            [l_half, -w_half, -h_half],
            [l_half, w_half, -h_half],
            [-l_half, w_half, -h_half],
            [-l_half, -w_half, h_half],
            [l_half, -w_half, h_half],
            [l_half, w_half, h_half],
            [-l_half, w_half, h_half],
        ],
        dtype=torch.float32,
    )

    return dict(
        rotation=rotation.float(),
        translation=translation.float(),
        points=canonical.float(),
        lwh=lwh.float(),
        bbox=bbox,
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
    gt_world: torch.Tensor,  # (8, 3)
    rotation: torch.Tensor,  # (3, 3)
    translation: torch.Tensor,  # (3,)
) -> torch.Tensor:
    return (gt_world - translation) @ rotation  # (8, 3)


def normalize(
    canonical_pts: torch.Tensor,  # (K, 3)
    gt_canonical: torch.Tensor,  # (8, 3)
    scale: float,
):
   return canonical_pts / scale, gt_canonical / scale, scale


def augment(
    pts: torch.Tensor,  # (N, 3)
    gt: torch.Tensor,  # (8, 3)
    cols: torch.Tensor,  # (N, 3)
):
    # random X-flip
    if random.random() < 0.5:
        pts = pts.clone()
        gt = gt.clone()
        pts[:, 0] *= -1
        gt[:, 0] *= -1

    # random Z-rotation ±15°
    angle = random.uniform(-15, 15) * math.pi / 180
    c, s = math.cos(angle), math.sin(angle)
    R = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float32)
    pts = pts @ R.T
    gt = gt @ R.T

    # point jitter
    pts = pts + torch.randn_like(pts) * 0.005
    pts = pts.clamp(-1.0, 1.0)

    # color jitter
    cols = cols + torch.randn_like(cols) * 0.02
    cols = cols.clamp(0.0, 1.0)

    return pts, gt, cols


def preprocess_instance(
    pc_full: np.ndarray,
    rgb_full: np.ndarray,
    mask_i: np.ndarray,
    gt_world: np.ndarray,  # (8, 3)
    N: int = 1024,
    min_pts: int = 100,
    max_dim: float = 0.5,
):
    pts, cols = extract_instance_points(pc_full, rgb_full, mask_i)

    pcd = create_o3d_pcd(pts, cols)
    pcd = statistical_outlier_rejection(pcd)
    pts = np.asarray(pcd.points, dtype=np.float32)
    cols = np.asarray(pcd.colors, dtype=np.float32)

    if len(pts) < min_pts:
        return None

    frame = get_gravity_aligned_frame(torch.from_numpy(pts))

    if frame["lwh"].max().item() > max_dim:
        return None

    gt_canonical = transform_gt_to_canonical(
        torch.from_numpy(gt_world.astype(np.float32)),
        frame["rotation"],
        frame["translation"],
    )

    pts_norm, gt_norm, scale = frame["points"], gt_canonical, 1.0

    pcd = create_o3d_pcd(pts_norm.numpy(), cols)
    pcd = resample(pcd, N)

    return dict(
        points =  np.asarray(pcd.points, dtype=np.float32),  # (N, 3)
        cols = np.asarray(pcd.colors, dtype=np.float32),  # (N, 3)
        gt_box = gt_norm.numpy().astype(np.float32),  # (8, 3)
        rotation = frame["rotation"].numpy(),  # (3, 3)
        translation = frame["translation"].numpy(),  # (3,)
        scale = np.float32(scale),
        lwh = frame["lwh"].numpy(),  # (3,)
    )
