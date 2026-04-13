import torch
import numpy as np

BBOX3D_CORNERS = torch.from_numpy(
    np.array(
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
)


def reconstruct_bbox(lwh: torch.Tensor, rot: torch.Tensor, tr: torch.Tensor):
    """
    Reconstruct box from lwh, rotation, and translation
    """
    bbox = (BBOX3D_CORNERS[None, ...].to(lwh.device) * lwh[:, None, :]) @ rot.transpose(-1, -2) + tr[:, None, :]
    return bbox  # (B,8,3)


def get_delta_lwh(prop_lwh, gt_lwh):
    return (gt_lwh - prop_lwh) / prop_lwh
