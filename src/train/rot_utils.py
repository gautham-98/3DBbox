import torch
import torch.nn.functional as F


def rot6d_to_rotmat(r6d: torch.Tensor) -> torch.Tensor:
    a1 = r6d[..., :3]
    a2 = r6d[..., 3:6]

    # gram-shmidt, predicted rotation vectors might not be exactly orthgonal
    b1 = F.normalize(a1, dim=-1)
    dot = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = F.normalize(a2 - dot * b1, dim=-1)

    # cross
    b3 = torch.cross(b1, b2, dim=-1)

    rotmat = torch.stack([b1, b2, b3], dim=-1)
    return rotmat


def rotmat_to_rot6d(rotmat: torch.Tensor) -> torch.Tensor:
    # First two columns
    col0 = rotmat[..., :, 0]  # (..., 3)
    col1 = rotmat[..., :, 1]  # (..., 3)
    r6d = torch.cat([col0, col1], dim=-1)  # (..., 6)
    return r6d
