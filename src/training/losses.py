import torch
import torch.nn.functional as F
from typing import TypedDict
from dataclasses import dataclass, field
from typing import List


def loss_tr(pred_tr, gt_tr, beta=0.1):
    return F.smooth_l1_loss(pred_tr, gt_tr, beta=beta)


def loss_cluster(pred_logits, gt_cluster_id, gamma=2.0, alpha=0.25):
    """focal loss to handle hard cases"""
    ce = F.cross_entropy(pred_logits, gt_cluster_id, reduction="none")
    p = torch.exp(-ce)
    focal_loss = alpha * ((1 - p) ** gamma) * ce
    return focal_loss.mean()


def loss_residual(pred_residual, gt_residual, beta=0.1):
    return F.smooth_l1_loss(pred_residual, gt_residual, beta=beta)


def loss_rot(pred_rot, gt_rot):
    # get error
    err_R = pred_rot.transpose(-2, -1) @ gt_rot  # (B, 3, 3)
    trace = torch.diagonal(err_R, dim1=-2, dim2=-1).sum(-1)  # (B,)

    # find cos angle and clamp
    cos_angle = ((trace - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)
    angle = torch.acos(cos_angle)
    weight = (angle.detach()/angle.detach().mean())
    return (weight*angle).mean()


def loss_corners(pred_corners, gt_corners):
    return (pred_corners - gt_corners).norm(dim=-1).sum(dim=-1).mean()

@dataclass
class LossLambda:
    cluster: float = None
    residual: float = None
    rot: float = None
    tr: float = None
    corner: float = None

    schedule: List[float] = field(default_factory=lambda: [0.2, 0.3, 0.5])
    schedule_weight: List[float] = field(default_factory=lambda: [0.1, 0.5, 1.0])
    schedule_lambda: List[str] = field(default_factory=lambda: ["residual", "corner"])

    progress: float = 0.0

    def __post_init__(self):
        total = sum(self.schedule)
        self.schedule = [s / total for s in self.schedule]

        if len(self.schedule) != len(self.schedule_weight):
            raise ValueError(
                f"schedule and schedule_weight must have same length, "
                f"got {len(self.schedule)} and {len(self.schedule_weight)}"
            )

    def set_progress(self, progress: float):
        self.progress = progress

    def _get_schedule_weight(self):
        bound = 0.0
        for i, period in enumerate(self.schedule):
            bound += period
            if self.progress <= bound:
                return self.schedule_weight[i]
        return self.schedule_weight[-1]  

    def __getitem__(self, key):
        if not hasattr(self, key):
            raise KeyError(f"{key} not defined in LossLambda")

        base = getattr(self, key)

        if base is None:
            raise KeyError(f"{key} has not been assigned a value")

        if key in self.schedule_lambda:
            return base * self._get_schedule_weight()
        else:
            return base
