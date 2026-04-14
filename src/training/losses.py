import torch
import torch.nn.functional as F
from typing import TypedDict

def loss_tr(pred_tr, gt_tr, beta=0.1):
    return F.smooth_l1_loss(pred_tr, gt_tr, beta=beta)

def loss_cluster(pred_logits, gt_cluster_id, alpha=0.25, gamma=2.0):
    """focal loss to handle hard cases"""
    ce = F.cross_entropy(pred_logits, gt_cluster_id, reduction='none')
    p = torch.exp(-ce)
    focal_loss = alpha * ((1 - p) ** gamma) * ce
    return focal_loss.mean()

def loss_residual(pred_residual, gt_residual, beta=0.1):
    return F.smooth_l1_loss(pred_residual, gt_residual, beta=beta)

def loss_rot(pred_rot, gt_rot, gamma=1.5):
    # get error
    err_R = pred_rot.transpose(-2,-1) @ gt_rot #(B, 3, 3)
    trace = torch.diagonal(err_R, dim1=-2, dim2=-1).sum(-1) #(B,)

    # find cos angle and clamp
    cos_angle = ((trace - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)
    angle = torch.acos(cos_angle)

    # find cosine score per element 0.5*(1-cos_angle) maps (0,180) -> (0,1)
    weight = (0.5*(1-cos_angle.detach())) ** gamma

    return (angle * weight).mean()

def loss_corners(pred_corners, gt_corners):
    return (pred_corners - gt_corners).norm(dim=-1).sum(dim=-1).mean()

class LossLambda(TypedDict):
    cluster: float
    residual: float
    rot: float
    tr: float
    corner: float

