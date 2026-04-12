import torch
import torch.nn.functional as F
from typing import TypedDict

def loss_tr(pred_tr, gt_tr):
    return F.smooth_l1_loss(pred_tr, gt_tr)

def loss_lwh(pred_delta_lwh, gt_delta_lwh):
    return F.smooth_l1_loss(pred_delta_lwh, gt_delta_lwh)

def loss_rot(pred_rot, gt_rot):
    # get error
    err_R = pred_rot.transpose(-2,-1) @ gt_rot #(B, 3, 3)
    trace = torch.diagonal(err_R, dim1=-2, dim2=-1).sum(-1) #(B,)

    # find cos angle and clamp 
    cos_angle = ((trace - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)
    
    return torch.acos(cos_angle).mean() # TODO: check if maps to 0-3.14

def loss_corners(pred_corners, gt_corners):
    return (pred_corners - gt_corners).norm(dim=-1).sum(dim=-1).mean()

class LossLambda(TypedDict):
    lwh: float
    rot: float
    tr: float
    corner: float

