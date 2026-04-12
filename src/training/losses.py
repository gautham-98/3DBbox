import torch
import torch.nn.functional as F
from .rot_utils import rot6d_to_rotmat

def loss_tr(pred_tr, gt_tr):
    return F.smooth_l1_loss(pred_tr, gt_tr)

def loss_lwh(pred_delta_lwh, gt_delta_lwh):
    return F.smooth_l1_loss(pred_delta_lwh, gt_delta_lwh)

def loss_rot(pred_6d, gt_R):
    """geodesic rotation loss"""
    pred_R = rot6d_to_rotmat(pred_6d)
    
    # get error
    err_R = pred_R.transpose(-2,-1) @ gt_R #(B, 3, 3)
    trace = torch.diagonal(err_R, dim1=-2, dim2=-1).sum(-1) #(B,)

    # find cos angle and clamp 
    cos_angle = ((trace - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)
    
    return torch.acos(cos_angle).mean() # TODO: check if maps to 0-3.14

def loss_corners(pred_corners, gt_corners):
    return (pred_corners - gt_corners).norm(dim=-1).sum(dim=-1).mean()