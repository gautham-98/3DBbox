"""
Sanity check: does the model actually use the point cloud?

Runs inference on val set twice per batch:
  - normal: pc matched to its own GT
  - shuffled: pc permuted across batch (mismatched instances), same GT

If the model is using the point cloud, normal IoU >> shuffled IoU.
If they are similar, the model is ignoring the geometry and relying on the anchors alone.
"""

import argparse
import torch
from src.data.dataset import get_dataloader
from src.data.splits import get_splits
from src.models.boxestimator import BoxEstimationNet
from src.utils.box_utils import reconstruct_bbox
from src.utils.rot_utils import rot6d_to_rotmat
from src.eval.metrics import iou3d

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--frame", type=str, default="obb", choices=["pca", "obb"])
    parser.add_argument("--N", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--in_channels", type=int, default=6, choices=[3, 6])
    return parser.parse_args()


def main():
    args = parse_args()

    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    kmeans_centers = ckpt["kmeans_centers"]  # (K, 3) numpy
    kmeans_tensor  = torch.from_numpy(kmeans_centers).float().to(DEVICE)

    split_paths = get_splits(data_dir="dataset")
    valloader = get_dataloader(
        split_paths["val"],
        augment=False,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=0,
        canonical_frame=args.frame,
        N=args.N,
        kmeans_centers=kmeans_centers,
    )

    model = BoxEstimationNet(in_channels=args.in_channels, num_clusters=kmeans_centers.shape[0]).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ious_normal, ious_shuffled = [], []

    with torch.no_grad():
        for batch in valloader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(DEVICE)

            points        = batch["points"]
            colors        = batch["colors"]
            gt_cluster_id = batch["gt_cluster_id"]
            gt_residual   = batch["gt_residual"]
            gt_rot        = batch["gt_rotation"]
            gt_tr         = batch["gt_translation"]

            if args.in_channels == 6:
                pc = torch.cat([points, colors], dim=-1)
            else:
                pc = points

            gt_lwh     = kmeans_tensor[gt_cluster_id] * (1 + gt_residual)
            gt_corners = reconstruct_bbox(gt_lwh, gt_rot, gt_tr)

            # normal
            cluster_logits, pred_6d, pred_tr, pred_residual = model(pc)
            pred_rot     = rot6d_to_rotmat(pred_6d)
            pred_lwh     = cluster_logits.softmax(dim=1) @ kmeans_tensor * (1 + pred_residual)
            pred_corners = reconstruct_bbox(pred_lwh, pred_rot, pred_tr)
            ious_normal.append(iou3d(pred_corners, gt_corners))

            # shuffled — permute pc across batch, keep GT fixed
            idx = torch.randperm(pc.shape[0], device=DEVICE)
            cluster_logits_s, pred_6d_s, pred_tr_s, pred_residual_s = model(pc[idx])
            pred_rot_s     = rot6d_to_rotmat(pred_6d_s)
            pred_lwh_s     = cluster_logits_s.softmax(dim=1) @ kmeans_tensor * (1 + pred_residual_s)
            pred_corners_s = reconstruct_bbox(pred_lwh_s, pred_rot_s, pred_tr_s)
            ious_shuffled.append(iou3d(pred_corners_s, gt_corners))

    import numpy as np
    ious_normal   = np.array(ious_normal)
    ious_shuffled = np.array(ious_shuffled)

    print(f"normal IoU:   {ious_normal.mean():.4f}")
    print(f"shuffled IoU: {ious_shuffled.mean():.4f}")
    print(f"difference:   {(ious_normal - ious_shuffled).mean():.4f}")


if __name__ == "__main__":
    main()
