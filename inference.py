import argparse
import numpy as np
import torch

from src.inference.pipeline import BoxPredictor
from src.models.boxestimator import BoxEstimationNet
from src.models.boxestimator_utonia import BoxEstimationNetUtonia


def parse_args():
    parser = argparse.ArgumentParser(description="3D Bounding Box Inference")

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth file")
    parser.add_argument("--pc",         type=str, required=True, help="Path to point cloud .npy file")
    parser.add_argument("--rgb",        type=str, required=True, help="Path to RGB image")
    parser.add_argument("--mask",       type=str, required=True, help="Path to instance mask .npy file")

    parser.add_argument("--model",       type=str, default="pointnet", choices=["pointnet", "utonia"])
    parser.add_argument("--frame",       type=str, default="obb",      choices=["obb", "pca"])
    parser.add_argument("--num_points",  type=int, default=1024)
    parser.add_argument("--batch_size",  type=int, default=8)
    parser.add_argument("--device",      type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--gt_boxes",   type=str, default=None, help="Optional path to GT bbox3d .npy for visualisation")
    parser.add_argument("--visualize",  action="store_true",    help="Visualise predictions with open3d")

    return parser.parse_args()


def main():
    args = parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    kmeans_centers = ckpt["kmeans_centers"]   # (K, 3) numpy
    K = kmeans_centers.shape[0]

    if args.model == "pointnet":
        model = BoxEstimationNet(in_channels=6, num_clusters=K)
    else:
        model = BoxEstimationNetUtonia(flash_attn=False, num_clusters=K)

    model.load_state_dict(ckpt["model_state_dict"])

    predictor = BoxPredictor(
        model=model,
        kmeans_centers=kmeans_centers,
        num_points=args.num_points,
        canonical_frame=args.frame,
        batch_sz=args.batch_size,
        device=args.device,
    )

    results = predictor(args.pc, args.rgb, args.mask)
    print(f"Predicted {len(results)} boxes")
    print(f"Output shape: {results.shape}")

    if args.visualize:
        gt_corners = np.load(args.gt_boxes) if args.gt_boxes else None
        predictor.visualise_result(gt_corners)


if __name__ == "__main__":
    main()
