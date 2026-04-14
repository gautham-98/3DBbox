import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.models.boxestimator import BoxEstimationNet
from src.utils.box_utils import reconstruct_bbox
from src.utils.rot_utils import rot6d_to_rotmat
from src.eval.metrics import iou3d
from src.training.losses import loss_corners

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Evaluator:
    def __init__(self, model: BoxEstimationNet, ckpt_path: str, testloader: DataLoader):
        self.model: BoxEstimationNet = model.to(DEVICE)
        self.testloader = testloader
        self.load_model(ckpt_path)

    def load_model(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.kmeans_centers = torch.from_numpy(ckpt["kmeans_centers"]).float().to(DEVICE)  # (K, 3)
        self.model.eval()

    def evaluate(self):
        in_channels = self.model.in_channels

        all_mcd = []
        all_iou = []

        with torch.no_grad():
            for batch in tqdm(self.testloader, desc="evaluating"):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(DEVICE)

                points        = batch["points"]           # (B, N, 3)
                colors        = batch["colors"]           # (B, N, 3)
                gt_cluster_id = batch["gt_cluster_id"]   # (B,)
                gt_residual   = batch["gt_residual"]      # (B, 3)
                gt_rot        = batch["gt_rotation"]      # (B, 3, 3)
                gt_tr         = batch["gt_translation"]   # (B, 3)

                if in_channels == 3:
                    pc = points
                elif in_channels == 6:
                    pc = torch.cat([points, colors], dim=-1)
                else:
                    raise ValueError("in_channels must be 3 or 6")

                cluster_logits, pred_6d, pred_tr, pred_residual = self.model(pc)
                pred_rot = rot6d_to_rotmat(pred_6d)

                cluster_probs = cluster_logits.softmax(dim=1)                        # (B, K)
                pred_lwh      = cluster_probs @ self.kmeans_centers + pred_residual  # (B, 3)
                gt_lwh        = self.kmeans_centers[gt_cluster_id] + gt_residual     # (B, 3)

                pred_corners = reconstruct_bbox(pred_lwh, pred_rot, pred_tr)
                gt_corners   = reconstruct_bbox(gt_lwh, gt_rot, gt_tr)

                all_mcd.append(loss_corners(pred_corners, gt_corners))
                all_iou.append(iou3d(pred_corners, gt_corners))

        mean_mcd = sum(all_mcd) / len(all_mcd)
        mean_iou = sum(all_iou) / len(all_iou)

        print(f"Mean Corner Distance : {mean_mcd * 100:.3f} cm")
        print(f"Mean 3D IoU          : {mean_iou:.4f}")

        return {
            "mean_corner_distance_m": mean_mcd,
            "mean_iou3d": mean_iou,
        }


if __name__ == "__main__":
    from src.data.dataset import get_dataloader
    from src.data.splits import get_splits

    FRAME = "obb"
    BATCH_SZ = 32
    NUM_WORKERS = 2
    N = 1024
    CKPT_PATH = "./ckpt_N1024_Fobb_EP400_B32_K8/checkpoint_epoch_best.pth"

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    kmeans_centers = ckpt["kmeans_centers"]  # (K, 3) numpy

    split_paths = get_splits(data_dir="dataset")
    testloader = get_dataloader(
        split_paths["test"],
        augment=False,
        shuffle=False,
        batch_size=BATCH_SZ,
        num_workers=NUM_WORKERS,
        canonical_frame=FRAME,
        N=N,
        kmeans_centers=kmeans_centers,
    )

    num_clusters = kmeans_centers.shape[0]
    model = BoxEstimationNet(in_channels=6, num_clusters=num_clusters)

    evaluator = Evaluator(model, CKPT_PATH, testloader)
    evaluator.evaluate()
