import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.models.boxestimator import BoxEstimationNet
from src.utils.box_utils import reconstruct_bbox, BBOX3D_CORNERS
from src.utils.rot_utils import rot6d_to_rotmat
from src.eval.metrics import iou3d
from src.training.losses import loss_corners

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Evaluator:
    def __init__(self, model: BoxEstimationNet, ckpt_path: str, testloader: DataLoader):
        self.model: BoxEstimationNet = model.to(DEVICE)
        self.load_model(ckpt_path)
        self.testloader = testloader

    def load_model(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def evaluate(
        self,
    ):
        in_channels = self.model.in_channels

        all_mcd = []
        all_iou = []
        all_mcd_proposal = []
        all_iou_proposal = []

        with torch.no_grad():
            for batch in tqdm(self.testloader, desc="evaluating"):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(DEVICE)

                points = batch["points"]
                colors = batch["colors"]
                prop_lwh = batch["proposed_lwh"]
                gt_lwh = batch["gt_lwh"]
                gt_rot = batch["gt_rotation"]
                gt_tr = batch["gt_translation"]

                if in_channels == 3:
                    pc = points
                elif in_channels == 6:
                    pc = torch.cat([points, colors], dim=-1)
                else:
                    raise ValueError("in_channels must be 3 or 6")

                pred_delta_lwh, pred_6d, pred_tr = self.model(pc)
                pred_lwh = prop_lwh * (1 + pred_delta_lwh)
                pred_rot = rot6d_to_rotmat(pred_6d)

                proposed_corners = (
                    BBOX3D_CORNERS[None, ...].to(prop_lwh.device)
                    * prop_lwh[:, None, :]
                )

                pred_corners = reconstruct_bbox(pred_lwh, pred_rot, pred_tr)
                gt_corners = reconstruct_bbox(gt_lwh, gt_rot, gt_tr)

                all_mcd.append(loss_corners(pred_corners, gt_corners))
                all_iou.append(iou3d(pred_corners, gt_corners))

                all_mcd_proposal.append(loss_corners(proposed_corners, gt_corners))
                all_iou_proposal.append(iou3d(proposed_corners, gt_corners))

        mean_mcd = sum(all_mcd) / len(all_mcd)
        mean_iou = sum(all_iou) / len(all_iou)
        mean_iou_proposal = sum(all_iou_proposal) / len(all_iou_proposal)
        mean_mcd_proposal = sum(all_mcd_proposal) / len(all_mcd_proposal)

        print(f"Mean Corner Distance : {mean_mcd * 100:.3f} cm")
        print(f"Mean 3D IoU          : {mean_iou:.4f}")
        print(f"Mean Proposed Corner Distance : {mean_mcd_proposal * 100:.3f} cm")
        print(f"Mean Proposed 3D IoU          : {mean_iou_proposal:.4f}")

        return {
            "mean_corner_distance_m": mean_mcd,
            "mean_iou3d": mean_iou,
            "mean_corner_distance_proposal_m": mean_mcd_proposal,
            "mean_iou3d_proposal": mean_iou_proposal,
        }


if __name__ == "__main__":
    from src.data.dataset import get_dataloader
    from src.data.splits import get_splits

    FRAME = "pca"
    BATCH_SZ = 32
    NUM_WORKERS = 2
    N = 1024
    CKPT_PATH = "./checkpoints/checkpoint_epoch_395.pth"

    split_paths = get_splits(data_dir="dataset")
    testloader = get_dataloader(
        split_paths["test"],
        augment=False,
        shuffle=False,
        batch_size=BATCH_SZ,
        num_workers=NUM_WORKERS,
        canonical_frame=FRAME,
        N=N,
    )

    model = BoxEstimationNet(in_channels=6)

    evaluator = Evaluator(model, CKPT_PATH, testloader)

    evaluator.evaluate()
