import torch
import numpy as np
from tqdm import tqdm

from PIL import Image
from src.data.preprocess import (
    preprocess_pointcloud,
    extract_instance_points,
    create_o3d_pcd,
)
from src.models.boxestimator import BoxEstimationNet
from src.utils.box_utils import reconstruct_bbox, BBOX3D_CORNERS
from src.utils.rot_utils import rot6d_to_rotmat


class BoxPredictor:
    def __init__(
        self,
        model: BoxEstimationNet,
        kmeans_centers: np.ndarray,
        num_points: int = 1024,
        canonical_frame="pca",
        batch_sz=4,
        device="cpu",
    ):
        self.device = torch.device(device)
        self.model = model.eval().to(self.device)
        self.kmeans_centers = torch.from_numpy(kmeans_centers).float().to(self.device)  # (K, 3)
        self.num_points = num_points
        self.frame = canonical_frame
        self.batch_sz = batch_sz

        self._pc_full: np.ndarray = None
        self._rgb_full: np.ndarray = None
        self._result: np.ndarray = None
        self._proposal: np.ndarray = None

    def __call__(self, pc_path: str, rgb_path: str, mask_path: str) -> np.ndarray:
        """returns an array of bbox corners in world frame"""
        self._pc_full, self._rgb_full, masks = self.load_data(
            pc_path, rgb_path, mask_path
        )
        self._result = []
        self._proposal = []
        dataloader = self.preprocess(self._pc_full, self._rgb_full, masks)

        for batch in tqdm(dataloader, desc="processing"):
            in_channels = self.model.in_channels
            if in_channels == 6:
                pc = torch.cat([batch["points"], batch["colors"]], dim=-1)
            elif in_channels == 3:
                pc = batch["points"]
            else:
                raise ValueError(
                    "in_channels for the model is expected to be either 3 or 6"
                )

            cluster_logits, pred_6d, pred_tr, pred_residual = self.model(pc)
            pred_rot = rot6d_to_rotmat(pred_6d)
            cluster_probs = cluster_logits.softmax(dim=1)                            # (B, K)
            pred_lwh = cluster_probs @ self.kmeans_centers + pred_residual           # (B, 3)

            bboxes_proposal: torch.Tensor = (
                BBOX3D_CORNERS[None, ...]
                * batch["prop_lwh"][:, None, :]
                @ batch["rotation"].transpose(-1, -2)
                + batch["translation"][:, None, :]
            )

            bboxes_canonical = reconstruct_bbox(pred_lwh, pred_rot, pred_tr)
            bboxes_world: torch.Tensor = (
                bboxes_canonical @ batch["rotation"].transpose(-1, -2)
                + batch["translation"][:, None, :]
            )

            self._result.extend([bbox.detach().cpu().numpy() for bbox in bboxes_world])
            self._proposal.extend(
                [bbox.detach().cpu().numpy() for bbox in bboxes_proposal]
            )

        self._result = np.array(self._result)
        return self._result

    def preprocess(self, pc_full, rgb_full, masks):
        """return batched instances"""

        points, colors, rotation, translation, lwh = [], [], [], [], []
        dataloader = []

        for idx in range(len(masks)):
            pts_i, cols_i = extract_instance_points(pc_full, rgb_full, masks[idx])
            processed_dict = preprocess_pointcloud(
                pts=pts_i, cols=cols_i, N=self.num_points, canonical_frame=self.frame
            )
            if processed_dict is not None:
                points.append(processed_dict["points"])
                colors.append(processed_dict["cols"])
                rotation.append(processed_dict["rotation"])
                translation.append(processed_dict["translation"])
                lwh.append(processed_dict["lwh"])

                if len(points) == self.batch_sz:
                    dataloader.append(
                        dict(
                            points=torch.stack(points, dim=0),
                            colors=torch.stack(colors, dim=0),
                            rotation=torch.stack(rotation, dim=0),
                            translation=torch.stack(translation, dim=0),
                            prop_lwh=torch.stack(lwh, dim=0),
                        )
                    )
                    points, colors, rotation, translation, lwh = [], [], [], [], []
        # add the rest as final data to dataloader
        if points:
            dataloader.append(
                dict(
                    points=torch.stack(points, dim=0),
                    colors=torch.stack(colors, dim=0),
                    rotation=torch.stack(rotation, dim=0),
                    translation=torch.stack(translation, dim=0),
                    prop_lwh=torch.stack(lwh, dim=0),
                )
            )
        return dataloader

    def load_data(self, pc_path, rgb_path, mask_path):
        pc_full = np.load(pc_path)
        rgb_full = np.array(Image.open(rgb_path))
        mask = np.load(mask_path)
        return pc_full, rgb_full, mask

    def visualise_result(self, gt_corners=None):
        import open3d as o3d

        valid = self._pc_full[2] > 0.01  # (H, W) bool

        points = self._pc_full[:, valid].T  # (N, 3)
        colors = self._rgb_full[valid] / 255.0  # (N, 3)

        pcd = create_o3d_pcd(pts=points, cols=colors)

        edges = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]

        pred_boxes = []
        for corners in self._result:
            box = o3d.geometry.LineSet()
            box.points = o3d.utility.Vector3dVector(corners)
            box.lines = o3d.utility.Vector2iVector(edges)
            box.paint_uniform_color([0.0, 1.0, 0.0])
            pred_boxes.append(box)

        prop_boxes = []
        for corners in self._proposal:
            box = o3d.geometry.LineSet()
            box.points = o3d.utility.Vector3dVector(corners)
            box.lines = o3d.utility.Vector2iVector(edges)
            box.paint_uniform_color([1.0, 0.0, 0.0])
            prop_boxes.append(box)

        if gt_corners is not None:
            gt_boxes = []
            for corners in gt_corners:
                box = o3d.geometry.LineSet()
                box.points = o3d.utility.Vector3dVector(corners)
                box.lines = o3d.utility.Vector2iVector(edges)
                box.paint_uniform_color([0.0, 0.0, 1.0])
                gt_boxes.append(box)

        geoms = [pcd] + pred_boxes + prop_boxes
        if gt_corners is not None:
            geoms += gt_boxes

        o3d.visualization.draw_geometries(
            geoms,
            window_name="BBox Prediction - Green, Prop - Red"
            + (", GT - Blue" if gt_corners is not None else ""),
        )


if __name__ == "__main__":
    import os
    from src.data.splits import get_splits

    os.environ["XDG_SESSION_TYPE"] = "x11"

    split_paths = get_splits("dataset")
    SAMPLE = split_paths["test"][4]

    PC_PATH = f"{SAMPLE}/pc.npy"
    RGB_PATH = f"{SAMPLE}/rgb.jpg"
    MASK_PATH = f"{SAMPLE}/mask.npy"
    BOX_PATH = f"{SAMPLE}/bbox3d.npy"
    CHECKPOINT = "checkpoints/checkpoint_epoch_best.pth"
    IN_CHANNELS = 6
    DEVICE = "cpu"

    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    kmeans_centers = ckpt["kmeans_centers"]  # (K, 3) numpy
    model = BoxEstimationNet(in_channels=IN_CHANNELS, num_clusters=kmeans_centers.shape[0])
    model.load_state_dict(ckpt["model_state_dict"])

    predictor = BoxPredictor(model=model, kmeans_centers=kmeans_centers, device=DEVICE)
    results = predictor(PC_PATH, RGB_PATH, MASK_PATH)
    print(f"{len(results)} boxes predicted")
    gt_corners = np.load(BOX_PATH)
    predictor.visualise_result(gt_corners)
