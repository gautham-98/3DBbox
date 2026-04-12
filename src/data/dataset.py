import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .preprocess import (
    extract_instance_points,
    preprocess_pointcloud,
    preprocess_bbox,
    augment as augment_fn,
    BBOX3D_CORNERS,
)
from .splits import get_splits


class BBox3DDataset(Dataset):
    def __init__(
        self,
        sample_paths: list,
        augment: bool = False,
        N: int = 1024,
        use_rgb: bool = False,
        cache_dir: str = "data_cache",
        canonical_frame: str = "pca",
    ):
        self.augment = augment
        self.N = N
        self.use_rgb = use_rgb
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.canonical_frame = canonical_frame

        self.items: list[tuple[Path, int, str]] = []  # scene, obj_id, frame
        self._build_index(sample_paths)

    def _inst_cache(self, sp: Path, i: int, frame: str) -> Path:
        return (
            self.cache_dir / f"{sp.name}_{i}_N{self.N}_{self.canonical_frame}Frame.npz"
        )

    def _build_index(self, sample_paths: list):
        for sp in tqdm(sample_paths, desc="Building cache"):
            pc_full = np.load(sp / "pc.npy")
            rgb_full = np.array(Image.open(sp / "rgb.jpg"))
            mask = np.load(sp / "mask.npy")
            bboxes = np.load(sp / "bbox3d.npy")  # (M, 8, 3)

            for i in range(len(mask)):
                cpath = self._inst_cache(sp, i, self.canonical_frame)
                if not cpath.exists():
                    pts, cols = extract_instance_points(pc_full, rgb_full, mask[i])
                    result = preprocess_pointcloud(
                        pts, cols, N=self.N, canonical_frame=self.canonical_frame
                    )
                    if result is None:
                        np.savez(cpath, valid=np.bool_(False))
                    else:
                        bbox = preprocess_bbox(
                            torch.from_numpy(bboxes[i].astype(np.float32)),
                            result["rotation"],
                            result["translation"],
                            scale=result["scale"],
                        )
                        np.savez(
                            cpath,
                            valid=np.bool_(True),
                            points=result["points"].numpy(),
                            cols=result["cols"].numpy(),
                            frame_rotation=result[
                                "rotation"
                            ].numpy(),  # (3,3) world to canonical
                            frame_translation=result["translation"].numpy(),  # (3,)
                            proposed_lwh=result["lwh"].numpy(),  # (3,) proposed
                            gt_lwh=bbox["lwh"].numpy(),  # (3,)
                            gt_rotation=bbox[
                                "rotation"
                            ].numpy(),  # (3,3) rotation of GT box in canonical frame
                            gt_translation=bbox[
                                "translation"
                            ].numpy(),  # (3,) translation of GT box in canonical frame
                        )

                if np.load(cpath, allow_pickle=False)["valid"]:
                    self.items.append((sp, i, self.canonical_frame))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        sp, obj_idx, frame = self.items[idx]
        c = np.load(self._inst_cache(sp, obj_idx, frame), allow_pickle=False)

        pts = torch.from_numpy(c["points"])  # (N, 3)
        cols = torch.from_numpy(c["cols"])  # (N, 3)
        frame_rotation = torch.from_numpy(c["frame_rotation"])  # (3, 3) world→canonical
        frame_translation = torch.from_numpy(c["frame_translation"])  # (3,)
        proposed_lwh = torch.from_numpy(c["proposed_lwh"])  # (3,)
        gt_lwh = torch.from_numpy(c["gt_lwh"])  # (3,)
        gt_rotation = torch.from_numpy(
            c["gt_rotation"]
        )  # (3, 3) GT box axes in canonical frame
        gt_translation = torch.from_numpy(
            c["gt_translation"]
        )  # (3,) GT box translation in canonical frame

        if self.augment:
            pts, cols, frame_rotation, gt_rotation, gt_translation = augment_fn(
                pts, cols, frame_rotation, gt_rotation, gt_translation
            )

        points = torch.cat([pts, cols], dim=1) if self.use_rgb else pts  # (N, 3 or 6)

        return dict(
            points=points,
            colors=cols,
            gt_lwh=gt_lwh,  # (3,)  GT box dimensions
            gt_rotation=gt_rotation,  # (3,3) GT box axes in canonical frame
            gt_translation=gt_translation,  # (3,) GT box translation in canonical frame
            frame_rotation=frame_rotation,  # (3,3) world to canonical  (canonical to world: rot.T)
            frame_translation=frame_translation,  # (3,)  canonical frame origin in world
            proposed_lwh=proposed_lwh,  # (3,)  proposed PCA box dimensions
            sample_id=sp.name,
            obj_idx=obj_idx,
        )


def get_dataloader(
    sample_paths: list,
    augment: bool = False,
    shuffle: bool = False,
    batch_size: int = 32,
    num_workers: int = 4,
    canonical_frame: str = "pca",
    **kwargs,
) -> DataLoader:
    ds = BBox3DDataset(
        sample_paths, augment=augment, canonical_frame=canonical_frame, **kwargs
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


if __name__ == "__main__":
    splits = get_splits("dataset")
    loader = get_dataloader(
        splits["train"][:10],
        augment=True,
        shuffle=True,
        batch_size=4,
        num_workers=0,
        canonical_frame="obb",
    )

    batch = next(iter(loader))
    print("points:            ", batch["points"].shape)
    print("colors:            ", batch["colors"].shape)
    print("gt_lwh:            ", batch["gt_lwh"].shape)
    print("gt_rotation:       ", batch["gt_rotation"].shape)
    print("frame_rotation:    ", batch["frame_rotation"].shape)
    print("frame_translation: ", batch["frame_translation"].shape)
    print("gt_translation:    ", batch["gt_translation"].shape)
    print("proposed_lwh:      ", batch["proposed_lwh"].shape)
    print(f"Total train items: {len(loader.dataset)}")

    # visualise
    os.environ["XDG_SESSION_TYPE"] = "x11"
    import open3d as o3d

    idx = 2
    pts_np = batch["points"][idx].numpy()
    col_np = batch["colors"][idx].numpy()
    rot = batch["frame_rotation"][idx].numpy()
    tr = batch["frame_translation"][idx].numpy()
    gt_lwh = batch["gt_lwh"][idx].numpy()
    gt_rot = batch["gt_rotation"][idx].numpy()
    p_lwh = batch["proposed_lwh"][idx].numpy()
    gt_tr = batch["gt_translation"][idx].numpy()

    # load raw GT bbox from disk and transform to canonical frame
    sp_str = batch["sample_id"][idx]
    obj_idx = batch["obj_idx"][idx]
    bbox_world = np.load(f"dataset/{sp_str}/bbox3d.npy")[obj_idx].astype(
        np.float32
    )  # (8,3)
    # pc_full = np.load(f"dataset/{sp_str}/pc.npy")  # (3, H, W)
    # rgb_full = np.array(Image.open(f"dataset/{sp_str}/rgb.jpg"))
    # mask = np.load(f"dataset/{sp_str}/mask.npy")[obj_idx].astype(bool)  # (H, W)
    # pts, cols = extract_instance_points(pc_full, rgb_full, mask)  # to ensure instance points are correct
    # pcd = o3d.geometry.PointCloud()
    # valid = pc_full[2] > 0.01  # filter zero-depth
    # pcd.points = o3d.utility.Vector3dVector(pts)
    # pcd.colors = o3d.utility.Vector3dVector(cols)
    # lines_world = o3d.geometry.LineSet()
    # lines_world.points = o3d.utility.Vector3dVector(bbox_world)
    # lines_world.lines = o3d.utility.Vector2iVector([
    #     [0,1],[1,2],[2,3],[3,0],
    #     [4,5],[5,6],[6,7],[7,4],
    #     [0,4],[1,5],[2,6],[3,7],
    # ])
    # lines_world.paint_uniform_color([0.0, 0.0, 1.0])  # blue = raw GT in world frame
    # o3d.visualization.draw_geometries([pcd, lines_world], window_name="GT box in world frame (blue)")

    # canonical frame visualisation
    bbox_canonical = (bbox_world - tr) @ rot  # world → canonical

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        pts_np @ rot.T + tr # maps back to canonical frame
    )  # transform points to canonical frame for visualisation
    pcd.colors = o3d.utility.Vector3dVector(col_np)

    corners_gt = (
        (BBOX3D_CORNERS * gt_lwh) @ gt_rot.T + gt_tr # express the corners in canonical frame
    )  @ rot.T + tr # map back to world
    corners_frame = (
        BBOX3D_CORNERS * p_lwh # proposed box corners in canonical frame
    )  @ rot.T + tr # map back to world
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
    lines1 = o3d.geometry.LineSet()
    lines1.points = o3d.utility.Vector3dVector(corners_gt)
    lines1.lines = o3d.utility.Vector2iVector(edges)
    lines1.paint_uniform_color(
        [0.0, 1.0, 0.0]
    )  # green = GT (reconstructed from lwh+rot)

    lines2 = o3d.geometry.LineSet()
    lines2.points = o3d.utility.Vector3dVector(corners_frame)
    lines2.lines = o3d.utility.Vector2iVector(edges)
    lines2.paint_uniform_color([1.0, 0.0, 0.0])  # red = proposed PCA box

    lines3 = o3d.geometry.LineSet()
    lines3.points = o3d.utility.Vector3dVector(bbox_world)
    lines3.lines = o3d.utility.Vector2iVector(edges)
    lines3.paint_uniform_color([0.0, 0.0, 1.0])  # blue = raw GT in canonical frame

    o3d.visualization.draw_geometries([pcd, lines1, lines2, lines3])
