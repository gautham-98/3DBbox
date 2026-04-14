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
        cache_dir: str = "data_cache",
        canonical_frame: str = "pca",
        kmeans_centers: np.ndarray | None = None,
    ):
        self.augment = augment
        self.N = N
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.canonical_frame = canonical_frame
        self.kmeans_centers = kmeans_centers  # (K, 3) or None

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

    # K-means anchor clustering
    def fit_kmeans(self, k: int) -> np.ndarray:
        """Fit K-means on gt_lwh of all cached items (training split only).

        Returns cluster centers of shape (k, 3) and stores them as
        self.kmeans_centers.
        """
        from sklearn.cluster import KMeans

        all_lwh = [
            np.load(self._inst_cache(*item), allow_pickle=False)["gt_lwh"]
            for item in self.items
        ]
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(np.stack(all_lwh))
        self.kmeans_centers = km.cluster_centers_.astype(np.float32)  # (K, 3)
        return self.kmeans_centers

    def save_kmeans(self, path: str) -> None:
        """Save cluster centers to a .npy file."""
        np.save(path, self.kmeans_centers)

    def load_kmeans(self, path: str) -> None:
        """Load cluster centers from a .npy file."""
        self.kmeans_centers = np.load(path).astype(np.float32)


    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        sp, obj_idx, frame = self.items[idx]
        c = np.load(self._inst_cache(sp, obj_idx, frame), allow_pickle=False)

        pts = torch.from_numpy(c["points"])  # (N, 3)
        cols = torch.from_numpy(c["cols"])  # (N, 3)
        frame_rotation = torch.from_numpy(c["frame_rotation"])  # (3, 3) world→canonical
        frame_translation = torch.from_numpy(c["frame_translation"])  # (3,)
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

        # Compute cluster id and residual from K-means centers
        centers = torch.from_numpy(self.kmeans_centers)  # (K, 3)
        dists = ((centers - gt_lwh.unsqueeze(0)) ** 2).sum(-1)  # (K,)
        cluster_id = dists.argmin().long()  # scalar int64
        residual = gt_lwh / centers[cluster_id] - 1  # (3,) relative residual

        return dict(
            points=pts,                           # (N, 3)
            colors=cols,                          # (N, 3)
            gt_cluster_id=cluster_id,             # () long — anchor cluster index
            gt_residual=residual,                 # (3,) residual from cluster center
            gt_rotation=gt_rotation,              # (3, 3) GT box axes in canonical frame
            gt_translation=gt_translation,        # (3,) GT box translation in canonical frame
            frame_rotation=frame_rotation,        # (3, 3) world to canonical
            frame_translation=frame_translation,  # (3,) canonical frame origin in world
        )


def get_dataloader(
    sample_paths: list,
    augment: bool = False,
    shuffle: bool = False,
    batch_size: int = 32,
    num_workers: int = 4,
    kmeans_centers: np.ndarray | None = None,
    **kwargs,
) -> DataLoader:
    ds = BBox3DDataset(
        sample_paths, augment=augment, kmeans_centers=kmeans_centers, **kwargs
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


if __name__ == "__main__":
    splits = get_splits("dataset")

    # Fit K-means on training data
    train_ds = BBox3DDataset(
        splits["train"][:10],
        augment=False,
        N=1024,
        canonical_frame="pca",
    )
    centers = train_ds.fit_kmeans(k=8)
    print(f"K-means centers shape: {centers.shape}")
    print(f"Cluster centers:\n{centers}")

    # Wrap in DataLoader
    loader = DataLoader(train_ds, batch_size=4, shuffle=False, num_workers=0)

    batch = next(iter(loader))
    print("points:             ", batch["points"].shape)
    print("colors:             ", batch["colors"].shape)
    print("gt_cluster_id:      ", batch["gt_cluster_id"].shape, batch["gt_cluster_id"].dtype)
    print("gt_residual:        ", batch["gt_residual"].shape)
    print("gt_rotation:        ", batch["gt_rotation"].shape)
    print("frame_rotation:     ", batch["frame_rotation"].shape)
    print("frame_translation:  ", batch["frame_translation"].shape)
    print("gt_translation:     ", batch["gt_translation"].shape)
    print(f"Total train items: {len(loader.dataset)}")

    # Sanity check: residual + center should reconstruct gt_lwh
    idx = 0
    cid = batch["gt_cluster_id"][idx].item()
    res = batch["gt_residual"][idx]
    reconstructed = torch.from_numpy(centers[cid]) + res
    print(f"\nSanity check (sample {idx}):")
    print(f"  cluster_id={cid}  center={centers[cid]}  residual={res.numpy()}")
    print(f"  reconstructed lwh = {reconstructed.numpy()}")
