import torch
from torch.utils.data import DataLoader

from src.data.dataset import BBox3DDataset, get_dataloader
from src.data.splits import get_splits
from src.models.boxestimator import BoxEstimationNet
from src.models.boxestimator_utonia import BoxEstimationNetUtonia
from src.training.losses import LossLambda
from src.training.trainer import Trainer
from src.eval.evaluator import Evaluator

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Training Box Estimation Network")

    # training parameters
    parser.add_argument(
        "--epochs", type=int, default=400, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")

    # data parameters
    parser.add_argument(
        "--frame",
        type=str,
        default="obb",
        choices=["obb", "pca"],
        help="Canonical frame type",
    )
    parser.add_argument(
        "--num_points", type=int, default=1024, help="Number of points per point cloud"
    )

    # anchor clustering parameters
    parser.add_argument(
        "--num_clusters", type=int, default=8, help="Number of K-means anchor clusters"
    )
    parser.add_argument(
        "--kmeans_path",
        type=str,
        default="kmeans_centers.npy",
        help="Path to save/load K-means cluster centers",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="pointnet",
        help="Model name 'utonia' or 'pointnet'"

    )

    return parser.parse_args()


def main():
    args = parse_args()

    EPOCHS = args.epochs
    BATCH_SZ = args.batch_size
    FRAME = args.frame
    N = args.num_points
    K = args.num_clusters
    MODEL = args.model
    NUM_WORKERS = 2

    split_paths = get_splits(data_dir="dataset")

    # Fit K-means on training data
    print("Fitting K-means on training data...")
    train_ds = BBox3DDataset(
        split_paths["train"],
        augment=True,
        N=N,
        canonical_frame=FRAME,
    )
    centers = train_ds.fit_kmeans(k=K)
    train_ds.save_kmeans(args.kmeans_path)
    print(f"K-means centers saved to '{args.kmeans_path}'  shape={centers.shape}")

    trainloader = DataLoader(
        train_ds,
        batch_size=BATCH_SZ,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    valloader = get_dataloader(
        split_paths["val"],
        augment=False,
        shuffle=False,
        batch_size=BATCH_SZ,
        num_workers=NUM_WORKERS,
        canonical_frame=FRAME,
        N=N,
        kmeans_centers=centers,
    )

    testloader = get_dataloader(
        split_paths["test"],
        augment=False,
        shuffle=False,
        batch_size=BATCH_SZ,
        num_workers=NUM_WORKERS,
        canonical_frame=FRAME,
        N=N,
        kmeans_centers=centers,
    )

    # loss weighing
    loss_lambda = LossLambda(cluster=1.0, residual=1.0, rot=2.0, tr=1.0, corner=2.0)

    # model
    if MODEL == "pointnet":
        model = BoxEstimationNet(in_channels=6, num_clusters=K)
    elif MODEL == "utonia":
        model = BoxEstimationNetUtonia(flash_attn=False, num_clusters=K)
    else:
        raise ValueError("Choose either 'utonia' or 'pointnet'")
    
    # count params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model parameters: total={total_params:,} trainable={trainable_params:,}")
    print(f"Training config: epochs={EPOCHS} batch_size={BATCH_SZ} frame={FRAME} N={N} K={K}")
    print(
        f"Data sizes: train={len(trainloader.dataset)} val={len(valloader.dataset)} test={len(testloader.dataset)}"
    )

    # training
    print("\n=== STARTING TRAINING ===")
    run_name = f"train_N{N}_M{MODEL}_F{FRAME}_EP{EPOCHS}_B{BATCH_SZ}_K{K}"
    ckpt_dir = f"ckpt_N{N}_M{MODEL}_F{FRAME}_EP{EPOCHS}_B{BATCH_SZ}_K{K}"

    kmeans_tensor = torch.from_numpy(centers).float()
    trainer = Trainer(
        model,
        trainloader,
        valloader,
        loss_lambda,
        kmeans_centers=kmeans_tensor,
        epochs=EPOCHS,
        run_name=run_name,
        ckpt_dir=ckpt_dir,
    )
    trainer.train()

    # evaluation
    CKPT_PATH = ckpt_dir + "/checkpoint_epoch_best.pth"

    print("\n=== STARTING EVALUATION ===")
    print(f"Checkpoint: {CKPT_PATH}")

    evaluator = Evaluator(model, CKPT_PATH, testloader)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
