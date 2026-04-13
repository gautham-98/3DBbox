from src.data.dataset import get_dataloader
from src.data.splits import get_splits
from src.models.boxestimator import BoxEstimationNet
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

    return parser.parse_args()


def main():
    args = parse_args()

    EPOCHS = args.epochs
    BATCH_SZ = args.batch_size
    FRAME = args.frame
    N = args.num_points

    NUM_WORKERS = 2

    # prepare dataloaders
    split_paths = get_splits(data_dir="dataset")

    trainloader = get_dataloader(
        split_paths["train"],
        augment=True,
        shuffle=True,
        batch_size=BATCH_SZ,
        num_workers=NUM_WORKERS,
        canonical_frame=FRAME,
        N=N,
    )

    valloader = get_dataloader(
        split_paths["val"],
        augment=False,
        shuffle=False,
        batch_size=BATCH_SZ,
        num_workers=NUM_WORKERS,
        canonical_frame=FRAME,
        N=N,
    )

    testloader = get_dataloader(
        split_paths["test"],
        augment=False,
        shuffle=False,
        batch_size=BATCH_SZ,
        num_workers=NUM_WORKERS,
        canonical_frame=FRAME,
        N=N,
    )

    # loss weighing
    loss_lambda = LossLambda(corner=2, lwh=3, rot=4, tr=1)

    # model
    model = BoxEstimationNet(in_channels=6)

    # count params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model parameters: total={total_params:,} trainable={trainable_params:,}")
    print(f"Training config: epochs={EPOCHS} batch_size={BATCH_SZ} frame={FRAME} N={N}")
    print(
        f"Data sizes: train={len(trainloader.dataset)} val={len(valloader.dataset)} test={len(testloader.dataset)}"
    )

    # training
    print("\n=== STARTING TRAINING ===")
    run_name = f"train_N{N}_frame{FRAME}_ep{EPOCHS}_btc{BATCH_SZ}"

    trainer = Trainer(
        model, trainloader, valloader, loss_lambda, epochs=EPOCHS, run_name=run_name
    )
    trainer.train()

    # evaluation
    CKPT_PATH = "./checkpoints/checkpoint_epoch_best.pth"

    print("\n=== STARTING EVALUATION ===")
    print(f"Checkpoint: {CKPT_PATH}")

    evaluator = Evaluator(model, CKPT_PATH, testloader)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
