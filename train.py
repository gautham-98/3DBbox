from src.data.dataset import get_dataloader
from src.data.splits import get_splits
from src.models.boxestimator import BoxEstimationNet
from src.training.losses import LossLambda
from src.training.trainer import Trainer
from src.eval.evaluator import Evaluator


# training parameters
EPOCHS = 100
BATCH_SZ = 32 

# data parameters
FRAME = "pca" # canonical frame
NUM_WORKERS = 2
N = 1024 # number of points in pointcloud after resampling

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


# count and print params in model
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: total={total_params:,} trainable={trainable_params:,}")
print(f"Training config: epochs={EPOCHS} batch_size={BATCH_SZ} frame={FRAME} N={N}")
print(f"Data sizes: train={len(trainloader.dataset)} val={len(valloader.dataset)} test={len(testloader.dataset)}")

# trainer init and training
print("\n=== STARTING TRAINING ===")
trainer = Trainer(model, trainloader, valloader, loss_lambda, epochs=EPOCHS)
trainer.train()

# evaluation
CKPT_PATH = "./checkpoints/checkpoint_epoch_best.pth"

# init evaluator and evaluate
print("\n=== STARTING EVALUATION ===")
print(f"Checkpoint: {CKPT_PATH}")
evaluator = Evaluator(
        model,
        CKPT_PATH,
        testloader
    )
evaluator.evaluate()