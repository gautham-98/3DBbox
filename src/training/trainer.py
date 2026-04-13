import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import wandb
import datetime

from src.models.boxestimator import BoxEstimationNet
from .losses import loss_lwh, loss_rot, loss_tr, loss_corners, LossLambda
from src.utils.box_utils import get_delta_lwh, reconstruct_bbox
from src.utils.rot_utils import rot6d_to_rotmat
from src.eval.metrics import iou3d

# at beginning of the script
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(
        self,
        model: BoxEstimationNet,
        trainloader: DataLoader,
        valloader: DataLoader,
        loss_lambda: LossLambda,
        epochs: int = 10,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.2,
        betas: list[float, float] = [0.9, 0.999],
        gamma: float = 2.0,
        ckpt_interval: int = 5,
        ckpt_dir: str = "./checkpoints",
        run_name: str = "training",
    ):
        self.model: BoxEstimationNet = model.to(DEVICE)
        self.trainloader = trainloader
        self.valloader = valloader
        self.epochs = epochs
        self.optimizer = self.get_optimizer(learning_rate, weight_decay, betas)
        self.loss_lambda: LossLambda = loss_lambda

        self.ckpt_interval = ckpt_interval
        self.ckpt_dir = ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)

        wandb.init(
            project="BBox3D",
            name=f"{run_name}_{datetime.datetime.now():%Y%m%d_%H%M%S}",
            config={
                "epochs": epochs,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "betas": betas,
                "gamma": gamma,
            },
        )

    def train(self):
        training_step = 0
        best_iou = 0
        for epoch in tqdm(range(self.epochs), desc="epochs", position=0):
            # Training
            self.model.train()

            epoch_loss = dict(
                train_total_loss=0,
                train_lwh_loss=0,
                train_rot_loss=0,
                train_tr_loss=0,
                train_corner_loss=0,
                train_iou_metric=0,
                epoch=epoch,
            )
            for batch in tqdm(
                self.trainloader, desc="training", position=1, leave=False
            ):
                loss_dict = self._step(batch)

                self.optimizer.zero_grad()
                loss_dict["total_loss"].backward()
                self.optimizer.step()

                # accumulate losses for every epoch
                for k, v in loss_dict.items():
                    epoch_loss["train_"+k] += v.item() if isinstance(v, torch.Tensor) else v

                # step losses
                step_loss = dict()
                for k, v in loss_dict.items():
                    step_loss[k] = v.item() if isinstance(v, torch.Tensor) else v

                loss_dict["epoch"] = epoch
                loss_dict["step"] = training_step

                # log step loss
                wandb.log(loss_dict)

                training_step += 1

            # epoch loss averaging across batch
            trainloader_len = len(self.trainloader)
            for k, v in epoch_loss.items():
                if k != "epoch":
                    epoch_loss[k] /= trainloader_len
            # log
            wandb.log(epoch_loss)

            # Validation
            self.model.eval()
            val_loss = dict(
                val_total_loss=0,
                val_lwh_loss=0,
                val_rot_loss=0,
                val_tr_loss=0,
                val_corner_loss=0,
                val_iou_metric=0,
                epoch=epoch,
            )
            with torch.no_grad():
                for batch in tqdm(
                    self.valloader, desc="validating", position=2, leave=False
                ):
                    loss_dict = self._step(batch)
                    # accumulate losses
                    for k, v in loss_dict.items():
                        val_loss["val_"+k] += v.item() if isinstance(v, torch.Tensor) else v

            # average out the losses
            valloader_len = len(self.valloader)
            for k, v in val_loss.items():
                if k != "epoch":
                    val_loss[k] /= valloader_len

            # log
            wandb.log(val_loss)

            tqdm.write(
                f"Epoch [{epoch + 1}/{self.epochs}] | Train Loss: {epoch_loss['train_total_loss']:.4f} | Val Loss: {val_loss['val_total_loss']:.4f} | Val IoU: {val_loss['val_iou_metric']:.4f}"
            )

            # save best
            if val_loss['val_iou_metric'] > best_iou:
                best_iou = val_loss['val_iou_metric']
                self.save_checkpoint("best", val_loss)

            # save every interval
            if (epoch % self.ckpt_interval == 0):
                # save checkpoints
                self.save_checkpoint(epoch, val_loss)
            
            # save last
            if epoch == self.epochs -1:
                self.save_checkpoint("last", val_loss)

    def _step(self, batch):
        # move everything to model device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(DEVICE)

        # extract the inputs and outputs from batch
        points = batch["points"]
        colors = batch["colors"]
        gt_lwh = batch["gt_lwh"]
        gt_tr = batch["gt_translation"]
        gt_rot = batch["gt_rotation"]
        prop_lwh = batch["proposed_lwh"]
        gt_delta_lwh = get_delta_lwh(prop_lwh, gt_lwh)

        # use the colors too in pc
        num_channels = self.model.in_channels
        if num_channels == 3:
            pc = points  # (B,N,3)
        elif num_channels == 6:
            pc = torch.cat([points, colors], dim=-1)  # (B,N,6)
        else:
            raise ValueError(
                "the number of channels can be either 3 or 6 for the model, check model init"
            )

        # model forward
        pred_delta_lwh, pred_6d, pred_tr = self.model(pc)
        pred_lwh = prop_lwh * (1 + pred_delta_lwh)
        pred_rot = rot6d_to_rotmat(pred_6d)

        pred_corners = reconstruct_bbox(pred_lwh, pred_rot, pred_tr)
        gt_corners = reconstruct_bbox(gt_lwh, gt_rot, gt_tr)
        
        # metric
        iou = iou3d(pred_corners, gt_corners)
        
        # loss
        lwh_loss, rot_loss, tr_loss, corner_loss = (
            loss_lwh(pred_delta_lwh, gt_delta_lwh),
            loss_rot(pred_rot, gt_rot),
            loss_tr(pred_tr, gt_tr),
            loss_corners(pred_corners, gt_corners),
        )
        total_loss = (
            self.loss_lambda["lwh"] * lwh_loss
            + self.loss_lambda["rot"] * rot_loss
            + self.loss_lambda["tr"] * tr_loss
            + self.loss_lambda["corner"] * corner_loss
        )

        return dict(
            total_loss=total_loss,
            lwh_loss=lwh_loss,
            rot_loss=rot_loss,
            tr_loss=tr_loss,
            corner_loss=corner_loss,
            iou_metric=iou
        )

    def get_optimizer(self, learning_rate, weight_decay, betas):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
        )
        return optimizer

    def save_checkpoint(self, epoch, val_loss):
        ckpt_path = os.path.join(self.ckpt_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
            },
            ckpt_path,
        )
        print(f"Saved checkpoint at {ckpt_path}")


if __name__ == "__main__":
    from src.data.dataset import get_dataloader
    from src.data.splits import get_splits
    from src.training.trainer import Trainer
    from src.training.losses import LossLambda
    from src.models.boxestimator import BoxEstimationNet

    FRAME = "pca"
    BATCH_SZ = 32
    NUM_WORKERS = 2
    N = 1024

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
    

    loss_lambda = LossLambda(corner=4, lwh=3, rot=2, tr=1)

    model = BoxEstimationNet(in_channels=6)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("total params:", total_params)
    print("trainable params:", trainable_params)

    trainer = Trainer(model, trainloader, valloader, loss_lambda, epochs=100)

    trainer.train()
