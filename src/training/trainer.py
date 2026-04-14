import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import wandb
import datetime

from src.models.boxestimator import BoxEstimationNet
from src.models.boxestimator_utonia import BoxEstimationNetUtonia

from .losses import loss_cluster, loss_residual, loss_rot, loss_tr, loss_corners, LossLambda
from src.utils.box_utils import reconstruct_bbox
from src.utils.rot_utils import rot6d_to_rotmat
from src.eval.metrics import iou3d

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(
        self,
        model: BoxEstimationNet|BoxEstimationNetUtonia,
        trainloader: DataLoader,
        valloader: DataLoader,
        loss_lambda: LossLambda,
        kmeans_centers: torch.Tensor,
        epochs: int = 10,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.2,
        betas: list[float, float] = [0.9, 0.999],
        gamma: float = 2.0,
        ckpt_interval: int = 5,
        ckpt_dir: str = "./checkpoints",
        run_name: str = "training",
    ):
        self.model: BoxEstimationNet|BoxEstimationNetUtonia = model.to(DEVICE)
        self.trainloader = trainloader
        self.valloader = valloader
        self.epochs = epochs
        self.optimizer = self.get_optimizer(learning_rate, weight_decay, betas)
        self.loss_lambda: LossLambda = loss_lambda

        # K-means cluster centers on device — shape (K, 3)
        self.kmeans_centers = kmeans_centers.to(DEVICE)

        self.ckpt_interval = ckpt_interval
        self.ckpt_dir = ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.wandb_run = wandb.init(
            project="BBox3D",
            name=f"{run_name}_{datetime.datetime.now():%Y%m%d_%H%M%S}",
            config={
                "epochs": epochs,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "betas": betas,
                "gamma": gamma,
                "num_clusters": kmeans_centers.shape[0],
            },
        )

    def train(self):
        training_step = 0
        best_iou = 0
        for epoch in tqdm(range(self.epochs), desc="epochs", position=0):
            # set progress to schedule lambdas
            self.loss_lambda.set_progress(epoch/self.epochs)

            # log losses            
            lambda_dict = dict(epoch=epoch)
            for key in ["corner", "cluster", "residual", "rot", "tr"]:
                lambda_dict[key] = self.loss_lambda[key]
            wandb.log(lambda_dict)

            # Training
            self.model.train()

            epoch_loss = dict(
                train_total_loss=0,
                train_cluster_loss=0,
                train_residual_loss=0,
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
                val_cluster_loss=0,
                val_residual_loss=0,
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
                self.save_checkpoint(epoch, val_loss)

            # save last
            if epoch == self.epochs - 1:
                self.save_checkpoint("last", val_loss)

    def _step(self, batch):
        # move everything to model device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(DEVICE)

        # extract inputs and targets from batch
        points        = batch["points"]
        colors        = batch["colors"]
        gt_cluster_id = batch["gt_cluster_id"]   # (B,) long
        gt_residual   = batch["gt_residual"]      # (B, 3)
        gt_tr         = batch["gt_translation"]   # (B, 3)
        gt_rot        = batch["gt_rotation"]      # (B, 3, 3)

        # build point cloud input
        num_channels = self.model.in_channels
        if num_channels == 3:
            pc = points
        elif num_channels == 6:
            pc = torch.cat([points, colors], dim=-1)  # (B, N, 6)
        else:
            raise ValueError(
                "the number of channels can be either 3 or 6 for the model, check model init"
            )

        # model forward
        cluster_logits, pred_6d, pred_tr, pred_residual = self.model(pc)
        pred_rot = rot6d_to_rotmat(pred_6d)

        # Reconstruct pred_lwh using soft (differentiable) cluster center weighted sum
        cluster_probs    = cluster_logits.softmax(dim=1)                        # (B, K)
        pred_lwh_cluster = cluster_probs @ self.kmeans_centers                  # (B, 3)
        pred_lwh         = pred_lwh_cluster * (1 + pred_residual)               # (B, 3)

        # GT lwh from cluster center * (1 + relative residual) (for corner loss)
        gt_lwh = self.kmeans_centers[gt_cluster_id] * (1 + gt_residual)        # (B, 3)

        pred_corners = reconstruct_bbox(pred_lwh, pred_rot, pred_tr)
        gt_corners   = reconstruct_bbox(gt_lwh, gt_rot, gt_tr)

        # metric
        iou = iou3d(pred_corners, gt_corners)

        # losses
        clust_loss  = loss_cluster(cluster_logits, gt_cluster_id)
        resid_loss  = loss_residual(pred_residual, gt_residual)
        rot_loss    = loss_rot(pred_rot, gt_rot)
        tr_loss     = loss_tr(pred_tr, gt_tr)
        corner_loss = loss_corners(pred_corners, gt_corners)

        total_loss = (
            self.loss_lambda["cluster"]  * clust_loss
            + self.loss_lambda["residual"] * resid_loss
            + self.loss_lambda["rot"]      * rot_loss
            + self.loss_lambda["tr"]       * tr_loss
            + self.loss_lambda["corner"]   * corner_loss
        )

        return dict(
            total_loss=total_loss,
            cluster_loss=clust_loss,
            residual_loss=resid_loss,
            rot_loss=rot_loss,
            tr_loss=tr_loss,
            corner_loss=corner_loss,
            iou_metric=iou,
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
                "kmeans_centers": self.kmeans_centers.cpu().numpy(),
            },
            ckpt_path,
        )
        if epoch == "last":
            artifact = wandb.Artifact('model-checkpoint', type='model')
            artifact.add_file(self.ckpt_dir+'/checkpoint_epoch_last.pth')
            artifact.add_file(self.ckpt_dir+'/checkpoint_epoch_best.pth')
            self.wandb_run.log_artifact(artifact)

        print(f"Saved checkpoint at {ckpt_path}")


if __name__ == "__main__":
    import numpy as np
    from torch.utils.data import DataLoader
    from src.data.dataset import BBox3DDataset
    from src.data.splits import get_splits
    from src.training.trainer import Trainer
    from src.training.losses import LossLambda
    from src.models.boxestimator import BoxEstimationNet

    FRAME = "pca"
    BATCH_SZ = 32
    NUM_WORKERS = 2
    N = 1024
    K = 8

    split_paths = get_splits(data_dir="dataset")

    # Fit K-means on training data
    train_ds = BBox3DDataset(
        split_paths["train"], augment=True, N=N, canonical_frame=FRAME
    )
    centers = train_ds.fit_kmeans(k=K)

    trainloader = DataLoader(
        train_ds, batch_size=BATCH_SZ, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )

    from src.data.dataset import get_dataloader
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

    loss_lambda = LossLambda(cluster=1.0, residual=0.5, corner=4, rot=2, tr=1)

    model = BoxEstimationNet(in_channels=6, num_clusters=K)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("total params:", total_params)
    print("trainable params:", trainable_params)

    kmeans_tensor = torch.from_numpy(centers).float()
    trainer = Trainer(
        model, trainloader, valloader, loss_lambda,
        kmeans_centers=kmeans_tensor, epochs=100,
    )

    trainer.train()
