import torch
import torch.nn as nn

class BoxEstimationNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_clusters: int = 8, dropout: float = 0.3):
        super().__init__()
        self.in_channels = in_channels
        self.num_clusters = num_clusters
        self.dropout = dropout

        # pointnet shared mlp
        self.point_mlp = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        # fully connected
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # regression heads
        self.head_translation = nn.Linear(64, 3)
        self.head_rotation = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(16, 6),
        )

        # cluster classification head 
        self.head_cluster = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(16, num_clusters),  # raw logits, no softmax
        )

        # residual regression head 
        self.head_residual = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(16, 3),
        )

    def forward(self, pc: torch.Tensor):
        x = pc.transpose(1, 2)   # (B, C, N)
        x = self.point_mlp(x)

        # global max-pool
        x = x.max(dim=2)[0]      # (B, 256)

        x = self.fc(x)            # (B, 64)

        cluster_logits = self.head_cluster(x)     # (B, K)
        pred_residual  = self.head_residual(x)    # (B, 3)
        pred_6d        = self.head_rotation(x)    # (B, 6)
        pred_tr        = self.head_translation(x) # (B, 3)

        return cluster_logits, pred_6d, pred_tr, pred_residual


if __name__ == "__main__":
    points = torch.randn(size=(2, 1024, 6), dtype=torch.float32)

    box_estimator = BoxEstimationNet(in_channels=6, num_clusters=8)

    cluster_logits, pred_6d, pred_tr, pred_residual = box_estimator(points)

    print("cluster_logits:", cluster_logits.shape)   # (2, 8)
    print("pred_6d:       ", pred_6d.shape)          # (2, 6)
    print("pred_tr:       ", pred_tr.shape)           # (2, 3)
    print("pred_residual: ", pred_residual.shape)     # (2, 3)
