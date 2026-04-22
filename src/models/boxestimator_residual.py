import torch
import torch.nn as nn


class ResBlock1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, 1),
            nn.BatchNorm1d(out_ch),
        )
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))


class ResBlockFC(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_ch, out_ch),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_ch, out_ch),
            nn.BatchNorm1d(out_ch),
        )
        self.shortcut = nn.Linear(in_ch, out_ch) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x) + self.shortcut(x))


class BoxEstimationResNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_clusters: int = 8, dropout: float = 0.3):
        super().__init__()
        self.in_channels = in_channels
        self.num_clusters = num_clusters

        self.point_mlp = nn.Sequential(
            ResBlock1d(in_channels, 64),
            ResBlock1d(64, 128),
            ResBlock1d(128, 256),
            ResBlock1d(256, 512),
        )

        self.fc = nn.Sequential(
            ResBlockFC(512, 256, dropout),
            ResBlockFC(256, 128, dropout),
            ResBlockFC(128, 64, dropout),
        )

        self.head_translation = nn.Linear(64, 3)
        self.head_rotation = nn.Sequential(
            ResBlockFC(64, 32, dropout),
            nn.Linear(32, 6),
        )
        self.head_cluster = nn.Sequential(
            ResBlockFC(64, 32, dropout),
            nn.Linear(32, num_clusters),
        )
        self.head_residual = nn.Sequential(
            ResBlockFC(64, 32, dropout),
            nn.Linear(32, 3),
        )

        nn.init.zeros_(self.head_rotation[-1].weight)
        self.head_rotation[-1].bias.data = torch.tensor([1., 0., 0., 0., 1., 0.])

        nn.init.zeros_(self.head_translation.weight)
        nn.init.zeros_(self.head_translation.bias)

    def forward(self, pc: torch.Tensor):
        x = pc.transpose(1, 2)        # (B, C, N)
        x = self.point_mlp(x)
        x = x.max(dim=2)[0]           # (B, 512)
        x = self.fc(x)                # (B, 64)

        cluster_logits = self.head_cluster(x)      # (B, K)
        pred_residual  = self.head_residual(x)     # (B, 3)
        pred_6d        = self.head_rotation(x)     # (B, 6)
        pred_tr        = self.head_translation(x)  # (B, 3)

        return cluster_logits, pred_6d, pred_tr, pred_residual


if __name__ == "__main__":
    points = torch.randn(size=(2, 1024, 6), dtype=torch.float32)

    box_estimator = BoxEstimationResNet(in_channels=6, num_clusters=8)

    cluster_logits, pred_6d, pred_tr, pred_residual = box_estimator(points)

    print("cluster_logits:", cluster_logits.shape)   # (2, 8)
    print("pred_6d:       ", pred_6d.shape)          # (2, 6)
    print("pred_tr:       ", pred_tr.shape)           # (2, 3)
    print("pred_residual: ", pred_residual.shape)     # (2, 3)
