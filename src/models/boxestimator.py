import torch
import torch.nn as nn

class BoxEstimationNet(nn.Module):
    """
    input: point in canonical frame
    output: lwh, translation and rotation residuals in canonical frame
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.in_channels = in_channels
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
        )

        # fully connected 
        self.fc = nn.Sequential(
            nn.Linear(256, 128),   
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # regression head
        self.head_translation = nn.Linear(64, 3)   
        self.head_rotation    = nn.Linear(64, 6)   
        self.head_lwh         = nn.Linear(64, 3)   

    def forward(self, pc: torch.Tensor):
        x = pc.transpose(1, 2)              # (B, C, N)
        x = self.point_mlp(x)               

        # maxpool for single descriptor representing the pointcloud
        x = x.max(dim=2)[0]                 # (B, 256)

        x = self.fc(x)                      # (B, 128)

        delta_t   = self.head_translation(x)   # (B, 3)
        delta_r   = self.head_rotation(x)      # (B, 6)
        delta_lwh = self.head_lwh(x)           # (B, 3)

        return delta_lwh, delta_r, delta_t # delta_r is pred_r in canonical frame same with translation
    
if __name__ == "__main__":
    points = torch.randn(size=(2, 1024, 6), dtype=torch.float32)

    box_estimator = BoxEstimationNet(in_channels=6)

    delta_t, delta_r, delta_lwh = box_estimator(points)

    print(delta_t.shape, delta_r.shape, delta_lwh.shape)
