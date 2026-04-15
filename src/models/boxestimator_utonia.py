import torch
import torch.nn as nn

try:
    import utonia
    from utonia.structure import Point
    UTONIA_AVAILABLE = True
except ImportError:
    UTONIA_AVAILABLE = False
    Point = None


UTONIA_FEAT_DIM = 1224


def _load_utonia(flash: bool = False) -> nn.Module:
    if flash:
        return utonia.load("utonia", repo_id="Pointcept/Utonia")
    return utonia.load(
        "utonia",
        repo_id="Pointcept/Utonia",
        custom_config=dict(enc_patch_size=[1024] * 5, enable_flash=False),
    )


def _upcast(point: Point, levels: int = 2) -> Point:
    """Upsample pooled features back to full point resolution (mirrors demo)."""
    for _ in range(levels):
        parent  = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent
    while "pooling_parent" in point.keys():
        parent  = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = point.feat[inverse]
        point = parent
    return point


class BoxEstimationNetUtonia(nn.Module):
    def __init__(
        self,
        num_clusters: int = 8,
        dropout: float = 0.3,
        flash_attn: bool = False,
        upcast_levels: int = 2,
    ):
        if not UTONIA_AVAILABLE:
            raise ImportError(
                "The 'utonia' package is not installed. "
            )
        super().__init__()
        self.in_channels   = 6
        self.num_clusters  = num_clusters
        self.upcast_levels = upcast_levels

        # utonia backbone
        self.backbone = _load_utonia(flash_attn)
        for p in self.backbone.parameters():
            p.requires_grad = False

 
        # pointnet decoder to compress utonia features to task relevant representation
        self.point_mlp = nn.Sequential(
            nn.Linear(UTONIA_FEAT_DIM, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )

        # fc 
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # heads 
        self.head_translation = nn.Linear(32, 3)

        self.head_rotation = nn.Sequential(
            nn.Linear(32, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(16,6)
        )

        self.head_cluster = nn.Sequential(
            nn.Linear(32, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(16, num_clusters),
        )

        self.head_residual = nn.Sequential(
            nn.Linear(32, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(16, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(8, 3),
        )

        nn.init.zeros_(self.head_rotation[-1].weight)
        self.head_rotation[-1].bias.data = torch.tensor([1., 0., 0., 0., 1., 0.])

        nn.init.zeros_(self.head_translation.weight)
        nn.init.zeros_(self.head_translation.bias)

    def _to_point(self, pc: torch.Tensor) -> Point:
        B, N, _ = pc.shape
        device  = pc.device

        coord = pc[..., :3]   # (B, N, 3)
        color = pc[..., 3:6]  # (B, N, 3)

        # grid_coord per sample so serialization z-order is correct
        coord_min  = coord.min(dim=1, keepdim=True).values          # (B, 1, 3)
        grid_coord = torch.div(coord - coord_min, 0.001,
                               rounding_mode="trunc").int()          # (B, N, 3)

        coord_flat      = coord.reshape(B * N, 3)
        color_flat      = color.reshape(B * N, 3)
        normal_flat     = torch.zeros_like(coord_flat)
        grid_coord_flat = grid_coord.reshape(B * N, 3)

        feat   = torch.cat([coord_flat, color_flat, normal_flat], dim=-1)  # (B*N, 9)
        offset = torch.arange(1, B + 1, device=device, dtype=torch.long) * N

        return Point(coord=coord_flat, grid_coord=grid_coord_flat,
                     feat=feat, offset=offset)

    def forward(self, pc: torch.Tensor):
        B, N, _ = pc.shape

        point: Point = self._to_point(pc)

        with torch.no_grad():
            point = self.backbone(point)

        point = _upcast(point, self.upcast_levels)

        feats = point.feat.reshape(B, N, -1)               # (B, N, 1224)

        feats = self.point_mlp(feats)

        feat = feats.mean(dim=1)                           # (B, 256)

        x = self.fc(feat)                                  # (B, 64)

        return (
            self.head_cluster(x),      # (B, K)
            self.head_rotation(x),     # (B, 6)
            self.head_translation(x),  # (B, 3)
            self.head_residual(x),     # (B, 3)
        )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pc = torch.randn(2, 1024, 6).to(device)

    model = BoxEstimationNetUtonia(num_clusters=8).to(device)
    model.eval()

    with torch.no_grad():
        cluster_logits, pred_6d, pred_tr, pred_residual = model(pc)

    print("cluster_logits:", cluster_logits.shape)   # (2, 8)
    print("pred_6d:       ", pred_6d.shape)          # (2, 6)
    print("pred_tr:       ", pred_tr.shape)           # (2, 3)
    print("pred_residual: ", pred_residual.shape)     # (2, 3)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: trainable={trainable:,}  frozen={total-trainable:,}  total={total:,}")
