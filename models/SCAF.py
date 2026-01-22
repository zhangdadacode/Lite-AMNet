import torch
import torch.nn as nn
class SCAF(nn.Module):
    def __init__(self, channels=64, r=4):
        super().__init__()
        inter_channels = int(channels // r)

        self.shared_compress = nn.Sequential(
            nn.Conv2d(channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )

        self.local_branch = nn.Sequential(
            nn.Conv2d(inter_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )



    def forward(self, x, residual):
        xa = x + residual

        # Local Attention
        xl_compressed = self.shared_compress(xa)
        xl = self.local_branch(xl_compressed)

        # Global Attention
        xg_pooled = self.avg_pool(xa)
        xg_compressed = self.shared_compress(xg_pooled)
        xg = self.global_branch(xg_compressed)

        # Broadcast global weights to spatial dimensions
        xg = xg.expand_as(xl)

        # Fusion
        weights = self.sigmoid(xl + xg)
        weights = self.enhance(weights) 

        return 2 * (x * weights + residual * (1 - weights))
