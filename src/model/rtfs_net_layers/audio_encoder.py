import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.rtfs_net_layers.global_layer_norm import GlobalLayerNorm2D

class AudioEncoder(nn.Module):
    def __init__(self, out_channels, kernel_size=3, bias: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=bias,
        )

        self.ln = GlobalLayerNorm2D(out_channels)

    def forward(self, magnit, phase):
        """
        Args:
            magnit (Tensor): (B, 1, T_a, F)
            phase (Tensor): (B, 1, T_a, F)

        Returns:
            Tensor: (B, C_a, T_a, F)
        """

        x = torch.cat([magnit, phase], dim=1)

        x = self.conv(x)
        x = self.ln(x)
        x = F.relu(x)

        return x
