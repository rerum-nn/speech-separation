import torch
import torch.nn as nn


class AudioEncoder(nn.Module):
    def __init__(self, out_channels, kernel_size=3, bias: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=bias,
        )

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

        return x
