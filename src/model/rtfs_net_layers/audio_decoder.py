import torch
import torch.nn as nn
from src.model.rtfs_net_layers.global_layer_norm import GlobalLayerNorm2D


class AudioDecoder(nn.Module):
    def __init__(self, in_channels, kernel_size=3, bias: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=2,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=bias,
        )

    def forward(self, audio_embedding):
        """
        Args:
            audio_embedding (Tensor): (B, C_a, T_a, F)
        Returns:
            dict with:
                magnit: (B, 1, T_a, F)
                phase: (B, 1, T_a, F)
        """

        # TODO maybe add decoding for all targets
        x = self.conv(audio_embedding)
        
        magnit, phase = torch.chunk(x, 2, dim=1)

        return {"magnit": magnit, "phase": phase}
