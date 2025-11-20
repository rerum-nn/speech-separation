import torch
import torch.nn as nn

from src.model.rtfs_net_layers.global_layer_norm import GlobalLayerNorm2D


class AudioDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        kernel_size=3,
        bias: bool = True,
        n_fft=256,
        win_length=256,
        hop_length=128,
        custom_init: bool = True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=2,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=bias,
        )

        if custom_init:
            nn.init.xavier_uniform_(self.conv.weight)

            if self.conv.bias is not None:
                nn.init.zeros_(self.conv.bias)

    def forward(self, audio_embedding):
        """
        Args:
            audio_embedding (Tensor): (B, C_a, T_a, F)
        Returns:
            Tensor: (B, 1, T_a)
        """

        x = self.conv(audio_embedding)

        magnit, phase = torch.chunk(x, 2, dim=1)

        spectrogram = torch.complex(magnit, phase).squeeze(1).transpose(-1, -2)
        audio = torch.istft(
            spectrogram,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        ).unsqueeze(1)

        return audio
