import torch
import torch.nn as nn


class AudioEncoder(nn.Module):
    def __init__(
        self,
        out_channels,
        kernel_size=3,
        bias: bool = False,
        n_fft=256,
        win_length=256,
        hop_length=128,
        custom_init: bool = True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=bias,
        )

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        if custom_init:
            nn.init.xavier_uniform_(self.conv.weight)

            if self.conv.bias is not None:
                nn.init.zeros_(self.conv.bias)

    def forward(self, audio):
        """
        Args:
            audio (Tensor): (B, 1, T_a)

        Returns:
            Tensor: (B, C_a, T_a, F)
        """

        spectrogram = torch.view_as_real(
            torch.stft(
                audio.squeeze(1),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                return_complex=True,
            )
        )
        magnit, phase = spectrogram[..., 0].unsqueeze(1).transpose(-1, -2), spectrogram[
            ..., 1
        ].unsqueeze(1).transpose(-1, -2)

        x = torch.cat([magnit, phase], dim=1)
        x = self.conv(x)

        return x
