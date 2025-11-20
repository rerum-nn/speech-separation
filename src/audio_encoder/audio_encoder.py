import math
import re

import torch
import torchaudio


class AudioEncoder:
    def __init__(
        self,
        n_fft=400,
        win_length=None,
        hop_length=None,
        window_fn=torch.hann_window,
        input_transform=None,
    ):
        """
        Args:
            n_fft (int): number of FFT points.
            win_length (int): window length. If None, it will be set to n_fft.
            hop_length (int): hop length. If None, it will be set to win_length // 2.
            window_fn (Callable): window function.
            input_transform (Callable): input transform.
        """
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        self.window_fn = window_fn
        self.input_transform = input_transform

    def encode_input(self, audio: torch.Tensor) -> torch.Tensor:
        audio = self.input_transform(audio) if self.input_transform is not None else audio
        return audio

    def encode(
        self, audio: torch.Tensor, device: str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.window_fn(self.win_length).to(device)
        spectrogram = torchaudio.functional.spectrogram(
            audio,
            pad=0,
            normalized=False,
            n_fft=self.n_fft,
            window=window,
            win_length=self.win_length,
            hop_length=self.hop_length,
            power=None,
        )
        magnit, phase = spectrogram.abs(), spectrogram.angle()
        return magnit, phase

    def get_spectrogram(
        self, audio: torch.Tensor, device: str = "cpu"
    ) -> (
        torch.Tensor
    ):  # TODO: Why the method needed? There are encode that does the same
        window = self.window_fn(self.win_length).to(device)
        return torchaudio.functional.spectrogram(
            audio,
            pad=0,
            normalized=False,
            n_fft=self.n_fft,
            window=window,
            win_length=self.win_length,
            hop_length=self.hop_length,
            power=None,
        ).abs()

    def decode(
        self,
        magnit: torch.Tensor,
        phase: torch.Tensor,
        length: int,
        device: str = "cpu",
    ) -> torch.Tensor:
        spectrogram = magnit * torch.exp(1j * phase)
        window = self.window_fn(self.win_length).to(device)
        return torchaudio.functional.inverse_spectrogram(
            spectrogram,
            pad=0,
            normalized=False,
            length=length,
            n_fft=self.n_fft,
            window=window,
            win_length=self.win_length,
            hop_length=self.hop_length,
        )

    def get_input_shape(self, signal_length: int, *args, **kwargs) -> tuple[int, int]:
        sample = torch.randn(1, signal_length)
        res = self.input_transform(sample) if self.input_transform is not None else sample
        return res.shape[1], res.shape[2]

    def get_output_shape(self, signal_length: int) -> tuple[int, int]:
        return self.n_fft // 2 + 1, math.ceil(signal_length / self.hop_length) + 1
