import re
from string import ascii_lowercase

import torch
import torchaudio

class AudioEncoder:
    def __init__(self, n_fft=400, win_length=None, hop_length=None, window_fn=torch.hann_window, input_transform=None):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length, window_fn=window_fn, power=None)
        self.inverse_spectrogram = torchaudio.transforms.InverseSpectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length, window_fn=window_fn)

        self.input_transform = input_transform

    def encode_input(self, audio: torch.Tensor) -> torch.Tensor:
        if self.input_transform is not None:
            audio = self.input_transform(audio)
        return audio

    def encode(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spectrogram = self.spectrogram(audio)
        magnit, phase = spectrogram.abs(), spectrogram.angle()
        return magnit, phase

    def get_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        return self.spectrogram(audio).abs()

    def decode(self, magnit: torch.Tensor, phase: torch.Tensor, length: int) -> torch.Tensor:
        spectrogram = magnit * torch.exp(1j * phase)
        return self.inverse_spectrogram(spectrogram, length=length)
