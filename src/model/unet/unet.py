import math

import torch.nn as nn

from src.model.unet.decoder import Decoder
from src.model.unet.encoder import Encoder


class UNet(nn.Module):
    def __init__(self, num_classes=2, num_filters=64, num_blocks=4, *args, **kwargs):
        super().__init__()
        self.encoder = Encoder(num_blocks, num_filters=num_filters)

        self.decoder = Decoder(num_filters=num_filters, num_blocks=num_blocks - 1)

        self.final = nn.Conv2d(
            in_channels=num_filters, out_channels=num_classes, kernel_size=1
        )

    def pad_spectrogram(self, x):
        div = 2**self.encoder.num_blocks
        h, w = x.shape[-2:]
        H = math.ceil(h / div) * div
        W = math.ceil(w / div) * div
        pad_h = H - h
        pad_w = W - w

        x = nn.functional.pad(x, (0, pad_w, 0, pad_h))

        return x, pad_h, pad_w

    def forward(self, input_mix_spectrogram, **batch):
        """
        Model forward method.

        Args:
            input_mix_spectrogram (Tensor): input mix spectrogram.
        Returns:
            output (dict): output dict containing mask1 and mask2.
        """

        spec, pad_h, pad_w = self.pad_spectrogram(input_mix_spectrogram)

        acts = self.encoder(spec)

        x = self.decoder(acts)

        x = self.final(x)

        if pad_h:
            x = x[:, :, :-pad_h, :]
        if pad_w:
            x = x[:, :, :, :-pad_w]

        mask1 = nn.functional.relu(x[:, 0, :, :])
        mask2 = nn.functional.relu(x[:, 1, :, :])

        return {"mask1": mask1, "mask2": mask2}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
