import torch
import torch.nn as nn


def center_crop(layer: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    """Center-crop to target_size (th, tw)"""
    _, _, h, w = layer.size()
    th, tw = target_size
    if h == th and w == tw:
        return layer
    dh = (h - th) // 2
    dw = (w - tw) // 2
    return layer[:, :, dh : dh + th, dw : dw + tw]


class DecoderBlock(nn.Module):
    def __init__(self, out_channels):
        super().__init__()

        self.conv_down = nn.Conv2d(
            in_channels=2 * out_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            padding=0,
            dilation=1,
            bias=False,  # because of bn after
        )
        self.bn_down = nn.BatchNorm2d(out_channels // 2)

        self.full_conv = nn.ConvTranspose2d(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            output_padding=1,
            bias=False,  # because of bn after
        )
        self.bn_full = nn.BatchNorm2d(out_channels // 2)

        self.conv_up = nn.Conv2d(
            in_channels=out_channels // 2,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            dilation=1,
            bias=False,  # because of bn after
        )
        self.bn_up = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, down, left):
        x = self.conv_down(down)
        x = self.bn_down(x)
        x = self.relu(x)

        x = self.full_conv(x)
        x = self.bn_full(x)
        x = self.relu(x)

        x = self.conv_up(x)
        x = self.bn_up(x)
        x = self.relu(x)

        if left.shape[-2:] != x.shape[-2:]:
            left = center_crop(left, x.shape[-2:])

        x = x + left

        x = self.relu(x)

        return x


class Decoder(nn.Module):
    def __init__(self, num_filters, num_blocks):
        super().__init__()

        self.blocks = nn.ModuleList()
        for idx in range(num_blocks):
            self.blocks.insert(0, DecoderBlock(num_filters * 2**idx))

    def forward(self, acts):
        up = acts[-1]
        for block, left in zip(self.blocks, acts[-2::-1]):
            up = block(up, left)
        return up
