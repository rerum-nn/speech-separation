import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class Encoder(nn.Module):
    def __init__(self, num_blocks, in_channels=1, num_filters=64):
        super().__init__()
        self.num_blocks = num_blocks

        self.blocks = nn.ModuleList()
        current_channels = in_channels

        for i in range(num_blocks):
            out_channels = num_filters * (2**i)

            block = EncoderBlock(current_channels, out_channels)
            self.blocks.append(block)

            current_channels = out_channels

    def forward(self, x):
        activations = []

        for block in self.blocks:
            x = block(x)
            activations.append(x)

            x = F.max_pool2d(x, kernel_size=2, stride=2)

        return activations
