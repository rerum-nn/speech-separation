import torch
import torch.nn as nn


class GlobalLayerNorm1D(nn.Module):
    """
    Global Layer Normalization (gLN) from https://arxiv.org/abs/1809.07454
    """

    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        *args,
        **kwargs
    ):
        super().__init__()

        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.gain = nn.Parameter(torch.ones(1, num_channels, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (Tensor): (B, C, T)

        Returns:
            Tensor: (B, C, T)
        """

        mean = x.mean(dim=(1, 2), keepdim=True)
        mean_x2 = (x**2).mean(dim=(1, 2), keepdim=True)

        var = mean_x2 - mean**2

        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        if self.elementwise_affine:
            x_norm = self.gain * x_norm + self.bias

        return x_norm


class GlobalLayerNorm2D(nn.Module):
    """
    Global Layer Normalization (gLN) from https://arxiv.org/abs/1809.07454
    """

    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        *args,
        **kwargs
    ):
        super().__init__()

        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.gain = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (Tensor): (B, C, H, W)

        Returns:
            Tensor: (B, C, H, W)
        """

        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        mean_x2 = (x**2).mean(dim=(1, 2, 3), keepdim=True)

        var = mean_x2 - mean**2

        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        if self.elementwise_affine:
            x_norm = self.gain * x_norm + self.bias

        return x_norm
