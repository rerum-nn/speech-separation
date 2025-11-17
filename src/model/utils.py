from torch import nn

class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()

        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)
