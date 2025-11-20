import torch
import torch.nn as nn


class Scale(nn.Module):
    def __init__(self, factor: float):
        super().__init__()
        # store as buffer so it moves with .to(device) and is saved in state_dict
        self.register_buffer("factor", torch.tensor(factor, dtype=torch.float32))

    def forward(self, x):
        return x.float() * self.factor
