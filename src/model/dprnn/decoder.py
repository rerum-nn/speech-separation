from torch import nn
import torch.nn.functional as F

class DPRNNDecoder(nn.Module):
    def __init__(self, in_channels=512, kernel_size=3, stride=1, padding=1):
        super().__init__()
        
        self.conv_transpose = nn.ConvTranspose1d(in_channels=in_channels, out_channels=1, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        
    def forward(self, x):
        x = self.conv_transpose(x)
        x = x.squeeze(1)
        return x
