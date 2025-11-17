from torch import nn
import torch.nn.functional as F

class DPRNNDecoder(nn.Module):
    def __init__(self, out_channels=512, kernel_size=3, stride=1, padding=1):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = F.relu(x1)
        x2 = self.conv2(x)
        x2 = F.sigmoid(x2)

        return x1 * x2
