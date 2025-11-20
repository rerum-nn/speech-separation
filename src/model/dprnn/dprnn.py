from torch import nn
import torch.nn.functional as F
import math


class DPRNNBlock(nn.Module):
    def __init__(self, in_dim, chunk_length, frames, rnn_hidden_dim=128):
        super().__init__()

        self.in_dim = in_dim
        self.chunk_length = chunk_length
        self.frames = frames

        self.intra_rnn = nn.LSTM(self.in_dim, rnn_hidden_dim, batch_first=True, bidirectional=True)
        self.intra_linear = nn.Linear(rnn_hidden_dim * 2, self.in_dim)
        self.intra_layer_norm = nn.GroupNorm(1, self.in_dim)

        self.inter_rnn = nn.LSTM(self.in_dim, rnn_hidden_dim, batch_first=True, bidirectional=True)
        self.inter_linear = nn.Linear(rnn_hidden_dim * 2, self.in_dim)
        self.inter_layer_norm = nn.GroupNorm(1, self.in_dim)

    def forward(self, x):
        B, C, K, S = x.shape

        x_residual = x
        x = x.permute(0, 3, 2, 1)
        x = x.reshape(B * self.frames, self.chunk_length, self.in_dim)
        x = self.intra_rnn(x)[0]
        x = self.intra_linear(x)
        x = x.view(B, self.frames, self.chunk_length, self.in_dim).permute(0, 3, 2, 1)
        x = self.intra_layer_norm(x) + x_residual

        x_residual = x
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(B * self.chunk_length, self.frames, self.in_dim)
        x = self.inter_rnn(x)[0]
        x = self.inter_linear(x)
        x = x.view(B, self.chunk_length, self.frames, self.in_dim).permute(0, 3, 1, 2)
        x = self.inter_layer_norm(x) + x_residual

        return x


class DPRNN(nn.Module):
    def __init__(self, in_dim, length, chunk_length, hop_length, dprnn_blocks=6, rnn_hidden_dim=128):
        super().__init__()

        self.in_dim = in_dim
        self.length = length
        self.chunk_length = chunk_length
        self.hop_length = hop_length
        self.dprnn_blocks = dprnn_blocks
        self.left_padding = self.chunk_length - self.hop_length
        self.right_padding = (self.length % self.hop_length) + self.chunk_length - self.hop_length
        self.padded_length = self.length + self.left_padding + self.right_padding
        self.S = math.ceil(self.length / self.hop_length) + 1
        
        self.unfold = nn.Unfold(kernel_size=(1, self.chunk_length), stride=(1, self.hop_length))
        self.fold = nn.Fold(output_size=(1, self.padded_length), kernel_size=(1, self.chunk_length), stride=(1, self.hop_length))

        self.dprnn_blocks = nn.ModuleList([DPRNNBlock(self.in_dim, self.chunk_length, self.S, rnn_hidden_dim) for _ in range(self.dprnn_blocks)])

        self.prelu = nn.PReLU()
        self.masks_conv = nn.Conv2d(self.in_dim, 2 * self.in_dim, 1)

        self.out = nn.Sequential(nn.Conv1d(self.in_dim, self.in_dim, 1), nn.Tanh())
        self.gate = nn.Sequential(nn.Conv1d(self.in_dim, self.in_dim, 1), nn.Sigmoid())

        self.end_conv = nn.Conv1d(self.in_dim, self.in_dim, 1, bias=False)

    def forward(self, x):
        # Segmentation
        B, C, L = x.shape
        
        x = F.pad(x, (self.left_padding, self.right_padding))
        x = self.unfold(x.unsqueeze(2))
        x = x.reshape(B, C, self.chunk_length, self.S)

        # Block processing
        for block in self.dprnn_blocks:
            x = block(x)

        x = self.prelu(x)
        x = self.masks_conv(x)
        x = x.reshape(B * 2, self.in_dim, self.chunk_length, self.S)

        # Overlap-add
        x = x.view(B * 2, self.in_dim * self.chunk_length, self.S).contiguous()
        x = self.fold(x).squeeze(2)
        x = x[:,:, self.left_padding:self.left_padding + L]

        x = self.out(x) * self.gate(x)
        x = self.end_conv(x)
        x = F.relu(x)
        x = x.reshape(B, 2, self.in_dim, L)

        return x
