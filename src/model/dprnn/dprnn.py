from torch import nn
import torch.nn.functional as F
import math


class DPRNNBlock(nn.Module):
    def __init__(self, in_dim, chunk_length, frames, rnn_hidden_dim=128):
        super().__init__()

        self.in_dim = in_dim
        self.chunk_length = chunk_length
        self.frames = frames

        self.intra_rnn = nn.LSTM(self.chunk_length, rnn_hidden_dim, batch_first=True, bidirectional=True)
        self.intra_linear = nn.Linear(rnn_hidden_dim * 2, self.chunk_length)
        self.intra_layer_norm = nn.LayerNorm([self.in_dim, self.frames, self.chunk_length])

        self.inter_rnn = nn.LSTM(self.frames, rnn_hidden_dim, batch_first=True, bidirectional=True)
        self.inter_linear = nn.Linear(rnn_hidden_dim * 2, self.frames)
        self.inter_layer_norm = nn.LayerNorm([self.in_dim, self.chunk_length, self.frames])

    def forward(self, x):
        B, C, K, S = x.shape

        x = x.permute(0, 3, 2, 1)
        x_residual = x
        x = x.view(B * self.frames, self.chunk_length, self.in_dim).contiguous()
        x = self.intra_rnn(x)
        x = self.intra_linear(x)
        x = x.view(B, self.frames, self.chunk_length, self.in_dim)
        x = self.intra_layer_norms(x) + x_residual

        x = x.permute(0, 2, 1, 3)
        x_residual = x
        x = x.view(B * self.chunk_length, self.frames, self.in_dim).contiguous()
        x = self.inter_rnn(x)
        x = self.inter_linear(x)
        x = x.view(B, self.chunk_length, self.frames, self.in_dim)
        x = self.inter_layer_norm(x) + x_residual

        x = x.permute(0, 3, 1, 2)

        return x


class DPRNN(nn.Module):
    def __init__(self, in_dim, length, chunk_length, hop_length, dprnn_blocks=6, rnn_hidden_dim=128):
        super().__init__()

        self.in_dim = in_dim
        self.length = length
        self.chunk_length = chunk_length
        self.hop_length = hop_length
        self.dprnn_blocks = dprnn_blocks
        self.S = math.ceil(self.length / self.hop_length)
        
        self.unfold = nn.Unfold(kernel_size=self.chunk_length, stride=self.hop_length)
        self.fold = nn.Fold(output_size=self.length, kernel_size=self.chunk_length, stride=self.hop_length)

        self.dprnn_blocks = nn.ModuleList([DPRNNBlock(self.in_dim, self.chunk_length, self.S, rnn_hidden_dim) for _ in range(self.dprnn_blocks)])

        self.prelu = nn.PReLU()
        self.masks_conv = nn.Conv2d(self.in_dim, self.in_dim, 1)

        self.out = nn.Sequential(nn.Conv1d(self.in_dim, self.in_dim, 1), nn.Tanh())
        self.gate = nn.Sequential(nn.Conv1d(self.in_dim, self.in_dim, 1), nn.Sigmoid())

        self.end_conv = nn.Conv1d(self.in_dim, self.in_dim, 1, bias=False)

    def forward(self, x):
        # Segmentation
        B, C, L = x.shape
        left_padding = self.chunk_length - self.hop_length
        right_padding = (L % self.hop_length) + self.chunk_length - self.hop_length
        x = F.pad(x, (left_padding, right_padding))
        print(x.shape)
        x = self.unfold(x)
        x = x.reshape(B, C, self.chunk_length, self.S)
        print(x.shape)
        # Block processing
        for block in self.dprnn_blocks:
            x = block(x)

        x = self.prelu(x)
        x = self.masks_conv(x)
        x = x.reshape(B * 2, self.in_dim, self.chunk_length, self.S)

        # Overlap-add
        print(x.shape)
        x = self.fold(x)
        print(x.shape)
        x = x[:,:, left_padding:left_padding + L]
        print(x.shape)

        x = self.out(x) * self.gate(x)
        x = self.end_conv(x)
        x = F.relu(x)
        x = x.reshape(B, 2, self.in_dim, L)

        return x
