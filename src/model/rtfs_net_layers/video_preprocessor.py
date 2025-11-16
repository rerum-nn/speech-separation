import torch.nn as nn

from src.model.rtfs_net_layers.global_layer_norm import GlobalLayerNorm1D
from src.model.rtfs_net_layers.rtfs_block import (
    AttentionReconstruction,
    CompressionPhase,
)


class ConvFFN(nn.Module):
    def __init__(self, in_channels, expansion_factor, drop=0.1):
        super().__init__()

        self.expand = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=expansion_factor * in_channels,
                kernel_size=1,
                bias=False,
            ),
            GlobalLayerNorm1D(in_channels * expansion_factor),
        )

        self.depth_wise = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels * expansion_factor,
                out_channels=in_channels * expansion_factor,
                kernel_size=5,
                padding=(5 - 1) // 2,
                bias=False,
                groups=in_channels * expansion_factor,
            ),
            GlobalLayerNorm1D(in_channels * expansion_factor),
            nn.ReLU(inplace=True),
        )

        self.project = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels * expansion_factor,
                out_channels=in_channels,
                kernel_size=1,
                bias=False,
            ),
            GlobalLayerNorm1D(in_channels),
        )

    def forward(self, x):
        x = self.expand(x)
        x = self.depth_wise(x)
        x = self.project(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        attn_bias: bool,
        dropout: float,
        is_causal: bool,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.is_causal = is_causal

        self.ln = nn.LayerNorm(input_dim)
        self.mhsa = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=attn_bias,
            batch_first=True,
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.ln(x)
        x, _ = self.mhsa(x, x, x, need_weights=False, is_causal=self.is_causal)
        x = x.transpose(1, 2)
        return x


class TDANetAttentionBlock(nn.Module):
    """
    https://arxiv.org/pdf/2209.15200
    """

    def __init__(self, input_dim, num_heads, attn_bias, dropout):
        super().__init__()

        self.mhsa = MultiHeadAttention(
            input_dim=input_dim,
            num_heads=num_heads,
            attn_bias=attn_bias,
            dropout=dropout,
            is_causal=False,
        )

        self.ffn = ConvFFN(input_dim, 2, drop=dropout)

    def forward(self, x):
        x = x + self.mhsa(x)
        x = x + self.ffn(x)
        return x


class VideoPreprocessor(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        compression_blocks=2,
        compression_kernel_size=4,
        compression_stride=2,
    ):
        super().__init__()

        self.compression_blocks = compression_blocks

        self.compression_phase = CompressionPhase(
            in_dim,
            hidden_dim,
            compression_blocks,
            kernel_size=compression_kernel_size,
            stride=compression_stride,
            is_2d=False,
        )

        self.attention = TDANetAttentionBlock(hidden_dim, 8, False, 0.1)

        self.attention_reconstruction = AttentionReconstruction(hidden_dim, is_2d=False)

        self.upsampling = nn.Conv1d(
            hidden_dim, in_dim, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        x_residual = x

        x, A = self.compression_phase(x)

        x = self.attention(x)

        new_A = []
        for Ai in A:
            new_A.append(self.attention_reconstruction(Ai, x))

        for _ in range(self.compression_blocks - 1):
            next_level = []
            for i in range(1, len(new_A)):
                A1 = new_A[i - 1]
                A2 = new_A[i]
                reconstructed_A = self.attention_reconstruction(A1, A2)
                next_level.append(reconstructed_A + A[i - 1])
            new_A = next_level

        assert len(new_A) == 1
        x = self.upsampling(new_A[0]) + x_residual

        return x
