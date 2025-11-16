import torch
from torch import nn
from src.model.rtfs_net_layers.utils import Permute
import torch.nn.functional as F
import math


class CompressionPhase(nn.Module):
    def __init__(self, in_channels, hid_channels, compression_blocks, kernel_size=4, stride=2):
        super().__init__()

        assert hid_channels < in_channels, "hid_channels must be less than in_channels"

        self.downstream = nn.Conv2d(in_channels, hid_channels, kernel_size=1, stride=1, padding=0)
        self.compression_phase = nn.ModuleList()

        for _ in range(compression_blocks):
            self.compression_phase.append(
                nn.Sequential(
                    nn.Conv2d(hid_channels, hid_channels, kernel_size=kernel_size, stride=stride, padding=0, groups=hid_channels),
                    nn.BatchNorm2d(hid_channels),
                    nn.ReLU(inplace=True)
                )
            )

    def forward(self, x):
        x = self.downstream(x)

        A = []

        for i in range(len(self.compression_phase)):
            A.append(x)
            x = self.compression_phase[i](x)

        A_resized = []

        target_size = A[-1].shape[2:]
        for i in range(len(A)):
            A_resized.append(F.adaptive_avg_pool2d(A[i], target_size))

        A_G = torch.stack(A_resized, dim=0).sum(dim=0)

        return A_G, A

class DualPathRNN(nn.Module):
    def __init__(self, channel_dim, freq_dim, rnn_layers=1, hidden_dim=32, kernel_size=8, stride=1):
        super().__init__()

        self.freq_dim = freq_dim
        self.kernel_size = kernel_size
        self.stride = stride

        self.unfold = nn.Unfold(kernel_size=(kernel_size, 1), stride=(stride, 1))
        self.ln1 = nn.LayerNorm(channel_dim)
        self.ln2 = nn.LayerNorm(channel_dim)

        self.rnn1 = nn.LSTM(channel_dim * kernel_size, hidden_dim, num_layers=rnn_layers, batch_first=True, bidirectional=True)
        self.rnn2 = nn.LSTM(channel_dim * kernel_size, hidden_dim, num_layers=rnn_layers, batch_first=True, bidirectional=True)

        self.tconv1 = nn.ConvTranspose1d(hidden_dim * 2, channel_dim, kernel_size=kernel_size, stride=stride)
        self.tconv2 = nn.ConvTranspose1d(hidden_dim * 2, channel_dim, kernel_size=kernel_size, stride=stride)

        padded_freq_dim = math.ceil((freq_dim - kernel_size) / stride) * stride + kernel_size

    def forward(self, x):
        B, C, old_T, old_F = x.shape
        
        padded_freq_dim = math.ceil((self.freq_dim - self.kernel_size) / self.stride) * self.stride + self.kernel_size
        padded_time_dim = math.ceil((old_T - self.kernel_size) / self.stride) * self.stride + self.kernel_size

        x = F.pad(x, (0, padded_freq_dim - old_F, 0, padded_time_dim - old_T))
        
        x_residual = x 
        x = x.permute(0, 2, 3, 1)
        x = self.ln1(x)
        x = x.permute(0, 1, 3, 2)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B * padded_time_dim, C, padded_freq_dim)
        x = self.unfold(x.unsqueeze(-1))
        x = x.permute(0, 2, 1)
        x = self.rnn1(x)[0]
        x = x.permute(0, 2, 1)
        x = self.tconv1(x).view(B, padded_time_dim, C, padded_freq_dim)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x + x_residual

        x_residual = x 
        x = x.permute(0, 2, 3, 1)
        x = self.ln2(x)
        x = x.permute(0, 3, 1, 2)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(B * padded_freq_dim , C, padded_time_dim)
        x = self.unfold(x.unsqueeze(-1))
        x = x.permute(0, 2, 1)
        x = self.rnn2(x)[0]
        x = x.permute(0, 2, 1)
        x = self.tconv2(x).view(B, padded_freq_dim, C, padded_time_dim)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x + x_residual

        return x[..., :old_T, :old_F]

class TFSelfAttention(nn.Module):
    def __init__(self, channel_dim, freq_dim, hidden_dim, n_heads=4):
        super().__init__()

        assert channel_dim % n_heads == 0
        assert hidden_dim % freq_dim == 0

        self.n_heads = n_heads
        self.channel_dim = channel_dim
        self.hidden_dim = hidden_dim

        qk_embedding_dim = hidden_dim // freq_dim

        self.Q_conv = nn.ModuleList()
        self.K_conv = nn.ModuleList()
        self.V_conv = nn.ModuleList()
        for i in range(n_heads):
            self.Q_conv.append(
                nn.Sequential(
                    nn.Conv2d(channel_dim, qk_embedding_dim, kernel_size=1),
                    nn.PReLU(),
                    Permute(dims=(0, 2, 1, 3)),
                    nn.LayerNorm([qk_embedding_dim, freq_dim]),
                    Permute(dims=(0, 2, 1, 3))
                )
            )
            self.K_conv.append(
                nn.Sequential(
                    nn.Conv2d(channel_dim, qk_embedding_dim, kernel_size=1),
                    nn.PReLU(),
                    Permute(dims=(0, 2, 1, 3)),
                    nn.LayerNorm([qk_embedding_dim, freq_dim]),
                    Permute(dims=(0, 2, 1, 3))
                )
            )
            self.V_conv.append(
                nn.Sequential(
                    nn.Conv2d(channel_dim, channel_dim // n_heads, kernel_size=1),
                    nn.PReLU(),
                    Permute(dims=(0, 2, 1, 3)),
                    nn.LayerNorm([channel_dim // n_heads, freq_dim]),
                    Permute(dims=(0, 2, 1, 3))
                )
            )
        
        self.attn_concat_proj = nn.Sequential(
            nn.Conv2d(channel_dim, channel_dim, kernel_size=1),
            nn.PReLU(),
            Permute(dims=(0, 2, 1, 3)),
            nn.LayerNorm([channel_dim, freq_dim]),
            Permute(dims=(0, 2, 1, 3))
        )

    def forward(self, x):
        B, C, T, F = x.shape

        Q, K, V = [], [], []
        for i in range(self.n_heads):
            Q.append(self.Q_conv[i](x))
            K.append(self.K_conv[i](x))
            V.append(self.V_conv[i](x))

        Q = torch.cat(Q, dim=0)
        K = torch.cat(K, dim=0)
        V = torch.cat(V, dim=0)

        Q = Q.permute(0, 2, 1, 3)
        Q = Q.reshape(Q.shape[0], Q.shape[1], -1)
        K = K.permute(0, 2, 1, 3)
        K = K.reshape(K.shape[0], K.shape[1], -1)
        V = V.permute(0, 2, 1, 3)
        old_shape = V.shape
        V = V.reshape(V.shape[0], V.shape[1], -1)

        attention = (Q @ K.transpose(-2, -1)) / (self.hidden_dim ** 0.5)
        attention = nn.functional.softmax(attention, dim=2)
        A = attention @ V

        A = A.view(old_shape)
        A = A.permute(0, 2, 1, 3).contiguous()

        x = A.view(self.n_heads, B, self.channel_dim // self.n_heads, T, F)
        x = x.transpose(0, 1)
        x = x.contiguous().view(B, self.channel_dim, T, F)
        x = self.attn_concat_proj(x)

        return x

class AttentionReconstruction(nn.Module):
    def __init__(self, channel_dim, freq_dim):
        super().__init__()

        self.w1 = nn.Sequential(
            nn.Conv2d(channel_dim, channel_dim, kernel_size=1, stride=1, padding=0, groups=channel_dim),
            nn.GroupNorm(num_groups=1, num_channels=channel_dim),
        )
        self.w2 = nn.Sequential(
            nn.Conv2d(channel_dim, channel_dim, kernel_size=1, stride=1, padding=0, groups=channel_dim),
            nn.GroupNorm(num_groups=1, num_channels=channel_dim),
        )
        self.w3 = nn.Sequential(
            nn.Conv2d(channel_dim, channel_dim, kernel_size=1, stride=1, padding=0, groups=channel_dim),
            nn.GroupNorm(num_groups=1, num_channels=channel_dim),
        )

    def forward(self, m, n):
        x1 = F.interpolate(F.sigmoid(self.w1(n)), size=m.shape[2:], mode='nearest')
        x2 = self.w2(m)
        x3 = F.interpolate(self.w3(n), size=m.shape[2:], mode='nearest')

        return x1 * x2 + x3

class RTFSBlock(nn.Module):
    def __init__(
            self, 
            in_dim, 
            hidden_dim, 
            freq_dim, 
            rnn_layers=1, 
            rnn_hidden_dim=32, 
            dual_path_kernel_size=8,
            dual_path_stride=1,
            compression_blocks=2, 
            n_heads=4, 
            compression_kernel_size=4,
            compression_stride=2,
            attention_kernel_size=8,
            attention_stride=1):
        super().__init__()

        self.compression_blocks = compression_blocks

        self.compression_phase = CompressionPhase(in_dim, hidden_dim, compression_blocks, kernel_size=compression_kernel_size, stride=compression_stride)
        self.freq_dim = freq_dim
        for i in range(compression_blocks):
            self.freq_dim = (self.freq_dim - 4) // 2 + 1

        self.dual_path_rnn = DualPathRNN(hidden_dim, self.freq_dim, rnn_layers=rnn_layers, hidden_dim=rnn_hidden_dim, kernel_size=dual_path_kernel_size, stride=dual_path_stride)
        self.tf_self_attention = TFSelfAttention(hidden_dim, freq_dim=self.freq_dim, hidden_dim=self.freq_dim * 4, n_heads=n_heads)

        self.attention_reconstruction = AttentionReconstruction(hidden_dim, self.freq_dim)

        self.upsampling = nn.Conv2d(hidden_dim, in_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x_residual = x

        x, A = self.compression_phase(x)

        x = self.dual_path_rnn(x)
        x_tf_attn_res = x
        x = self.tf_self_attention(x)
        x = x + x_tf_attn_res

        new_A = []
        for Ai in A:
            new_A.append(self.attention_reconstruction(Ai, x))

        upsampled_A = []
        for i in range(self.compression_blocks - 1):
            for i in range(1, len(new_A)):
                A1 = new_A[i - 1]
                A2 = new_A[i]
                reconstructed_A = self.attention_reconstruction(A1, A2)
                upsampled_A.append(reconstructed_A + A[i - 1])
            new_A = upsampled_A
            upsampled_A = []

        assert len(new_A) == 1
        x = self.upsampling(new_A[0]) + x_residual

        return x
        