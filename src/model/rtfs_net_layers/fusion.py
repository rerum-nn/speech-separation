import torch
import torch.nn as nn

from src.model.rtfs_net_layers.global_layer_norm import (
    GlobalLayerNorm1D,
    GlobalLayerNorm2D,
)


class GatedFusion(nn.Module):
    def __init__(self, C_v: int, C_a: int, use_bn2d: bool = False, *args, **kwargs):
        super().__init__()

        self.C_v = C_v
        self.C_a = C_a

        Norm2d = nn.BatchNorm2d if use_bn2d else GlobalLayerNorm2D

        self.p2 = nn.Sequential(
            nn.Conv2d(C_a, C_a, kernel_size=1, groups=C_a, bias=not use_bn2d),
            Norm2d(C_a),
            nn.ReLU(inplace=True),
        )

        self.f2 = nn.Sequential(
            nn.Conv1d(C_v, C_a, kernel_size=1, groups=C_a),
            GlobalLayerNorm1D(C_a),
        )

    def forward(self, audio_embedding, video_embedding):
        """
        Args:
            audio_embedding (Tensor): (B, C_a, T_a, F)
            video_embedding (Tensor): (B, C_v, T_v)

        Returns:
            Tensor: (B, C_a, T_a, F)
        """

        _, _, T_a, _ = audio_embedding.shape

        a_gate = self.p2(audio_embedding)

        v_key = self.f2(video_embedding)

        v_key = torch.nn.functional.interpolate(v_key, size=T_a, mode="nearest")

        return a_gate * v_key.unsqueeze(-1)


class AttentionFusion(nn.Module):
    def __init__(
        self,
        C_v: int,
        C_a: int,
        num_heads: int = 4,
        use_bn2d: bool = False,
        *args,
        **kwargs
    ):
        super().__init__()

        self.num_heads = num_heads

        Norm2d = nn.BatchNorm2d if use_bn2d else GlobalLayerNorm2D

        self.p1 = nn.Sequential(
            nn.Conv2d(C_a, C_a, kernel_size=1, groups=C_a, bias=not use_bn2d),
            Norm2d(C_a),
        )

        self.f1 = nn.Sequential(
            nn.Conv1d(C_v, C_a * num_heads, kernel_size=1, groups=C_a),
            GlobalLayerNorm1D(C_a * num_heads),
        )

    def forward(self, audio_embedding, video_embedding):
        """
        Args:
            audio_embedding (Tensor): (B, C_a, T_a, F)
            video_embedding (Tensor): (B, C_v, T_v)

        Returns:
            Tensor: (B, C_a, T_a, F)
        """

        B, C_a, T_a, F = audio_embedding.shape
        B, C_v, T_v = video_embedding.shape

        a_val = self.p1(audio_embedding)

        v_h = self.f1(video_embedding)

        v_m = v_h.reshape(B, C_a, self.num_heads, T_v)

        v_h = v_m.mean(dim=2)

        v_attn = torch.softmax(v_h, dim=-1)

        v_attn = torch.nn.functional.interpolate(v_attn, size=T_a, mode="nearest")

        return a_val * v_attn.unsqueeze(-1)


class CAF(nn.Module):
    def __init__(
        self,
        C_v: int,
        C_a: int,
        num_heads: int = 4,
        use_bn2d: bool = False,
        *args,
        **kwargs
    ):
        super().__init__()

        self.attention_fusion = AttentionFusion(
            C_v, C_a, num_heads, use_bn2d, *args, **kwargs
        )
        self.gated_fusion = GatedFusion(C_v, C_a, use_bn2d, *args, **kwargs)

    def forward(self, audio_embedding, video_embedding, *args, **kwargs):
        """
        Args:
            audio_embedding (Tensor): (B, C_a, T_a, F)
            video_embedding (Tensor): (B, C_v, T_v)

        Returns:
            Tensor: (B, C_a, T_a, F)
        """
        attention_fusion = self.attention_fusion(audio_embedding, video_embedding)
        gated_fusion = self.gated_fusion(audio_embedding, video_embedding)

        return attention_fusion + gated_fusion  # TODO maybe try smth like concat, *, se
