import torch
import torch.nn as nn
from typing import Callable, Optional

class S3MaskGenerator(nn.Module):
    """
    S³-only mask generator: learns a complex mask and applies it
    to a complex audio mixture embedding via explicit complex multiplication.
    """

    def __init__(
        self,
        in_dim: int,
        kernel_size: int = 1,
        n_spks: int = 2,
        dim: int = 2,            # 1 for Conv1d, 2 for Conv2d
        mask_act: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__()

        assert in_dim % 2 == 0, f"bottleneck_chan must be even (real + imag)."
        assert dim in (1, 2), "dim must be 1 (Conv1d) or 2 (Conv2d)."

        self.in_dim = in_dim
        self.dim = dim
        self.n_spks = n_spks

        if dim == 1:
            padding = (kernel_size - 1) // 2
            self.conv = nn.Conv1d(
                in_channels=in_dim,
                out_channels=in_dim * n_spks,
                kernel_size=kernel_size,
                padding=padding,
            )
        else: 
            padding = (kernel_size - 1) // 2
            self.conv = nn.Conv2d(
                in_channels=in_dim,
                out_channels=in_dim * n_spks,
                kernel_size=kernel_size,
                padding=padding,
            )

        self.prelu = nn.PReLU()
        self.mask_act = nn.ReLU() if mask_act is None else mask_act

    def _apply_complex_masks(
        self,
        masks: torch.Tensor,
        audio_mixture_embedding: torch.Tensor,
    ) -> torch.Tensor:
        
        B = audio_mixture_embedding.size(0)
        assert masks.shape[1] % 2 == 0 and audio_mixture_embedding.shape[1] % 2 == 0
        half_c = self.in_dim // 2

        spatial = audio_mixture_embedding.shape[2:]  

        masks = masks.view(B, self.n_spks, 2, half_c, *spatial)
        audio_mixture_embedding = audio_mixture_embedding.view(B, 2, half_c, *spatial)

        mask_real = masks[:, :, 0]  
        mask_imag = masks[:, :, 1]  

        emb_real = audio_mixture_embedding[:, 0].unsqueeze(1)
        emb_imag = audio_mixture_embedding[:, 1].unsqueeze(1) 

        est_real = emb_real * mask_real - emb_imag * mask_imag  
        est_imag = emb_real * mask_imag + emb_imag * mask_real  

        separated_audio_embedding = torch.cat([est_real, est_imag], dim=2) 

        return separated_audio_embedding

    def forward(
        self,
        refined_features: torch.Tensor,
        audio_mixture_embedding: torch.Tensor,
    ) -> torch.Tensor:
        
        x = self.prelu(refined_features)
        masks = self.conv(x)
        masks = masks.view(masks.shape[0], self.n_spks, -1, *masks.shape[2:])
        masks = self.mask_act(masks)  

        separated_audio_embedding = self._apply_complex_masks(masks, audio_mixture_embedding)

        return separated_audio_embedding
