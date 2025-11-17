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
        n_src: int,
        audio_emb_dim: int,
        bottleneck_chan: int,
        kernel_size: int = 1,
        dim: int = 2,            # 1 for Conv1d, 2 for Conv2d
        mask_act: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__()

        assert audio_emb_dim % 2 == 0, "audio_emb_dim must be even (real + imag)."
        assert dim in (1, 2), "dim must be 1 (Conv1d) or 2 (Conv2d)."

        self.n_src = n_src
        self.audio_emb_dim = audio_emb_dim
        self.bottleneck_chan = bottleneck_chan
        self.kernel_size = kernel_size
        self.dim = dim

        mask_output_chan = n_src * audio_emb_dim

        if dim == 1:
            padding = (kernel_size - 1) // 2
            self.conv = nn.Conv1d(
                in_channels=bottleneck_chan,
                out_channels=mask_output_chan,
                kernel_size=kernel_size,
                padding=padding,
            )
        else: 
            padding = (kernel_size - 1) // 2
            self.conv = nn.Conv2d(
                in_channels=bottleneck_chan,
                out_channels=mask_output_chan,
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
        C = self.audio_emb_dim
        assert C % 2 == 0
        half_c = C // 2

        spatial = audio_mixture_embedding.shape[2:]  

        masks = masks.view(B, self.n_src, 2, half_c, *spatial)
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
        masks = self.mask_act(masks)  

        separated_audio_embedding = self._apply_complex_masks(masks, audio_mixture_embedding)

        return separated_audio_embedding
