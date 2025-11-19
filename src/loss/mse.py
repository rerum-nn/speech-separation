from typing import Dict

import torch.nn.functional as F
from torch import Tensor, nn
from torchmetrics.audio import PermutationInvariantTraining


class MSE(nn.Module):
    def __init__(self, use_pit: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if use_pit:
            self.loss = PermutationInvariantTraining(
                metric_func=F.mse_loss, eval_func="min", mode="speaker-wise"
            )
        else:
            self.loss = F.mse_loss

    def forward(self, predicted: Tensor, target: Tensor, **batch) -> Dict[str, Tensor]:
        """
        Args:
            predicted (Tensor): (batch, n_speakers, ...)
            target (Tensor): (batch, n_speakers, ...)

        Returns:
            dict with:
                loss (Tensor)
        """
        return {"loss": self.loss(predicted, target)}
