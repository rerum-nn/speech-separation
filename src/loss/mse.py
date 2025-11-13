from typing import Dict

import torch.nn.functional as F
from torch import Tensor, nn
from torchmetrics.audio import PermutationInvariantTraining


class MSE(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.loss = PermutationInvariantTraining(
            metric_func=F.mse_loss, eval_func="min", mode="speaker-wise"
        )

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
