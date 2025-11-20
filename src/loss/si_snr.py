from typing import Dict

from torch import Tensor, nn
from torchmetrics.audio import PermutationInvariantTraining
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio


class SI_SNR(nn.Module):
    def __init__(self, use_pit: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if use_pit:
            self.loss = PermutationInvariantTraining(
                metric_func=ScaleInvariantSignalNoiseRatio(),
                eval_func="max",
                mode="speaker-wise",
            )
        else:
            self.loss = ScaleInvariantSignalNoiseRatio()

    def forward(self, predicted: Tensor, target: Tensor, **batch) -> Dict[str, Tensor]:
        """
        Args:
            predicted (Tensor): (batch, n_speakers, time)
            target (Tensor): (batch, n_speakers, time)

        Returns:
            dict with:
                loss (Tensor)
        """
        return {"loss": -self.loss(predicted, target)}
