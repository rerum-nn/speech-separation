from typing import Dict

from torch import Tensor, nn
from torchmetrics.audio import PermutationInvariantTraining
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio


class SI_SNR(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.loss = PermutationInvariantTraining(
            metric_func=scale_invariant_signal_noise_ratio,
            eval_func="max",
            mode="speaker-wise",
        )

    def forward(self, predicted: Tensor, target: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            predicted (Tensor): (batch, n_speakers, time)
            target (Tensor): (batch, n_speakers, time)

        Returns:
            dict with:
                loss (Tensor)
        """
        return {"loss": self.loss(predicted, target)}
