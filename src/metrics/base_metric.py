from abc import abstractmethod

from torch import Tensor
from torchmetrics.audio.pit import PermutationInvariantTraining


class BaseMetric:
    """
    Base class for all metrics
    """

    def __init__(self, name=None, *args, **kwargs):
        """
        Args:
            name (str | None): metric name to use in logger and writer.
        """
        self.name = name if name is not None else type(self).__name__

    @abstractmethod
    def __call__(self, **batch):
        """
        Defines metric calculation logic for a given batch.
        Can use external functions (like TorchMetrics) or custom ones.
        """
        raise NotImplementedError()


class PermutationInvariantMetric(BaseMetric):
    def __init__(self, metric_func, eval_func: str = "max", use_pit: bool = True, device: str = "cpu", *args, **kwargs):
        super().__init__(*args, **kwargs)

        if use_pit:
            self.metric = PermutationInvariantTraining(
                metric_func=metric_func, eval_func=eval_func
            ).to(device)
        else:
            self.metric = metric_func.to(device)

    def __call__(self, predicted: Tensor, target: Tensor, **batch):
        """
        Args:
            predicted (Tensor): (batch, n_speakers, time)
            target (Tensor): (batch, n_speakers, time)

        Returns:
            dict with:
                metric (Tensor)
        """
        return self.metric(predicted, target)


class ImprovementPermutationInvariantMetric(PermutationInvariantMetric):
    """
    Computes improvement of a model prediction over the mixture input:

    Improvement = PITMetric(pred, target) - PITMetric(mix, target)
    """

    def __call__(self, predicted: Tensor, target: Tensor, mix: Tensor, **batch):
        """
        Args:
            predicted (Tensor): (batch, n_speakers, time)  
            target (Tensor): (batch, n_speakers, time)
            mix (Tensor): (batch, time)

        Returns:
            dict with:
                metric (Tensor)
        """
        pred_metric = self.metric(predicted, target)
        mix_metric = self.metric(mix.expand_as(target), target)

        return pred_metric - mix_metric
