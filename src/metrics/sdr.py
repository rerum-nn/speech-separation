from torchmetrics.functional.audio import signal_distortion_ratio

from src.metrics.base_metric import PermutationInvariantMetric


class SDR(PermutationInvariantMetric):
    """
    Calculate Signal-to-Distortion Ratio (SDR)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            metric_func=signal_distortion_ratio, eval_func="max", *args, **kwargs
        )
