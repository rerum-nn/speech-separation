from torchmetrics.functional.audio import signal_distortion_ratio

from src.metrics.base_metric import ImprovementPermutationInvariantMetric


class SDRi(ImprovementPermutationInvariantMetric):
    """
    Calculate Signal-to-Distortion Ratio Improvement (SDRi)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            metric_func=signal_distortion_ratio, eval_func="max", *args, **kwargs
        )
