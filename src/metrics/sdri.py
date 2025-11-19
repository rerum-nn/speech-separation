from torchmetrics.audio import SignalDistortionRatio

from src.metrics.base_metric import ImprovementPermutationInvariantMetric


class SDRi(ImprovementPermutationInvariantMetric):
    """
    Calculate Signal-to-Distortion Ratio Improvement (SDRi)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            metric_func=SignalDistortionRatio(), eval_func="max", *args, **kwargs
        )
