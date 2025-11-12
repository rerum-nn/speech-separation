from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility

from src.metrics.base_metric import PermutationInvariantMetric


class STOI(PermutationInvariantMetric):
    """
    Calculate Short-Time Objective Intelligibility (STOI)
    """

    def __init__(self, fs, *args, **kwargs):
        super().__init__(
            metric_func=ShortTimeObjectiveIntelligibility(fs=fs),
            eval_func="max",
            *args,
            **kwargs
        )
