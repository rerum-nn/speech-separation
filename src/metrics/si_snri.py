from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio

from src.metrics.base_metric import ImprovementPermutationInvariantMetric


class SI_SNRi(ImprovementPermutationInvariantMetric):
    """
    Calculate Scale-Invariant Signal-to-Noise Ratio Improvement (SI-SNRi)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            metric_func=scale_invariant_signal_noise_ratio,
            eval_func="max",
            *args,
            **kwargs
        )
