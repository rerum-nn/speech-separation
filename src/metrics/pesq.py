from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality

from src.metrics.base_metric import PermutationInvariantMetric


class PESQ(PermutationInvariantMetric):
    """
    Calculate Perceptual Evaluation of Speech Quality (PESQ)
    """

    def __init__(self, fs, mode, *args, **kwargs):
        super().__init__(
            metric_func=PerceptualEvaluationSpeechQuality(fs=fs, mode=mode),
            eval_func="max",
            *args,
            **kwargs
        )
