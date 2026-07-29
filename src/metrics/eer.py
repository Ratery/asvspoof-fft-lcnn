import torch

from src.metrics.base_metric import BaseMetric
from src.metrics.calculate_eer import compute_eer


class EERMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        """
        Metric calculation logic.

        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            metric (float): calculated metric.
        """

        logits = logits.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        scores = logits[:, 0]
        bonafide_scores = scores[labels == 1]
        other_scores = scores[labels == 0]

        eer, threshold = compute_eer(bonafide_scores, other_scores)
        return eer
