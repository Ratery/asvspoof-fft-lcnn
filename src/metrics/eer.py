import numpy as np
import torch

from src.metrics.calculate_eer import compute_eer
from src.metrics.epoch_metric import EpochMetric


class EERMetric(EpochMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scores = []
        self._labels = []

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        scores = logits[:, 0] - logits[:, 1]
        self._scores.append(scores.detach().cpu())
        self._labels.append(labels.detach().cpu())
        return None

    def reset(self):
        self._scores.clear()
        self._labels.clear()

    def compute(self):
        if not self._scores:
            raise RuntimeError("Can not calculate EER on empty scores list")

        scores = torch.cat(self._scores).numpy()
        labels = torch.cat(self._labels).numpy()

        bonafide_scores = scores[labels == 1]
        other_scores = scores[labels == 0]

        if len(bonafide_scores) == 0 or len(other_scores) == 0:
            return np.nan

        eer, threshold = compute_eer(bonafide_scores, other_scores)
        return eer
