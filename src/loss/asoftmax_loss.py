import torch
import torch.nn.functional as F
from torch import nn


class ASoftmaxLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        margin_logits: torch.Tensor,
        labels: torch.Tensor,
        **batch
    ):
        one_hot = F.one_hot(labels, num_classes=logits.size(1)).float()
        final_logits = one_hot * margin_logits + (1 - one_hot) * logits

        return {"loss": F.cross_entropy(final_logits, labels)}
