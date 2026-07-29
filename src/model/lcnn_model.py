import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Sequential


class AngularClassifier(nn.Module):
    def __init__(self, num_features: int, num_classes: int, margin_value: int):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.margin_value = margin_value
        self.weights = nn.Parameter(torch.Tensor(num_classes, num_features))

    def forward(self, embeddings):
        eps = 1e-9
        embeddings_norm = torch.linalg.vector_norm(
            embeddings, dim=1, keepdim=True
        ).clamp_min(eps)
        normalized_embeddings = embeddings / embeddings_norm

        normalized_weights = F.normalize(self.weights, dim=1)
        cos_theta = F.linear(
            input=normalized_embeddings, weight=normalized_weights
        ).clamp(min=-1.0 + eps, max=1.0 - eps)
        theta = torch.acos(cos_theta)

        margin_logits = torch.cos(self.margin_value * theta) * embeddings_norm
        logits = cos_theta * embeddings_norm

        return logits, margin_logits


class MFMActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = torch.chunk(x, chunks=2, dim=1)
        return torch.max(out[0], out[1])


class LCNNModel(nn.Module):
    def __init__(self, angular_margin: int, dropout: int):
        super().__init__()

        self.net = Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2
            ),
            MFMActivation(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0
            ),
            MFMActivation(),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(
                in_channels=32, out_channels=96, kernel_size=3, stride=1, padding=1
            ),
            MFMActivation(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(num_features=48),
            nn.Conv2d(
                in_channels=48, out_channels=96, kernel_size=1, stride=1, padding=0
            ),
            MFMActivation(),
            nn.BatchNorm2d(num_features=48),
            nn.Conv2d(
                in_channels=48, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            MFMActivation(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0
            ),
            MFMActivation(),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            MFMActivation(),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0
            ),
            MFMActivation(),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            MFMActivation(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Flatten(),
            nn.Linear(in_features=32 * 53 * 37, out_features=160),
            MFMActivation(),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(num_features=80),
        )

        self.classifier = AngularClassifier(
            num_features=80, num_classes=2, margin_value=angular_margin
        )

    def forward(self, audio, **batch):
        """
        Model forward method.

        Args:
            audio (Tensor): input audio.
        Returns:
            output (dict): output dict containing logits and margin logits for a-softmax.
        """
        embeddings = self.net(audio)
        logits, margin_logits = self.classifier(embeddings)
        return {"logits": logits, "margin_logits": margin_logits}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
