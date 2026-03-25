import torch
import torch.nn as nn
from torchvision.models import resnet18


def build_cifar_resnet18(num_classes: int | None = None) -> nn.Module:
    """
    CIFAR-style ResNet-18 stem:
    - 3x3 conv, stride 1
    - no initial maxpool
    """
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    if num_classes is not None:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class ResNet18Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = build_cifar_resnet18()
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.output_dim = in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim, bias=False),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimCLRModel(nn.Module):
    def __init__(self, proj_dim = 128):
        super().__init__()
        self.encoder = ResNet18Encoder()
        self.projector = ProjectionHead(self.encoder.output_dim, proj_dim)

    def forward(self, x: torch.Tensor):
        features = self.encoder(x)
        projections = self.projector(features)
        return features, projections


class CIFARClassifier(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.model = build_cifar_resnet18(num_classes=None)
        self.feature_dim = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def forward_logits_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return self.forward_logits_from_features(features)
