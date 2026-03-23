import torch
import torch.nn as nn
from torchvision.models import resnet18

class ResNet18Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=None)
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
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, proj_dim),
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
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)