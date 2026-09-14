"""Shared image encoder with a multiscale expression regression head."""
from torch import nn
from torchvision.models import (
    resnet18, ResNet18_Weights, efficientnet_b0, EfficientNet_B0_Weights,
)


class ContextFusion(nn.Module):
    def __init__(self, backbone='resnet18', scales=3, genes=200, hidden=256, pretrained=False):
        super().__init__()
        if backbone == 'resnet18':
            self.encoder = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
            self.encoder.fc = nn.Identity()
            features = 512
        elif backbone == 'efficientnet_b0':
            self.encoder = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            self.encoder.classifier = nn.Identity()
            features = 1280
        else:
            raise ValueError(f'Unknown backbone: {backbone}')
        self.scales = scales
        self.head = nn.Sequential(
            nn.LayerNorm(features*scales), nn.Dropout(.3),
            nn.Linear(features*scales, hidden), nn.GELU(), nn.Dropout(.2), nn.Linear(hidden, genes),
        )

    def forward(self, images):
        batch, scales, channels, height, width = images.shape
        if scales != self.scales:
            raise ValueError(f'Expected {self.scales} views, received {scales}')
        encoded = self.encoder(images.reshape(batch*scales, channels, height, width))
        return self.head(encoded.reshape(batch, -1))
