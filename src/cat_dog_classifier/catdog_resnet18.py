import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

class CatDogResNet18(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)

        # Freeze all pretrained layers

        for param in self.backbone.parameters():
            param.requires_grad =False

        # replace final classification layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features,1)

    def forward(self, x):
        return self.backbone(x)
