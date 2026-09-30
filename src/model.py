# Project model: an ImageNet-pretrained torchvision ResNet-18 with a
# temperature-regression head.
# The official DIR imdb-wiki-dir/train.py uses the ResNet-50 implementation in
# its resnet.py; that model is not copied here.
# The official FDS module is not integrated. LDS acts through the training loss
# and does not change the forward pass in this file.
import torch
import torch.nn as nn

from torchvision.models import (
    ResNet18_Weights,
    resnet18,
)

class TemperatureResNet(nn.Module):
    def __init__(self,pretrained = True):
        super().__init__()
        weights = (
            ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )
        self.backbone = resnet18(weights=weights)
        feature_dim = self.backbone.fc.in_features
        # Project adaptation: replace the classifier with a linear head that predicts temperature in degrees Celsius.
        self.backbone.fc = nn.Identity()
        self.regression_head = nn.Linear(
            in_features=feature_dim,
            out_features=1,
        )


    def forward(self,images):
        features = self.backbone(images)
        temperatures = self.regression_head(features)
        # Return shape [batch_size] to match the temperature labels and LDS weights.
        temperatures = temperatures.squeeze(-1)
        return temperatures

def build_model(pretrained=True):
    model = TemperatureResNet(
        pretrained=pretrained,
    )
    return model

def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

def run_smoke_test():
    model = build_model(pretrained=False)

    images = torch.randn(
        4,
        3,
        224,
        224,
    )

    predictions = model(images)

    print("Input shape:", images.shape)
    print("Output shape:", predictions.shape)

    assert predictions.shape == (4,)


if __name__ == "__main__":
    run_smoke_test()
