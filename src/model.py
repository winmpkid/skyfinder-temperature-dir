# 本项目模型：使用 torchvision 的 ImageNet 预训练 ResNet-18，添加温度回归头。
# DIR 官方 imdb-wiki-dir/train.py 使用其 resnet.py 的 ResNet-50；这里未直接复制该模型。
# 当前没有接入官方 FDS 模块；LDS 通过训练损失起作用，不改变本文件的前向传播。
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
        # 本项目适配：移除分类层，保留特征，再用一个线性输出预测摄氏温度。
        self.backbone.fc = nn.Identity()
        self.regression_head = nn.Linear(
            in_features=feature_dim,
            out_features=1,
        )


    def forward(self,images):
        features = self.backbone(images)
        temperatures = self.regression_head(features)
        # 本项目接口：输出 [batch_size]，与温度标签及 LDS 权重形状一致。
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
