import timm
import torch.nn as nn


class FaceOcclusionModel(nn.Module):
    def __init__(self, backbone_name="convnext_tiny.fb_in22k_ft_in1k", pretrained=True, drop_rate=0.3):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        num_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(num_features, 1),
            nn.Sigmoid(),
        )

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features).squeeze(-1)
