import timm
import torch.nn as nn


class FaceOcclusionModel(nn.Module):
    def __init__(self, backbone_name="convnext_tiny.fb_in22k_ft_in1k", pretrained=True,
                 drop_rate=0.3, predict_gender=False):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        num_features = self.backbone.num_features
        self.predict_gender = predict_gender

        self.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(num_features, 256),
            nn.GELU(),
            nn.Dropout(drop_rate / 2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        if predict_gender:
            self.gender_head = nn.Sequential(
                nn.Dropout(drop_rate),
                nn.Linear(num_features, 1),
            )

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.backbone(x)
        occlusion = self.head(features).squeeze(-1)
        if self.predict_gender:
            gender_logit = self.gender_head(features).squeeze(-1)
            return occlusion, gender_logit
        return occlusion
