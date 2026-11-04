import torch
from torch import nn
from torchvision.models import resnet18


NUM_FRAMES = 30


class RadarFrameCNN(nn.Module):
    """Shared CNN compression for each radar frame."""

    def __init__(self, model_dim=128):
        super().__init__()
        backbone = resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projection = nn.Sequential(
            nn.Linear(512, model_dim),
            nn.LayerNorm(model_dim),
        )

    def forward(self, radar):
        if radar.ndim != 5 or radar.shape[1] != NUM_FRAMES or radar.shape[2] != 1:
            raise ValueError(f"expected [B,{NUM_FRAMES},1,H,W], got {tuple(radar.shape)}")
        batch, frames = radar.shape[:2]
        features = self.backbone(radar.reshape(-1, *radar.shape[2:]))
        return self.projection(features).reshape(batch, frames, -1)


class RadarCNNTransformerClassifier(nn.Module):
    """Per-frame radar CNN followed by direct 30-frame classification."""

    def __init__(
        self,
        model_dim=128,
        transformer_layers=2,
        attention_heads=8,
        dropout=0.1,
        classes=8,
    ):
        super().__init__()
        self.frame_cnn = RadarFrameCNN(model_dim)
        self.class_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.position = nn.Parameter(torch.zeros(1, NUM_FRAMES + 1, model_dim))
        layer = nn.TransformerEncoderLayer(
            model_dim,
            attention_heads,
            model_dim * 4,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            layer,
            transformer_layers,
            norm=nn.LayerNorm(model_dim),
        )
        self.classifier = nn.Linear(model_dim, classes)
        nn.init.normal_(self.class_token, std=0.02)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, radar, return_tokens=False):
        frame_tokens = self.frame_cnn(radar)
        class_token = self.class_token.expand(frame_tokens.shape[0], -1, -1)
        tokens = torch.cat((class_token, frame_tokens), dim=1) + self.position
        contextual = self.temporal_transformer(tokens)
        logits = self.classifier(contextual[:, 0])
        if return_tokens:
            return logits, frame_tokens, contextual
        return logits
