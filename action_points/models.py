from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ActionPointModelConfig:
    image_size: int = 224
    bbox_feature_dim: int = 8
    hidden_dim: int = 256
    predict_depth: bool = False
    dropout: float = 0.10


def build_action_point_model(config: ActionPointModelConfig):
    try:
        import torch
        from torch import nn
    except Exception as exc:
        raise RuntimeError("Action-point model requires torch.") from exc

    class SimpleCropEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, 5, stride=2, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 192, 3, stride=2, padding=1),
                nn.BatchNorm2d(192),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            self.out_dim = 192

        def forward(self, image):
            return self.net(image)

    class ResidualMLPBlock(nn.Module):
        def __init__(self, dim: int, dropout: float) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(dim, dim),
            )
            self.act = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.act(x + self.net(x))

    class ActionPointHead(nn.Module):
        def __init__(self, cfg: ActionPointModelConfig) -> None:
            super().__init__()
            self.config = cfg
            self.encoder = SimpleCropEncoder()
            in_dim = self.encoder.out_dim + int(cfg.bbox_feature_dim)
            self.stem = nn.Sequential(
                nn.Linear(in_dim, int(cfg.hidden_dim)),
                nn.ReLU(inplace=True),
                nn.Dropout(float(cfg.dropout)),
            )
            self.blocks = nn.Sequential(
                ResidualMLPBlock(int(cfg.hidden_dim), float(cfg.dropout)),
                ResidualMLPBlock(int(cfg.hidden_dim), float(cfg.dropout)),
            )
            self.xy = nn.Linear(int(cfg.hidden_dim), 2)
            self.depth = nn.Linear(int(cfg.hidden_dim), 1) if cfg.predict_depth else None

        def forward(self, image, bbox_features):
            feat = self.encoder(image)
            x = torch.cat([feat, bbox_features], dim=1)
            x = self.blocks(self.stem(x))
            out = {"xy": torch.sigmoid(self.xy(x))}
            if self.depth is not None:
                out["depth"] = torch.nn.functional.softplus(self.depth(x))
            return out

    return ActionPointHead(config)
