import copy

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


NUM_FRAMES = 30
LANDMARK_FEATURE_DIM = 126 + 126 + 6 + 3 + 2


def canonicalize_single_hand(keypoints):
    present = keypoints.abs().sum(dim=(1, 3, 4)) > 0
    swap = (~present[:, 0]) & present[:, 1]
    if not swap.any():
        return keypoints
    canonical = keypoints.clone()
    canonical[swap, :, 0] = keypoints[swap, :, 1]
    canonical[swap, :, 1] = keypoints[swap, :, 0]
    return canonical


def landmark_features(keypoints):
    keypoints = canonicalize_single_hand(keypoints)
    valid = keypoints.abs().sum(dim=(-1, -2)) > 0
    wrist = keypoints[..., 0, :]
    centered = (keypoints - wrist.unsqueeze(-2)) * valid[..., None, None]
    articulation_velocity = torch.zeros_like(centered)
    articulation_velocity[:, 1:] = centered[:, 1:] - centered[:, :-1]
    wrist_velocity = torch.zeros_like(wrist)
    wrist_velocity[:, 1:] = wrist[:, 1:] - wrist[:, :-1]
    both = (valid[..., 0] & valid[..., 1]).unsqueeze(-1)
    interhand = (wrist[..., 1, :] - wrist[..., 0, :]) * both
    return torch.cat(
        (
            centered.flatten(2),
            articulation_velocity.flatten(2),
            wrist_velocity.flatten(2),
            interhand,
            valid.float(),
        ),
        dim=-1,
    )


class BundledRadarIMUFrameEncoder(nn.Module):
    """Encode each synchronous radar/IMU frame independently."""

    def __init__(self, model_dim=128):
        super().__init__()
        radar = resnet18(weights=None)
        radar.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        radar.fc = nn.Identity()
        self.radar = radar
        self.radar_projection = nn.Sequential(
            nn.Linear(512, model_dim), nn.LayerNorm(model_dim)
        )
        self.imu = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(model_dim * 4, model_dim * 2),
            nn.GELU(),
            nn.Linear(model_dim * 2, model_dim),
            nn.LayerNorm(model_dim),
        )

    def forward(self, radar, imu):
        if radar is None or imu is None:
            raise RuntimeError("radar and IMU are an inseparable sensor input")
        if radar.shape[:2] != imu.shape[:2] or imu.shape[-1] != 6:
            raise ValueError(
                f"misaligned radar/IMU shapes: {tuple(radar.shape)}, {tuple(imu.shape)}"
            )
        batch, frames = radar.shape[:2]
        radar_latent = self.radar_projection(
            self.radar(radar.reshape(-1, *radar.shape[2:]))
        ).reshape(batch, frames, -1)
        imu_latent = self.imu(imu)
        return self.fusion(
            torch.cat(
                (
                    radar_latent,
                    imu_latent,
                    radar_latent * imu_latent,
                    radar_latent - imu_latent,
                ),
                dim=-1,
            )
        )


class LandmarkFrameEncoder(nn.Module):
    def __init__(self, model_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(LANDMARK_FEATURE_DIM, model_dim * 2),
            nn.GELU(),
            nn.LayerNorm(model_dim * 2),
            nn.Linear(model_dim * 2, model_dim),
            nn.LayerNorm(model_dim),
        )

    def forward(self, keypoints):
        return self.network(landmark_features(keypoints))


class Predictor(nn.Module):
    def __init__(self, model_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim * 2),
            nn.GELU(),
            nn.Linear(model_dim * 2, model_dim),
        )

    def forward(self, latent):
        return self.network(latent)


class FramewiseBundledEMAJEPA(nn.Module):
    """v019 EMA-asymmetric JEPA with mandatory framewise radar+IMU fusion."""

    def __init__(self, model_dim=128):
        super().__init__()
        self.model_dim = model_dim
        self.enc_sensor = BundledRadarIMUFrameEncoder(model_dim)
        self.enc_landmark = LandmarkFrameEncoder(model_dim)
        self.target_sensor = copy.deepcopy(self.enc_sensor)
        self.target_landmark = copy.deepcopy(self.enc_landmark)
        for module in (self.target_sensor, self.target_landmark):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.predict_sensor_to_landmark = Predictor(model_dim)
        self.predict_landmark_to_sensor = Predictor(model_dim)

    def forward(self, radar, imu, keypoints):
        sensor = self.enc_sensor(radar, imu)
        landmark = self.enc_landmark(keypoints)
        with torch.no_grad():
            target_sensor = self.target_sensor(radar, imu)
            target_landmark = self.target_landmark(keypoints)
        return {
            "sensor": sensor,
            "landmark": landmark,
            "target_sensor": target_sensor,
            "target_landmark": target_landmark,
            "sensor_to_landmark": self.predict_sensor_to_landmark(sensor),
            "landmark_to_sensor": self.predict_landmark_to_sensor(landmark),
        }

    @torch.no_grad()
    def update_targets(self, momentum):
        for online_module, target_module in (
            (self.enc_sensor, self.target_sensor),
            (self.enc_landmark, self.target_landmark),
        ):
            for online, target in zip(
                online_module.parameters(), target_module.parameters()
            ):
                target.data.mul_(momentum).add_(online.data, alpha=1.0 - momentum)
            for online, target in zip(online_module.buffers(), target_module.buffers()):
                if target.dtype.is_floating_point:
                    target.data.mul_(momentum).add_(online.data, alpha=1.0 - momentum)
                else:
                    target.data.copy_(online.data)


class TemporalClassifier(nn.Module):
    """The only temporal model in v101; the frozen encoder remains framewise."""

    def __init__(self, model_dim=128, layers=2, heads=8, dropout=0.1, classes=8):
        super().__init__()
        self.class_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.position = nn.Parameter(torch.zeros(1, NUM_FRAMES + 1, model_dim))
        layer = nn.TransformerEncoderLayer(
            model_dim,
            heads,
            model_dim * 4,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, layers, norm=nn.LayerNorm(model_dim)
        )
        self.classifier = nn.Linear(model_dim, classes)
        nn.init.normal_(self.class_token, std=0.02)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, frames):
        if frames.shape[1] != NUM_FRAMES:
            raise ValueError(f"expected {NUM_FRAMES} tokens, got {frames.shape[1]}")
        cls = self.class_token.expand(frames.shape[0], -1, -1)
        tokens = self.transformer(torch.cat((cls, frames), dim=1) + self.position)
        return self.classifier(tokens[:, 0])


def normalized_regression(prediction, stopped_target):
    prediction = F.normalize(prediction, dim=-1)
    stopped_target = F.normalize(stopped_target.detach(), dim=-1)
    return (2 - 2 * (prediction * stopped_target).sum(dim=-1)).mean()


def sigreg_loss(representation, seed, slices=256, points=17):
    representation = representation.reshape(-1, representation.shape[-1]).float()
    generator = torch.Generator(device=representation.device)
    generator.manual_seed(int(seed))
    directions = torch.randn(
        representation.shape[-1], slices, generator=generator, device=representation.device
    )
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)
    locations = torch.linspace(-5.0, 5.0, points, device=representation.device)
    gaussian_cf = torch.exp(-0.5 * locations.square())
    projected = (representation @ directions).unsqueeze(-1) * locations
    empirical_cf = torch.exp(1j * projected).mean(dim=0)
    error = (empirical_cf - gaussian_cf).abs().square() * gaussian_cf
    return (torch.trapz(error, locations, dim=-1) * representation.shape[0]).mean()


def latent_statistics(representation):
    representation = representation.reshape(-1, representation.shape[-1]).float()
    centered = representation - representation.mean(dim=0)
    std = torch.sqrt(centered.var(dim=0, unbiased=False) + 1e-4)
    covariance = centered.T @ centered / max(representation.shape[0] - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(1e-12)
    probability = eigenvalues / eigenvalues.sum()
    rank = torch.exp(-(probability * probability.log()).sum())
    return std.mean(), rank
