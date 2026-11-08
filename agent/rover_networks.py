"""Neural components shared by ROVER agents: encoders and the transition projector.

Encoders return non-negative features normalised to unit L1 mass,
``phi(x) >= 0`` with ``||phi(x)||_1 = 1`` (paper Sec. 3, Step 1).  Unit mass is
required by the sink-state construction (Eq. 16-17), which uses
``<1, psi(x, a)> = 1``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class Encoder(nn.Module):
    """MLP encoder for vector observations (one-hot states, PointMaze states)."""

    def __init__(self, obs_shape, hidden_dim: int, feature_dim: int):
        super().__init__()
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.repr_dim = feature_dim
        self.fc = nn.Sequential(
            nn.Linear(obs_shape[0], hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim, bias=False),
            nn.ReLU(),
        )
        self.apply(utils.weight_init)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.view(obs.shape[0], -1)
        return F.normalize(self.fc(obs), p=1, dim=-1)

    def encode_and_project(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs)


class CNNEncoder(nn.Module):
    """Convolutional encoder for uint8 pixel observations."""

    def __init__(self, obs_shape, feature_dim: int):
        super().__init__()
        assert len(obs_shape) == 3
        self.conv = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.repr_dim = 32 * 7 * 7
        self.projector = nn.Sequential(
            nn.Linear(self.repr_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
        )
        self.apply(utils.weight_init)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.adaptive_pool(self.conv(obs / 255.0))
        return h.view(h.shape[0], -1)

    def encode_and_project(self, obs: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projector(self.forward(obs)), p=1, dim=-1)


class ProjectSA(nn.Module):
    """Predicts the next-state embedding from psi(x, a) = phi(x) (x) e_a."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.project_sa = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, encoded_state_action: torch.Tensor) -> torch.Tensor:
        return self.project_sa(encoded_state_action)
