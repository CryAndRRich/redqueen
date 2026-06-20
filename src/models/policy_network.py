from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
        if module.bias is not None:
            nn.init.zeros_(module.bias)


_H_FULL: int = 13
_W_FULL: int = 13
_H_DOWN: int = 7
_W_DOWN: int = 7


class ResBlock(nn.Module):
    def __init__(self, channels: int, h: int, w: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm([channels, h, w])
        self.norm2 = nn.LayerNorm([channels, h, w])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.norm1(self.conv1(x)), inplace=True)
        out = self.norm2(self.conv2(out))
        return F.relu(out + residual, inplace=True)


class _SpatialEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(15, 64, kernel_size=3, padding=1)
        self.res1 = ResBlock(64, _H_FULL, _W_FULL)
        self.res2 = ResBlock(64, _H_FULL, _W_FULL)
        self.downsample = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.res3 = ResBlock(128, _H_DOWN, _W_DOWN)
        self.flatten = nn.Flatten()
        self.out_dim: int = 128 * _H_DOWN * _W_DOWN

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.stem(x), inplace=True)
        x = self.res1(x)
        x = self.res2(x)
        x = F.relu(self.downsample(x), inplace=True)
        x = self.res3(x)
        return self.flatten(x)


class _AuxEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )
        self.out_dim: int = 32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


FEATURES_DIM = 256


class BomberCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = FEATURES_DIM) -> None:
        super().__init__(observation_space, features_dim)
        self.spatial_enc = _SpatialEncoder()
        self.aux_enc = _AuxEncoder()
        fusion_in = self.spatial_enc.out_dim + self.aux_enc.out_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, features_dim),
            nn.ReLU(inplace=True),
        )
        self.apply(_init_weights)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        spatial_feat = self.spatial_enc(observations["spatial"])
        aux_feat = self.aux_enc(observations["aux"])
        combined = torch.cat([spatial_feat, aux_feat], dim=1)
        return self.fusion(combined)


class BomberPolicyNet(nn.Module):
    def __init__(self, features_dim: int = FEATURES_DIM) -> None:
        super().__init__()
        self.spatial_enc = _SpatialEncoder()
        self.aux_enc = _AuxEncoder()
        fusion_in = self.spatial_enc.out_dim + self.aux_enc.out_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, features_dim),
            nn.ReLU(inplace=True),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(features_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 6),
        )
        self.value_head = nn.Sequential(
            nn.Linear(features_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.apply(_init_weights)

    def forward(
        self,
        spatial: torch.Tensor,
        aux: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spatial_feat = self.spatial_enc(spatial)
        aux_feat = self.aux_enc(aux)
        combined = torch.cat([spatial_feat, aux_feat], dim=1)
        fused = self.fusion(combined)
        return self.policy_head(fused), self.value_head(fused)

    def get_action_logits(
        self,
        spatial: torch.Tensor,
        aux: torch.Tensor,
    ) -> torch.Tensor:
        logits, _ = self.forward(spatial, aux)
        return logits

    def save(self, path: str) -> None:
        torch.save({"model_state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "BomberPolicyNet":
        net = cls()
        ckpt = torch.load(path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        net.load_state_dict(state)
        net.eval()
        return net

    def load_from_sb3(self, sb3_policy) -> None:
        fe = sb3_policy.features_extractor
        pol = sb3_policy
        self.spatial_enc.load_state_dict(fe.spatial_enc.state_dict())
        self.aux_enc.load_state_dict(fe.aux_enc.state_dict())
        self.fusion.load_state_dict(fe.fusion.state_dict())
        self.policy_head[0].load_state_dict(pol.mlp_extractor.policy_net[0].state_dict())
        self.policy_head[2].load_state_dict(pol.action_net.state_dict())
        self.value_head[0].load_state_dict(pol.mlp_extractor.value_net[0].state_dict())
        self.value_head[2].load_state_dict(pol.value_net.state_dict())

    def init_from_bc(self, bc_path: str, device: str = "cpu") -> None:
        ckpt = torch.load(bc_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        self.load_state_dict(state, strict=False)


def make_observation_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "spatial": spaces.Box(
                low=0.0, high=1.0, shape=(15, 13, 13), dtype=np.float32
            ),
            "aux": spaces.Box(
                low=-1.0, high=2.0, shape=(7,), dtype=np.float32
            ),
        }
    )
