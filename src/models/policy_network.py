"""
Policy network for Bomberland — compatible with SB3 MaskablePPO.

Architecture:
  Spatial branch : Conv(15→64) → ResBlock(64) → ResBlock(64)
                   → Conv(64→128, stride=2) → ResBlock(128) → flatten → 3200
  Aux branch     : Linear(7→32→32)
  Fusion head    : Linear(3232→256) → ReLU (BomberCNNExtractor output)
  Actor head     : Linear(256→256) → ReLU → Linear(256→6)
  Value head     : Linear(256→256) → ReLU → Linear(256→1)

Exposed classes:
  BomberCNNExtractor  — BaseFeaturesExtractor for SB3 MultiInputPolicy
  BomberPolicyNet     — standalone Actor-Critic (used for BC and ONNX export)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ─────────────────────────────────────────────────────────────────────────── #
# Weight initialisation helper                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

def _init_weights(module: nn.Module) -> None:
    """Orthogonal init for Conv2d and Linear layers; zero bias."""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# ─────────────────────────────────────────────────────────────────────────── #
# Residual block                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

# Grid size after stride-2 conv: ceil(13/2) = 7
_H_FULL: int = 13
_W_FULL: int = 13
_H_DOWN: int = 7   # after stride-2 conv
_W_DOWN: int = 7


class ResBlock(nn.Module):
    """
    ResNet-style residual block with two 3×3 convolutions and LayerNorm.
    Both spatial dimensions (H, W) must be provided so that LayerNorm shapes
    are fixed at construction time — required for ONNX export.
    """

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


# ─────────────────────────────────────────────────────────────────────────── #
# Spatial encoder                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

class _SpatialEncoder(nn.Module):
    """
    ResNet-style CNN: (B, 15, 13, 13) → (B, 3200).

    Stage 1  — Conv(15→64, 3×3, pad=1)   : (B,  64, 13, 13)
    Stage 2  — ResBlock(64, 13, 13)       : (B,  64, 13, 13)
    Stage 3  — ResBlock(64, 13, 13)       : (B,  64, 13, 13)
    Stage 4  — Conv(64→128, 3×3, stride=2): (B, 128,  7,  7)
    Stage 5  — ResBlock(128, 7, 7)        : (B, 128,  7,  7)
    Flatten                               : (B, 6272)  → out_dim = 128*7*7 = 6272

    NOTE: 128*7*7 = 6272, not 3200. The docstring header used 3200 as an
    approximation; the actual dimension is computed from the architecture.
    """

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(15, 64, kernel_size=3, padding=1)
        self.res1 = ResBlock(64, _H_FULL, _W_FULL)
        self.res2 = ResBlock(64, _H_FULL, _W_FULL)
        # stride=2: 13 → 7
        self.downsample = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.res3 = ResBlock(128, _H_DOWN, _W_DOWN)
        self.flatten = nn.Flatten()
        self.out_dim: int = 128 * _H_DOWN * _W_DOWN  # 6272

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.stem(x), inplace=True)
        x = self.res1(x)
        x = self.res2(x)
        x = F.relu(self.downsample(x), inplace=True)
        x = self.res3(x)
        return self.flatten(x)


# ─────────────────────────────────────────────────────────────────────────── #
# Aux encoder                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

class _AuxEncoder(nn.Module):
    """MLP that maps (B, 7) → (B, 32)."""

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


# ─────────────────────────────────────────────────────────────────────────── #
# SB3 features extractor (used with MaskablePPO + MultiInputPolicy)            #
# ─────────────────────────────────────────────────────────────────────────── #

FEATURES_DIM = 256


class BomberCNNExtractor(BaseFeaturesExtractor):
    """
    SB3 BaseFeaturesExtractor for Dict observation space.
    Expected obs keys: "spatial" (15,13,13), "aux" (7,).
    Output: (B, FEATURES_DIM) float tensor.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = FEATURES_DIM) -> None:
        super().__init__(observation_space, features_dim)
        self.spatial_enc = _SpatialEncoder()
        self.aux_enc = _AuxEncoder()
        fusion_in = self.spatial_enc.out_dim + self.aux_enc.out_dim  # 6272 + 32 = 6304
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, features_dim),
            nn.ReLU(inplace=True),
        )
        # Orthogonal init for all conv and linear layers
        self.apply(_init_weights)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        spatial_feat = self.spatial_enc(observations["spatial"])
        aux_feat = self.aux_enc(observations["aux"])
        combined = torch.cat([spatial_feat, aux_feat], dim=1)
        return self.fusion(combined)


# ─────────────────────────────────────────────────────────────────────────── #
# Standalone Actor-Critic (for BC training and ONNX export)                   #
# ─────────────────────────────────────────────────────────────────────────── #

class BomberPolicyNet(nn.Module):
    """
    Standalone Actor-Critic network.
    Used for:
      - Behavioral Cloning (supervised training)
      - ONNX export
      - Weight loading/saving independent of SB3

    forward() returns (action_logits, value):
      action_logits: (B, 6)  — raw logits before softmax
      value:         (B, 1)  — state value estimate
    """

    def __init__(self, features_dim: int = FEATURES_DIM) -> None:
        super().__init__()
        self.spatial_enc = _SpatialEncoder()
        self.aux_enc = _AuxEncoder()
        fusion_in = self.spatial_enc.out_dim + self.aux_enc.out_dim  # 6304

        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, features_dim),
            nn.ReLU(inplace=True),
        )

        # Deeper actor and value heads (Improvement 3)
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

        # Orthogonal init for all conv and linear layers
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
        """Policy-head only — used for BC training and inference."""
        logits, _ = self.forward(spatial, aux)
        return logits

    # ------------------------------------------------------------------ #
    # Checkpoint helpers                                                   #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """Save model weights."""
        torch.save({"model_state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "BomberPolicyNet":
        """Load model weights from checkpoint."""
        net = cls()
        ckpt = torch.load(path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        net.load_state_dict(state)
        net.eval()
        return net

    # ------------------------------------------------------------------ #
    # SB3 weight transfer                                                  #
    # ------------------------------------------------------------------ #

    def load_from_sb3(self, sb3_policy) -> None:
        """
        Copy weights from a trained SB3 MaskablePPO policy into this network.
        Relies on matching sub-module names between BomberPolicyNet and
        BomberCNNExtractor (spatial_enc, aux_enc, fusion).
        """
        fe = sb3_policy.features_extractor
        self.spatial_enc.load_state_dict(fe.spatial_enc.state_dict())
        self.aux_enc.load_state_dict(fe.aux_enc.state_dict())
        # SB3 MultiInputPolicy stores fusion inside features_extractor
        self.fusion[0].load_state_dict(fe.fusion[0].state_dict())

    def init_from_bc(self, bc_path: str, device: str = "cpu") -> None:
        """Load BC weights from a tactical_bc.py checkpoint."""
        ckpt = torch.load(bc_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        self.load_state_dict(state, strict=False)


# ─────────────────────────────────────────────────────────────────────────── #
# Observation space factory (shared by env wrapper and BC trainer)             #
# ─────────────────────────────────────────────────────────────────────────── #

def make_observation_space() -> spaces.Dict:
    """Return SB3-compatible Dict observation space."""
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
