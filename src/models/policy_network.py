"""
Policy network for Bomberland — compatible with SB3 MaskablePPO.

Architecture:
  Spatial branch : Conv2d(15→32→64→64) with stride-2 on last conv → flatten → 3136
  Aux branch     : Linear(7→32→32)
  Fusion head    : Linear(3168→256→128) → split into policy (128→6) and value (128→1)

Exposed classes:
  BomberCNNExtractor  — BaseFeaturesExtractor for SB3 MultiInputPolicy
  BomberPolicyNet     — standalone Actor-Critic (used for BC and ONNX export)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ─────────────────────────────────────────────────────────────────────────── #
# Spatial encoder                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

class _SpatialEncoder(nn.Module):
    """3-layer CNN that maps (B, 15, 13, 13) → (B, 3136)."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(15, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # stride=2 reduces 13→7; output: 64×7×7 = 3136
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.out_dim: int = 64 * 7 * 7  # 3136

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
        fusion_in = self.spatial_enc.out_dim + self.aux_enc.out_dim  # 3168
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, features_dim),
            nn.ReLU(inplace=True),
        )

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
        fusion_in = self.spatial_enc.out_dim + self.aux_enc.out_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, features_dim),
            nn.ReLU(inplace=True),
            nn.Linear(features_dim, 128),
            nn.ReLU(inplace=True),
        )
        self.policy_head = nn.Linear(128, 6)
        self.value_head = nn.Linear(128, 1)

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
        Relies on matching sub-module names.
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
