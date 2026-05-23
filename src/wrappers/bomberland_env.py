"""
Single-agent Gymnasium wrapper around BomberEnv.

Exposes a standard Gymnasium interface for one controlled agent (agent_id=0 by default)
while internally stepping the other three agents using rule-based opponents.

Supports sb3-contrib MaskablePPO via the action_masks() method.

Usage:
    from src.wrappers.bomberland_env import SingleAgentBomberEnv
    from agent import GeniusRuleAgent

    opponents = [GeniusRuleAgent(i) for i in range(1, 4)]
    env = SingleAgentBomberEnv(opponents=opponents, agent_id=0, seed=42)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(3)
    mask = env.action_masks()   # (6,) bool
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Project root on path so engine/ and agent/ are importable
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from engine import BomberEnv                                   # noqa: E402
from src.utils.feature_extractor import (                      # noqa: E402
    extract_features, count_boxes, obs_to_dict,
)
from src.logic.action_masking import compute_action_mask       # noqa: E402
from src.training.reward_v2 import compute_reward              # noqa: E402
from src.models.policy_network import make_observation_space   # noqa: E402


class SingleAgentBomberEnv(gym.Env):
    """
    Single-agent Gymnasium wrapper around BomberEnv (4-player FFA).

    Args:
        opponents:  list of 3 rule-based agent objects with .act(obs) method.
                    They fill slots [1, 2, 3] (or 3 minus agent_id slots).
        agent_id:   which player index we control (default 0).
        max_steps:  game length (default 500).
        seed:       base random seed.
    """

    metadata: dict = {"render_modes": []}

    def __init__(
        self,
        opponents: list,
        agent_id: int = 0,
        max_steps: int = 500,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        self._aid = int(agent_id)
        self._max_steps = max_steps
        self._base_seed = seed
        self._opponents = opponents  # length 3, in order of OTHER player ids

        self.observation_space: spaces.Dict = make_observation_space()
        self.action_space: spaces.Discrete = spaces.Discrete(6)

        # Internal engine instance
        self._engine = BomberEnv(max_steps=max_steps, seed=seed)

        # Runtime state
        self._raw_obs: dict | None = None
        self._prev_obs: dict | None = None
        self._step: int = 0
        self._initial_boxes: int = 50
        self._my_kills: int = 0
        self._my_boxes: int = 0
        self._episode_seed: int = seed or 0

        # Build ordered list of [controlled_id, opp0_id, opp1_id, opp2_id]
        all_ids = list(range(4))
        all_ids.remove(self._aid)
        self._opp_ids: list[int] = all_ids  # other 3 player ids in engine order

    # ------------------------------------------------------------------ #
    # Gymnasium interface                                                  #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        if seed is not None:
            self._episode_seed = seed
        else:
            self._episode_seed += 1  # different seed each episode

        raw = self._engine.reset(seed=self._episode_seed)
        self._raw_obs = raw
        self._prev_obs = None
        self._step = 0
        self._initial_boxes = count_boxes(raw["map"])
        self._my_kills = 0
        self._my_boxes = 0

        # Re-init opponents to reset any internal state
        for opp in self._opponents:
            if hasattr(opp, "escape_mode"):
                opp.escape_mode = False

        return self._build_obs(), {}

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        assert self._raw_obs is not None, "Call reset() before step()"

        actions = [0, 0, 0, 0]
        actions[self._aid] = int(action)
        for opp, opp_id in zip(self._opponents, self._opp_ids):
            try:
                actions[opp_id] = int(opp.act(self._raw_obs))
            except Exception:
                actions[opp_id] = 0

        next_raw, terminated, truncated = self._engine.step(actions)

        # Track stats for aux features
        prev_p = np.asarray(self._raw_obs["players"])
        curr_p = np.asarray(next_raw["players"])
        prev_enemies = sum(int(prev_p[i][2]) for i in range(4) if i != self._aid)
        curr_enemies = sum(int(curr_p[i][2]) for i in range(4) if i != self._aid)
        self._my_kills += max(0, prev_enemies - curr_enemies)

        prev_boxes = int((np.asarray(self._raw_obs["map"]) == 2).sum())
        curr_boxes = int((np.asarray(next_raw["map"]) == 2).sum())
        self._my_boxes += max(0, prev_boxes - curr_boxes)

        reward = compute_reward(self._raw_obs, next_raw, self._aid)

        self._prev_obs = self._raw_obs
        self._raw_obs = next_raw
        self._step += 1

        obs_dict = self._build_obs()
        info: dict = {
            "kills": self._my_kills,
            "boxes_destroyed": self._my_boxes,
            "step": self._step,
        }

        return obs_dict, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Return (6,) bool mask for MaskablePPO. True = valid action."""
        if self._raw_obs is None:
            return np.ones(6, dtype=np.bool_)
        return compute_action_mask(self._raw_obs, self._aid)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _build_obs(self) -> dict[str, np.ndarray]:
        boxes_now = int((np.asarray(self._raw_obs["map"]) == 2).sum())
        spatial, aux = extract_features(
            obs=self._raw_obs,
            agent_id=self._aid,
            step=self._step,
            total_steps=self._max_steps,
            initial_boxes=max(1, self._initial_boxes),
            boxes_remaining=boxes_now,
            my_kills=self._my_kills,
            my_boxes_destroyed=self._my_boxes,
        )
        return {"spatial": spatial, "aux": aux}

    # ------------------------------------------------------------------ #
    # Convenience                                                          #
    # ------------------------------------------------------------------ #

    @property
    def raw_obs(self) -> dict | None:
        """Access underlying engine obs (useful for rule-based fallback)."""
        return self._raw_obs

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass
