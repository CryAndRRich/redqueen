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
from src.training.reward import compute_reward                  # noqa: E402
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
        # Episode stats tracked per-step and exposed to compute_reward()
        self.episode_kills: int = 0
        self.episode_boxes: int = 0
        self.episode_items: int = 0
        # Position history for multi-step stagnation detection.
        # Catches oscillation (A↔B) that single-step standing_still misses.
        self._pos_history: list[tuple[int, int]] = []

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
        self.episode_kills = 0
        self.episode_boxes = 0
        self.episode_items = 0
        self._pos_history = []

        # Re-init opponents to reset any internal state.
        # Call reset() for wrappers that track per-episode state (e.g. PastAgentWrapper);
        # fall back to clearing escape_mode for legacy rule agents that don't expose reset().
        for opp in self._opponents:
            if hasattr(opp, "reset"):
                opp.reset()
            elif hasattr(opp, "escape_mode"):
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

        next_raw, engine_terminated, truncated = self._engine.step(actions)

        # Track stats for aux features
        prev_p = np.asarray(self._raw_obs["players"])
        curr_p = np.asarray(next_raw["players"])
        prev_enemies = sum(int(prev_p[i][2]) for i in range(4) if i != self._aid)
        curr_enemies = sum(int(curr_p[i][2]) for i in range(4) if i != self._aid)
        kills_this_step = max(0, prev_enemies - curr_enemies)
        self._my_kills += kills_this_step
        self.episode_kills += kills_this_step

        # Box destruction: only credit boxes destroyed by MY bombs (timer==1 in prev_obs).
        # Counting all map box disappearances would attribute enemy bomb hits to us,
        # inflating the aux feature and causing mismatched reward signals.
        prev_p_arr = np.asarray(self._raw_obs["players"], dtype=np.int32)
        prev_grid = np.asarray(self._raw_obs["map"], dtype=np.int32)
        curr_grid = np.asarray(next_raw["map"], dtype=np.int32)
        prev_bombs_raw = self._raw_obs.get("bombs")
        my_boxes_this_step = 0
        if prev_bombs_raw is not None:
            prev_bombs_arr = np.asarray(prev_bombs_raw, dtype=np.int32)
            if prev_bombs_arr.ndim == 2 and prev_bombs_arr.shape[0] > 0:
                for row in prev_bombs_arr:
                    bx, by, timer, owner = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                    if owner != self._aid or timer != 1:
                        continue  # only my bombs exploding this step
                    radius = 1 + int(prev_p_arr[self._aid][4])
                    # Vectorized blast expansion along 4 axes, blocked by walls/boxes
                    blast = np.zeros((13, 13), dtype=np.bool_)
                    blast[bx, by] = True
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        for r in range(1, radius + 1):
                            nx, ny = bx + dx * r, by + dy * r
                            if nx < 0 or nx >= 13 or ny < 0 or ny >= 13:
                                break
                            if prev_grid[nx, ny] == 1:  # wall blocks
                                break
                            blast[nx, ny] = True
                            if prev_grid[nx, ny] == 2:  # box blocks further
                                break
                    destroyed = int(np.sum((prev_grid == 2) & (curr_grid != 2) & blast))
                    my_boxes_this_step += destroyed
        self._my_boxes += my_boxes_this_step
        self.episode_boxes += my_boxes_this_step

        # Item collection: detect via players array stat increases (radius_bonus or
        # bombs_left going up), consistent with reward.py detection logic.
        # Map-cell disappearance is unreliable (simultaneous collection destroys the item
        # without either agent collecting it).
        aid = self._aid
        prev_radius_bonus = int(prev_p[aid][4])
        curr_radius_bonus = int(curr_p[aid][4])
        prev_cap = int(prev_p[aid][3])
        curr_cap = int(curr_p[aid][3])
        # capacity can decrease when a bomb is placed; only count increases from pickup
        if int(curr_p[aid][2]) == 1:  # only if alive after step
            if curr_radius_bonus > prev_radius_bonus or curr_cap > prev_cap:
                self.episode_items += 1

        # Increment step BEFORE passing to compute_reward so current_step reflects
        # the step that just completed (1-indexed), matching late-game threshold math.
        self._step += 1

        episode_stats = {
            "kills": self.episode_kills,
            "boxes_destroyed": self.episode_boxes,
            "items_collected": self.episode_items,
        }
        reward, reward_info = compute_reward(
            self._raw_obs, next_raw, int(action), self._aid,
            episode_stats, self._step,
        )

        # Terminate when OUR agent dies, not just when the whole game ends.
        # The engine returns terminated=(alive_count<=1), which is False when our agent
        # dies but 2+ enemies remain — causing hundreds of useless "dead" steps that
        # flood the rollout buffer with STOP-only transitions and dilute training.
        our_alive = int(curr_p[self._aid][2]) == 1
        terminated = engine_terminated or (not our_alive)

        # Multi-step stagnation penalty: catches camping / oscillation (A↔B).
        # Fires when agent occupied the same cell 6+ times in last 8 steps.
        # Complements reward standing_still (which only sees 1-step position change).
        my_pos = (int(curr_p[self._aid][0]), int(curr_p[self._aid][1]))
        self._pos_history.append(my_pos)
        if len(self._pos_history) > 8:
            self._pos_history.pop(0)
        if len(self._pos_history) == 8 and int(curr_p[self._aid][2]) == 1:
            freq: dict[tuple, int] = {}
            for pos in self._pos_history:
                freq[pos] = freq.get(pos, 0) + 1
            if max(freq.values()) >= 6:
                reward -= 0.03  # ~4× standing_still rate; discourages camping/oscillation

        self._prev_obs = self._raw_obs
        self._raw_obs = next_raw

        obs_dict = self._build_obs()
        info: dict = {
            "kills": self.episode_kills,
            "boxes_destroyed": self.episode_boxes,
            "items_collected": self.episode_items,
            "step": self._step,
            "reward_info": reward_info,
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
    def current_step(self) -> int:
        """Current episode step count (alias for _step)."""
        return self._step

    @property
    def raw_obs(self) -> dict | None:
        """Access underlying engine obs (useful for rule-based fallback)."""
        return self._raw_obs

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass
