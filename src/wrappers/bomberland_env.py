from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from engine import BomberEnv
from src.utils.feature_extractor import (
    extract_features, count_boxes, obs_to_dict,
)
from src.logic.action_masking import compute_action_mask
from src.training.reward import compute_reward
from src.models.policy_network import make_observation_space


class SingleAgentBomberEnv(gym.Env):
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
        self._opponents = opponents

        self.observation_space: spaces.Dict = make_observation_space()
        self.action_space: spaces.Discrete = spaces.Discrete(6)

        self._engine = BomberEnv(max_steps=max_steps, seed=seed)

        self._raw_obs: dict | None = None
        self._prev_obs: dict | None = None
        self._step: int = 0
        self._initial_boxes: int = 50
        self._my_kills: int = 0
        self._my_boxes: int = 0
        self._episode_seed: int = seed or 0
        self.episode_kills: int = 0
        self.episode_boxes: int = 0
        self.episode_items: int = 0
        self._recent_bomb_blasts: list = []
        self._pos_history: list[tuple[int, int]] = []

        all_ids = list(range(4))
        all_ids.remove(self._aid)
        self._opp_ids: list[int] = all_ids

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        if seed is not None:
            self._episode_seed = seed
        else:
            self._episode_seed += 1

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
        self._recent_bomb_blasts = []
        self._pos_history = []

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

        prev_p = np.asarray(self._raw_obs["players"])
        curr_p = np.asarray(next_raw["players"])
        prev_enemies = sum(int(prev_p[i][2]) for i in range(4) if i != self._aid)
        curr_enemies = sum(int(curr_p[i][2]) for i in range(4) if i != self._aid)
        kills_this_step = max(0, prev_enemies - curr_enemies)
        self._my_kills += kills_this_step
        self.episode_kills += kills_this_step

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
                        continue
                    radius = 1 + int(prev_p_arr[self._aid][4])
                    blast = np.zeros((13, 13), dtype=np.bool_)
                    blast[bx, by] = True
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        for r in range(1, radius + 1):
                            nx, ny = bx + dx * r, by + dy * r
                            if nx < 0 or nx >= 13 or ny < 0 or ny >= 13:
                                break
                            if prev_grid[nx, ny] == 1:
                                break
                            blast[nx, ny] = True
                            if prev_grid[nx, ny] == 2:
                                break
                    destroyed = int(np.sum((prev_grid == 2) & (curr_grid != 2) & blast))
                    my_boxes_this_step += destroyed
        self._my_boxes += my_boxes_this_step
        self.episode_boxes += my_boxes_this_step

        aid = self._aid
        prev_radius_bonus = int(prev_p[aid][4])
        curr_radius_bonus = int(curr_p[aid][4])
        prev_cap = int(prev_p[aid][3])
        curr_cap = int(curr_p[aid][3])
        if int(curr_p[aid][2]) == 1:
            cx_item = int(curr_p[aid][0])
            cy_item = int(curr_p[aid][1])
            radius_item = curr_radius_bonus > prev_radius_bonus
            cap_item = curr_cap > prev_cap and prev_grid[cx_item, cy_item] == 4
            if radius_item or cap_item:
                self.episode_items += 1

        self._step += 1

        episode_stats = {
            "kills": self.episode_kills,
            "boxes_destroyed": self.episode_boxes,
            "items_collected": self.episode_items,
            "recent_bomb_blasts": self._recent_bomb_blasts,
        }
        reward, reward_info = compute_reward(
            self._raw_obs, next_raw, int(action), self._aid,
            episode_stats, self._step,
        )
        self._recent_bomb_blasts = episode_stats.get("recent_bomb_blasts", [])

        our_alive = int(curr_p[self._aid][2]) == 1
        terminated = engine_terminated or (not our_alive)

        my_pos = (int(curr_p[self._aid][0]), int(curr_p[self._aid][1]))
        self._pos_history.append(my_pos)
        if len(self._pos_history) > 8:
            self._pos_history.pop(0)
        if len(self._pos_history) == 8 and int(curr_p[self._aid][2]) == 1:
            freq: dict[tuple, int] = {}
            for pos in self._pos_history:
                freq[pos] = freq.get(pos, 0) + 1
            if max(freq.values()) >= 6:
                reward -= 0.03

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
        if self._raw_obs is None:
            return np.ones(6, dtype=np.bool_)
        return compute_action_mask(self._raw_obs, self._aid)

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

    @property
    def current_step(self) -> int:
        return self._step

    @property
    def raw_obs(self) -> dict | None:
        return self._raw_obs

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass
