"""
Reward function v2 — tie-break aware for Bomberland FFA.

Changes from v1 (agent/dqn_agent/reward.py):
  - kill_credit +1.5 (tie-break #1, was 1.0)
  - box_destroyed +0.4 per confirmed box (tie-break #2, previously absent)
  - plant_near_box REMOVED (misleading proxy)
  - bomb_placed +0.003 (tie-break #4 padding, new)
  - survival_step +0.005/step (new — explicit survival signal)
  - approach_enemy reduced 0.02 → 0.006 (dangerous in FFA)
  - own_blast_loiter scaled by urgency (unchanged from v1)
"""

from __future__ import annotations

import numpy as np

# ─────────────────────────────────────────────────────────────────────────── #
# Reward table                                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

REWARDS: dict[str, float] = {
    # Terminal
    "win":               3.0,
    "agent_death":      -2.0,
    # Tie-break stats
    "kill_credit":       1.5,
    "box_destroyed":     0.4,
    "item_collected":    0.15,
    "bomb_placed":       0.003,
    # Danger shaping
    "danger_evasion":    0.12,
    "danger_enter":     -0.08,
    "own_blast_loiter": -0.04,
    # Movement
    "survival_step":     0.005,
    "standing_still":   -0.008,
    "time_penalty":     -0.003,
    # Enemy pressure (FFA-safe)
    "approach_enemy":    0.006,
}

# Internal constants
_DEFAULT_BOMB_TIMER = 7
WALL = 1
BOX = 2


# ─────────────────────────────────────────────────────────────────────────── #
# Internal helpers                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

def _parse_bombs(bombs_raw) -> np.ndarray | None:
    if bombs_raw is None:
        return None
    arr = np.asarray(bombs_raw)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _blast_tiles(grid: np.ndarray, bx: int, by: int, radius: int) -> set[tuple[int, int]]:
    """Cross-shaped blast, blocked by walls, passes then stops at boxes."""
    h, w = grid.shape
    tiles: set[tuple[int, int]] = {(bx, by)}
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for r in range(1, radius + 1):
            tx, ty = bx + dx * r, by + dy * r
            if not (0 <= tx < h and 0 <= ty < w):
                break
            cell = int(grid[tx, ty])
            if cell == WALL:
                break
            tiles.add((tx, ty))
            if cell == BOX:
                break
    return tiles


def _danger_at(
    obs: dict,
    x: int,
    y: int,
) -> tuple[bool, int | None]:
    """Return (in_blast, min_timer) for tile (x, y)."""
    bombs = _parse_bombs(obs["bombs"])
    if bombs is None:
        return False, None
    grid = obs["map"]
    players = obs["players"]
    in_blast = False
    min_timer: int | None = None
    for row in bombs:
        bx, by, timer, owner_id = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        owner_id = int(owner_id)
        if 0 <= owner_id < len(players):
            radius = 1 + int(players[owner_id][4])
        else:
            radius = 2
        if (x, y) in _blast_tiles(grid, bx, by, radius):
            in_blast = True
            min_timer = timer if min_timer is None else min(min_timer, timer)
    return in_blast, min_timer


def _own_blast_timer_at(obs: dict, agent_id: int, x: int, y: int) -> int | None:
    """Smallest timer among agent's own bombs whose blast covers (x, y)."""
    bombs = _parse_bombs(obs["bombs"])
    if bombs is None:
        return None
    grid = obs["map"]
    players = obs["players"]
    best: int | None = None
    for row in bombs:
        bx, by, timer, owner_id = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        if int(owner_id) != agent_id:
            continue
        radius = 1 + int(players[agent_id][4])
        if (x, y) in _blast_tiles(grid, bx, by, radius):
            best = timer if best is None else min(best, timer)
    return best


def _enemies_alive(players: np.ndarray, agent_id: int) -> int:
    return sum(int(players[i][2]) for i in range(len(players)) if i != agent_id)


def _manhattan_nearest_enemy(players: np.ndarray, agent_id: int, x: int, y: int) -> int | None:
    best: int | None = None
    for i in range(len(players)):
        if i == agent_id or int(players[i][2]) == 0:
            continue
        d = abs(x - int(players[i][0])) + abs(y - int(players[i][1]))
        best = d if best is None else min(best, d)
    return best


# ─────────────────────────────────────────────────────────────────────────── #
# Public API                                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

def compute_reward(
    prev_obs: dict | None,
    curr_obs: dict,
    agent_id: int,
) -> float:
    """
    Compute shaped reward for agent_id given consecutive obs frames.

    Args:
        prev_obs:  observation at t-1 (None for the first step)
        curr_obs:  observation at t
        agent_id:  which player we are

    Returns:
        reward (float)
    """
    if prev_obs is None:
        return 0.0

    aid = int(agent_id)
    prev_p = np.asarray(prev_obs["players"])
    curr_p = np.asarray(curr_obs["players"])

    prev_alive = int(prev_p[aid][2])
    curr_alive = int(curr_p[aid][2])

    # ── Death (terminal, return immediately) ─────────────────────────── #
    if prev_alive == 1 and curr_alive == 0:
        return float(REWARDS["agent_death"])

    reward = 0.0

    # ── Survival bonus ────────────────────────────────────────────────── #
    reward += REWARDS["survival_step"]
    reward += REWARDS["time_penalty"]

    # ── Kill credit ───────────────────────────────────────────────────── #
    prev_enemies = _enemies_alive(prev_p, aid)
    curr_enemies = _enemies_alive(curr_p, aid)
    kills = max(0, prev_enemies - curr_enemies)
    if kills:
        reward += kills * REWARDS["kill_credit"]
        if curr_enemies == 0:
            reward += REWARDS["win"]

    # ── Confirmed box destruction ─────────────────────────────────────── #
    prev_grid = np.asarray(prev_obs["map"])
    curr_grid = np.asarray(curr_obs["map"])
    boxes_destroyed = int(((prev_grid == BOX) & (curr_grid != BOX)).sum())
    if boxes_destroyed:
        reward += boxes_destroyed * REWARDS["box_destroyed"]

    # ── Item collection ───────────────────────────────────────────────── #
    prev_radius = int(prev_p[aid][4])
    curr_radius = int(curr_p[aid][4])
    prev_cap = int(prev_p[aid][3])
    curr_cap = int(curr_p[aid][3])
    if curr_radius > prev_radius or curr_cap > prev_cap:
        reward += REWARDS["item_collected"]

    # ── Bomb placed ───────────────────────────────────────────────────── #
    if prev_cap > curr_cap:  # used a bomb slot
        reward += REWARDS["bomb_placed"]

    # ── Movement / standing still ─────────────────────────────────────── #
    px, py = int(prev_p[aid][0]), int(prev_p[aid][1])
    cx, cy = int(curr_p[aid][0]), int(curr_p[aid][1])
    if px == cx and py == cy:
        reward += REWARDS["standing_still"]

    # ── Danger evasion / entry ────────────────────────────────────────── #
    bombs_exist = (
        (prev_obs["bombs"] is not None and len(prev_obs["bombs"]) > 0)
        or (curr_obs["bombs"] is not None and len(curr_obs["bombs"]) > 0)
    )
    if bombs_exist:
        prev_in_blast, prev_timer = _danger_at(prev_obs, px, py)
        curr_in_blast, _ = _danger_at(curr_obs, cx, cy)
        if prev_in_blast and not curr_in_blast:
            urgency = 1.5 if (prev_timer is not None and prev_timer <= 3) else 1.0
            reward += REWARDS["danger_evasion"] * urgency
        elif not prev_in_blast and curr_in_blast and (px != cx or py != cy):
            reward += REWARDS["danger_enter"]

    # Standing in own blast: penalize more as fuse burns down
    own_timer = _own_blast_timer_at(curr_obs, aid, cx, cy)
    if curr_alive == 1 and own_timer is not None:
        urgency = max(1, _DEFAULT_BOMB_TIMER - int(own_timer))
        reward += REWARDS["own_blast_loiter"] * urgency

    # ── Enemy pressure (damped for FFA) ──────────────────────────────── #
    if curr_alive == 1 and prev_enemies > 0 and curr_enemies > 0:
        pd = _manhattan_nearest_enemy(prev_p, aid, px, py)
        cd = _manhattan_nearest_enemy(curr_p, aid, cx, cy)
        if pd is not None and cd is not None:
            reward += REWARDS["approach_enemy"] * (pd - cd)

    return float(reward)
