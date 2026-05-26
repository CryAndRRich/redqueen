"""
Reward function v3 — tie-break + game-mechanics aware for Bomberland FFA.

Changes from v2:
  - item_collected 0.15 → 0.3  (tie-break #3 was underweighted)
  - kill_credit 1.5 → 2.0 with late-game bonus ×1.5 when enemies_alive drops to 1
    (last kill = game-winner; aligns with win: 3.0 terminal reward)
  - chain_reaction_bonus +0.3: placed bomb near existing bomb(s) with short timer
    (tie-break synergy: chain reactions destroy more boxes and can multi-kill)
  - item_contest_urgency: +0.1 when moving toward an item that an enemy is also
    approaching (simultaneous collection destroys item per competition rules)
  - box_destroyed attribution improved: only count confirmed-mine boxes when
    a MY bomb has timer <= 1 in prev_obs; fall back to all-boxes otherwise
  - approach_enemy kept at 0.006 (FFA-safe, do not raise)
"""

from __future__ import annotations

import numpy as np

# ─────────────────────────────────────────────────────────────────────────── #
# Reward table                                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

REWARDS: dict[str, float] = {
    # Terminal
    "win":                  3.0,
    "agent_death":         -2.0,
    # Tie-break stats (priority order: kills > boxes > items > bombs)
    "kill_credit":          2.0,   # v3: raised from 1.5; last-kill bonus applied separately
    "box_destroyed":        0.4,
    "item_collected":       0.3,   # v3: raised from 0.15 (tie-break #3 was underweighted)
    "bomb_placed":          0.003,
    # Tactical bonuses
    "chain_reaction":       0.3,   # v3: bonus for placing bomb adjacent to another active bomb
    "item_contest":         0.1,   # v3: bonus for moving toward item an enemy is also approaching
    # Danger shaping
    "danger_evasion":       0.12,
    "danger_enter":        -0.08,
    "own_blast_loiter":    -0.04,
    # Movement
    "survival_step":        0.005,
    "standing_still":      -0.008,
    "time_penalty":        -0.003,
    # Enemy pressure (FFA-safe — do not raise, dangerous in multi-agent)
    "approach_enemy":       0.006,
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
        # Late-game bonus: last kill secures win — extra incentive when only 1 enemy left
        if curr_enemies == 0:
            reward += REWARDS["win"]
        elif prev_enemies == 1:
            # Down to last enemy: killing this one is especially valuable
            reward += REWARDS["kill_credit"] * 0.5

    # ── Confirmed box destruction (attributed to MY bombs only) ──────────── #
    # Previous code counted ALL boxes destroyed — gave credit for enemy bomb explosions.
    # Now: only credit boxes in blast zone of MY bombs that had timer==1 in prev_obs,
    # because timer==1 → timer becomes 0 → explosion resolves this step.
    prev_grid = np.asarray(prev_obs["map"])
    curr_grid = np.asarray(curr_obs["map"])
    my_boxes_destroyed = 0
    prev_bombs_arr = _parse_bombs(prev_obs["bombs"])
    if prev_bombs_arr is not None:
        for row in prev_bombs_arr:
            bx, by, timer, owner = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            if int(owner) != aid or int(timer) != 1:
                continue  # only my bombs that explode this step
            radius = 1 + int(prev_p[aid][4])
            blast = _blast_tiles(prev_grid, bx, by, radius)
            for tx, ty in blast:
                if prev_grid[tx, ty] == BOX and curr_grid[tx, ty] != BOX:
                    my_boxes_destroyed += 1
    if my_boxes_destroyed > 0:
        reward += my_boxes_destroyed * REWARDS["box_destroyed"]

    # ── Item collection ───────────────────────────────────────────────── #
    prev_radius = int(prev_p[aid][4])
    curr_radius = int(curr_p[aid][4])
    prev_cap = int(prev_p[aid][3])
    curr_cap = int(curr_p[aid][3])
    if curr_radius > prev_radius or curr_cap > prev_cap:
        reward += REWARDS["item_collected"]

    # ── Item contest bonus ────────────────────────────────────────────── #
    # Reward moving toward an item that an enemy is also approaching.
    # Per competition rules: simultaneous collection destroys the item.
    px, py = int(prev_p[aid][0]), int(prev_p[aid][1])
    cx, cy = int(curr_p[aid][0]), int(curr_p[aid][1])
    if px != cx or py != cy:
        item_tiles = set(
            zip(*np.where(np.isin(np.asarray(curr_obs["map"]), [3, 4])))
        )
        if item_tiles:
            my_dist_to_nearest = min(abs(cx - ix) + abs(cy - iy) for ix, iy in item_tiles)
            enemy_min_dist = min(
                (min(abs(int(curr_p[i][0]) - ix) + abs(int(curr_p[i][1]) - iy) for ix, iy in item_tiles)
                 for i in range(len(curr_p)) if i != aid and int(curr_p[i][2]) == 1),
                default=999,
            )
            if my_dist_to_nearest <= 3 and enemy_min_dist <= my_dist_to_nearest + 2:
                reward += REWARDS["item_contest"]

    # ── Bomb placed + chain reaction bonus ───────────────────────────── #
    if prev_cap > curr_cap:  # used a bomb slot
        reward += REWARDS["bomb_placed"]
        # Chain reaction bonus: placing bomb adjacent to another active bomb
        # that has a short timer encourages strategic chaining
        curr_bombs = _parse_bombs(curr_obs["bombs"])
        if curr_bombs is not None:
            for row in curr_bombs:
                bx, by, timer, owner = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                if int(owner) == aid:
                    continue  # skip own just-placed bomb
                if timer <= 4 and abs(bx - cx) + abs(by - cy) <= 2:
                    reward += REWARDS["chain_reaction"]
                    break  # one bonus per step

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
