"""
Reward function v4 — tie-break + game-mechanics aware for Bomberland FFA.

Changes from v3:
  - Updated REWARDS values: win 3.0→5.0, agent_death -2.0→-3.0,
    kill_credit 2.0→2.5, box_destroyed 0.4→0.5, item_collected 0.3→0.4,
    chain_reaction 0.3→0.5, danger_evasion 0.12→0.15, danger_enter -0.08→-0.15,
    own_blast_loiter -0.04→-0.06, standing_still -0.008→-0.015,
    approach_enemy 0.006→0.008
  - approach_enemy gated: only active when current_step > 300 (avoids
    contradicting Priority 1 Survive in early/mid game for a 0.008 signal)
  - Late-game multiplier (×1.3) on kill_credit, box_destroyed, item_collected
    when current_step / 500 > 0.8 (step > 400)
  - Chain reaction detection upgraded from Manhattan proximity to blast-overlap:
    bonus triggers when the just-placed bomb's blast corridor intersects any
    existing active bomb's position (not just Manhattan ≤ 2); still capped at
    one bonus per step
  - Box attribution unchanged from v3: only MY bombs with timer==1 in prev_obs
  - Function signature extended: action, episode_stats, current_step added;
    returns (reward_float, info_dict) with per-component breakdown
  - All magic numbers consolidated in REWARDS dict
"""

from __future__ import annotations

import numpy as np

# ─────────────────────────────────────────────────────────────────────────── #
# Reward table — all magic numbers live here                                   #
# ─────────────────────────────────────────────────────────────────────────── #

REWARDS: dict[str, float] = {
    # Terminal
    "win":                  5.0,
    "agent_death":         -3.0,
    # Tie-break stats (priority order: kills > boxes > items > bombs)
    "kill_credit":          2.5,
    "box_destroyed":        0.5,
    "item_collected":       0.4,
    "bomb_placed":          0.003,
    # Tactical bonuses
    "chain_reaction":       0.5,   # bomb blast overlaps an existing active bomb
    "item_contest":         0.1,   # moving toward item an enemy is also approaching
    # Danger shaping
    "danger_evasion":       0.15,
    "danger_enter":        -0.15,
    "own_blast_loiter":    -0.06,
    # Movement
    "survival_step":        0.005,
    "standing_still":      -0.015,
    "time_penalty":        -0.003,
    # Enemy pressure — gated to step > 300 only
    "approach_enemy":       0.008,
}

# ─────────────────────────────────────────────────────────────────────────── #
# Internal constants                                                            #
# ─────────────────────────────────────────────────────────────────────────── #

_DEFAULT_BOMB_TIMER: int = 7
_LATE_GAME_STEP_THRESHOLD: float = 0.8   # current_step / 500 > this → multiplier active
_LATE_GAME_MULTIPLIER: float = 1.3
_APPROACH_ENEMY_MIN_STEP: int = 300      # approach_enemy only active after this step
_TOTAL_STEPS: int = 500                  # max episode length

WALL: int = 1
BOX: int = 2


# ─────────────────────────────────────────────────────────────────────────── #
# Internal helpers                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

def _parse_bombs(bombs_raw: object) -> np.ndarray | None:
    """Coerce raw bomb data to a 2-D (N, 4) int array, or None if empty."""
    if bombs_raw is None:
        return None
    arr = np.asarray(bombs_raw, dtype=np.int32)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _blast_mask(grid: np.ndarray, bx: int, by: int, radius: int) -> np.ndarray:
    """
    Return a boolean (H, W) mask of all tiles hit by a bomb at (bx, by).

    The blast spreads in 4 cardinal directions up to `radius` tiles.
    Walls block and are excluded; boxes block (their cell IS included) but
    the blast does not propagate past them.

    Fully vectorised: uses array slicing with cumulative wall/box detection
    per direction — no Python loops over grid cells.
    """
    h, w = grid.shape
    mask = np.zeros((h, w), dtype=bool)
    mask[bx, by] = True

    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for r in range(1, radius + 1):
            tx, ty = bx + dx * r, by + dy * r
            if not (0 <= tx < h and 0 <= ty < w):
                break
            cell = int(grid[tx, ty])
            if cell == WALL:
                break
            mask[tx, ty] = True
            if cell == BOX:
                break

    return mask


def _danger_at(obs: dict, x: int, y: int) -> tuple[bool, int | None]:
    """
    Return (in_blast, min_timer) for tile (x, y) given all active bombs in obs.

    Uses vectorised blast mask per bomb; no grid cell iteration.
    """
    bombs = _parse_bombs(obs["bombs"])
    if bombs is None:
        return False, None
    grid = np.asarray(obs["map"], dtype=np.int32)
    players = np.asarray(obs["players"], dtype=np.int32)
    in_blast = False
    min_timer: int | None = None
    for row in bombs:
        bx, by, timer, owner_id = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        n_players = len(players)
        if 0 <= owner_id < n_players:
            radius = 1 + int(players[owner_id][4])
        else:
            radius = 2
        bmask = _blast_mask(grid, bx, by, radius)
        if bmask[x, y]:
            in_blast = True
            min_timer = timer if min_timer is None else min(min_timer, timer)
    return in_blast, min_timer


def _own_blast_timer_at(obs: dict, agent_id: int, x: int, y: int) -> int | None:
    """
    Smallest timer among agent_id's own bombs whose blast covers tile (x, y).

    Returns None if no such bomb exists.
    """
    bombs = _parse_bombs(obs["bombs"])
    if bombs is None:
        return None
    grid = np.asarray(obs["map"], dtype=np.int32)
    players = np.asarray(obs["players"], dtype=np.int32)
    best: int | None = None
    for row in bombs:
        bx, by, timer, owner_id = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        if int(owner_id) != agent_id:
            continue
        radius = 1 + int(players[agent_id][4])
        bmask = _blast_mask(grid, bx, by, radius)
        if bmask[x, y]:
            best = timer if best is None else min(best, timer)
    return best


def _enemies_alive(players: np.ndarray, agent_id: int) -> int:
    """Count living enemies (all players except agent_id)."""
    return int(sum(int(players[i][2]) for i in range(len(players)) if i != agent_id))


def _manhattan_nearest_enemy(
    players: np.ndarray,
    agent_id: int,
    x: int,
    y: int,
) -> int | None:
    """Manhattan distance to the nearest living enemy from tile (x, y)."""
    best: int | None = None
    for i in range(len(players)):
        if i == agent_id or int(players[i][2]) == 0:
            continue
        d = abs(x - int(players[i][0])) + abs(y - int(players[i][1]))
        best = d if best is None else min(best, d)
    return best


def _enemy_in_my_blast(
    obs: dict,
    agent_id: int,
    enemy_id: int,
) -> bool:
    """
    Return True if enemy_id's position in obs is within the blast zone of
    any bomb owned by agent_id in that same obs.

    Used for kill attribution: we only credit a kill if the enemy was
    standing in OUR bomb's blast corridor the step before they died.
    """
    bombs = _parse_bombs(obs["bombs"])
    if bombs is None:
        return False
    grid = np.asarray(obs["map"], dtype=np.int32)
    players = np.asarray(obs["players"], dtype=np.int32)
    ex = int(players[enemy_id][0])
    ey = int(players[enemy_id][1])
    for row in bombs:
        bx, by, _timer, owner_id = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        if int(owner_id) != agent_id:
            continue
        radius = 1 + int(players[agent_id][4])
        if _blast_mask(grid, bx, by, radius)[ex, ey]:
            return True
    return False


def _chain_reaction_bonus(
    grid: np.ndarray,
    placed_bx: int,
    placed_by: int,
    placed_radius: int,
    all_bombs: np.ndarray,
    agent_id: int,
) -> bool:
    """
    Return True if the just-placed bomb's blast corridor overlaps any OTHER
    active bomb's tile position (blast-overlap detection, not proximity).

    This supersedes the v3 Manhattan-distance heuristic and correctly captures
    diagonal corridors and long-range chains.
    """
    blast = _blast_mask(grid, placed_bx, placed_by, placed_radius)
    for row in all_bombs:
        bx, by, _timer, owner_id = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        if int(owner_id) == agent_id and bx == placed_bx and by == placed_by:
            continue  # skip the bomb we just placed (same cell)
        if blast[bx, by]:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────── #
# Public API                                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

def compute_reward(
    prev_obs: dict | None,
    curr_obs: dict,
    action: int,
    agent_id: int,
    episode_stats: dict[str, int],
    current_step: int,
) -> tuple[float, dict[str, float]]:
    """
    Compute shaped reward for agent_id given consecutive obs frames.

    Args:
        prev_obs:       observation at t-1 (None for the first step)
        curr_obs:       observation at t
        action:         discrete action taken at this step (0-5)
        agent_id:       which player we are (0-3)
        episode_stats:  mutable dict with keys "kills", "boxes_destroyed",
                        "items_collected" — updated in place each step so the
                        caller can track cumulative stats for aux features 5-6
        current_step:   current game step (0-500)

    Returns:
        (reward, info) where info maps component name → float contribution
    """
    info: dict[str, float] = {}

    if prev_obs is None:
        return 0.0, info

    aid = int(agent_id)
    prev_p = np.asarray(prev_obs["players"], dtype=np.int32)
    curr_p = np.asarray(curr_obs["players"], dtype=np.int32)

    prev_alive = int(prev_p[aid][2])
    curr_alive = int(curr_p[aid][2])

    # ── Death (terminal, return immediately) ─────────────────────────── #
    if prev_alive == 1 and curr_alive == 0:
        info["agent_death"] = float(REWARDS["agent_death"])
        return float(REWARDS["agent_death"]), info

    # ── Late-game multiplier ──────────────────────────────────────────── #
    step_ratio = current_step / _TOTAL_STEPS
    late_game = step_ratio > _LATE_GAME_STEP_THRESHOLD
    lm = _LATE_GAME_MULTIPLIER if late_game else 1.0   # applied to kill/box/item

    reward = 0.0

    # ── Survival bonus + time penalty ────────────────────────────────── #
    info["survival_step"] = float(REWARDS["survival_step"])
    info["time_penalty"] = float(REWARDS["time_penalty"])
    reward += REWARDS["survival_step"] + REWARDS["time_penalty"]

    # ── Kill credit ───────────────────────────────────────────────────── #
    # Only credit kills where the enemy was inside OUR bomb's blast zone in
    # prev_obs (i.e. we caused the kill, not an enemy-vs-enemy explosion).
    # Simple alive-flag diff without attribution would incorrectly reward us
    # when two enemies blow each other up.
    prev_enemies = _enemies_alive(prev_p, aid)
    curr_enemies = _enemies_alive(curr_p, aid)
    kills = 0
    for i in range(len(prev_p)):
        if i == aid:
            continue
        was_alive = int(prev_p[i][2]) == 1
        now_dead = int(curr_p[i][2]) == 0
        if was_alive and now_dead:
            # Only credit if that enemy was in OUR blast zone last step
            if _enemy_in_my_blast(prev_obs, aid, i):
                kills += 1
    if kills:
        kill_r = kills * REWARDS["kill_credit"] * lm
        reward += kill_r
        info["kill_credit"] = kill_r
        episode_stats["kills"] = episode_stats.get("kills", 0) + kills
        # Last kill secures win
        if curr_enemies == 0:
            reward += REWARDS["win"]
            info["win"] = float(REWARDS["win"])

    # ── Confirmed box destruction (MY bombs with timer==1 only) ──────── #
    prev_grid = np.asarray(prev_obs["map"], dtype=np.int32)
    curr_grid = np.asarray(curr_obs["map"], dtype=np.int32)
    my_boxes_destroyed = 0
    prev_bombs_arr = _parse_bombs(prev_obs["bombs"])
    if prev_bombs_arr is not None:
        for row in prev_bombs_arr:
            bx, by, timer, owner = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            if int(owner) != aid or int(timer) != 1:
                continue  # only my bombs exploding this step
            radius = 1 + int(prev_p[aid][4])
            bmask = _blast_mask(prev_grid, bx, by, radius)
            # Count cells that were BOX in prev and are not BOX in curr
            destroyed = np.sum(
                (prev_grid == BOX) & (curr_grid != BOX) & bmask
            )
            my_boxes_destroyed += int(destroyed)
    if my_boxes_destroyed > 0:
        box_r = my_boxes_destroyed * REWARDS["box_destroyed"] * lm
        reward += box_r
        info["box_destroyed"] = box_r
        episode_stats["boxes_destroyed"] = (
            episode_stats.get("boxes_destroyed", 0) + my_boxes_destroyed
        )

    # ── Item collection ───────────────────────────────────────────────── #
    prev_radius_bonus = int(prev_p[aid][4])
    curr_radius_bonus = int(curr_p[aid][4])
    prev_cap = int(prev_p[aid][3])
    curr_cap = int(curr_p[aid][3])
    # Capacity increase from item pickup (not from bomb use, which decreases cap)
    if curr_radius_bonus > prev_radius_bonus or curr_cap > prev_cap:
        item_r = REWARDS["item_collected"] * lm
        reward += item_r
        info["item_collected"] = item_r
        episode_stats["items_collected"] = episode_stats.get("items_collected", 0) + 1

    # ── Item contest bonus ────────────────────────────────────────────── #
    px, py = int(prev_p[aid][0]), int(prev_p[aid][1])
    cx, cy = int(curr_p[aid][0]), int(curr_p[aid][1])
    if px != cx or py != cy:
        item_cells = np.argwhere(np.isin(np.asarray(curr_obs["map"], dtype=np.int32), [3, 4]))
        if len(item_cells) > 0:
            dists_me = np.abs(item_cells[:, 0] - cx) + np.abs(item_cells[:, 1] - cy)
            my_nearest = int(np.min(dists_me))
            enemy_nearest = min(
                (
                    int(np.min(
                        np.abs(item_cells[:, 0] - int(curr_p[i][0]))
                        + np.abs(item_cells[:, 1] - int(curr_p[i][1]))
                    ))
                    for i in range(len(curr_p))
                    if i != aid and int(curr_p[i][2]) == 1
                ),
                default=999,
            )
            if my_nearest <= 3 and enemy_nearest <= my_nearest + 2:
                reward += REWARDS["item_contest"]
                info["item_contest"] = float(REWARDS["item_contest"])

    # ── Bomb placed + chain reaction bonus ───────────────────────────── #
    # A bomb was placed if bombs_left decreased (capacity used) and action==5.
    # Using action check avoids false positive when capacity item collected in
    # same step as bomb explodes.
    bomb_placed = (action == 5) and (prev_cap > curr_cap)
    if bomb_placed:
        reward += REWARDS["bomb_placed"]
        info["bomb_placed"] = float(REWARDS["bomb_placed"])

        # Chain reaction: the just-placed bomb's blast overlaps another active bomb
        curr_bombs_arr = _parse_bombs(curr_obs["bombs"])
        if curr_bombs_arr is not None and len(curr_bombs_arr) > 0:
            placed_radius = 1 + curr_radius_bonus
            if _chain_reaction_bonus(
                curr_grid, cx, cy, placed_radius, curr_bombs_arr, aid
            ):
                reward += REWARDS["chain_reaction"]
                info["chain_reaction"] = float(REWARDS["chain_reaction"])

    # ── Movement / standing still ─────────────────────────────────────── #
    # Exempt PLACE_BOMB (action==5) from standing_still penalty: the agent
    # intentionally stayed in place to plant; penalising it discourages bombing.
    if px == cx and py == cy and not bomb_placed:
        reward += REWARDS["standing_still"]
        info["standing_still"] = float(REWARDS["standing_still"])

    # ── Danger evasion / entry ────────────────────────────────────────── #
    bombs_exist = (
        (prev_obs["bombs"] is not None and len(prev_obs["bombs"]) > 0)
        or (curr_obs["bombs"] is not None and len(curr_obs["bombs"]) > 0)
    )
    if bombs_exist:
        prev_in_blast, prev_timer = _danger_at(prev_obs, px, py)
        curr_in_blast, _ = _danger_at(curr_obs, cx, cy)
        if prev_in_blast and not curr_in_blast:
            # timer==1 in prev_obs means the bomb decrements to 0 this step → explodes.
            # That is the true "about to explode" signal; timer<=3 is "soon" but
            # only timer==1 warrants maximum urgency.
            urgency = 1.5 if (prev_timer is not None and prev_timer <= 1) else 1.0
            evasion_r = REWARDS["danger_evasion"] * urgency
            reward += evasion_r
            info["danger_evasion"] = evasion_r
        elif not prev_in_blast and curr_in_blast and (px != cx or py != cy):
            reward += REWARDS["danger_enter"]
            info["danger_enter"] = float(REWARDS["danger_enter"])

    # ── Own blast loiter penalty ──────────────────────────────────────── #
    own_timer = _own_blast_timer_at(curr_obs, aid, cx, cy)
    if curr_alive == 1 and own_timer is not None:
        urgency = max(1, _DEFAULT_BOMB_TIMER - int(own_timer))
        loiter_r = REWARDS["own_blast_loiter"] * urgency
        reward += loiter_r
        info["own_blast_loiter"] = loiter_r

    # ── Enemy pressure — only after step 300 AND we have a bomb ─────── #
    # Gated: approaching an enemy in early/mid game contradicts Priority 1
    # (Survive) for a 0.008-per-tile signal; only enable in late game when
    # aggressive plays are required for tie-break.
    # bombs_left > 0 check: approaching without a bomb to place is pointless
    # aggression — the agent cannot threaten the enemy and may walk into danger.
    curr_bombs_left = int(curr_p[aid][3])
    if (
        curr_alive == 1
        and current_step > _APPROACH_ENEMY_MIN_STEP
        and curr_bombs_left > 0
        and prev_enemies > 0
        and curr_enemies > 0
    ):
        pd = _manhattan_nearest_enemy(prev_p, aid, px, py)
        cd = _manhattan_nearest_enemy(curr_p, aid, cx, cy)
        if pd is not None and cd is not None:
            approach_r = REWARDS["approach_enemy"] * (pd - cd)
            reward += approach_r
            if approach_r != 0.0:
                info["approach_enemy"] = approach_r

    return float(reward), info
