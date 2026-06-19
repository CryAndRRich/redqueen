"""
Reward function v5 — adaptive shaping + kill-assist + zone control.

Changes from v4:
  - IMPROVEMENT 1: Smooth late-game ramp (step 350→500) replaces the hard
    ×1.3 step at step>400.  Formula: 1.0 + 0.3 * max(0, (step - 350) / 150).
    Gives a gradual incentive to get aggressive as the game ends.
  - IMPROVEMENT 2: Kill-assist reward (0.3 × kill_credit).  If an enemy dies
    this step AND we placed a bomb in the last 7 steps whose blast zone covered
    the enemy's last known position, we receive a partial credit even when the
    enemy was not inside an active bomb at the previous tick.  Tracked via a new
    episode_stats["recent_bomb_blasts"] key (list of blast masks, capped at 3).
  - IMPROVEMENT 3: Zone control reward (+0.005/step) when our agent is within
    4 Manhattan tiles of the map centre (6, 6) AND ≥2 enemies are still alive.
    Capped to one bonus per step; irrelevant in 1v1 end-game.
  - IMPROVEMENT 4: Continuous own_blast_loiter penalty scaled by
    (7 - timer) / 7 * 0.25, giving a smooth range [0.04, 0.21] instead of a
    stepped multiplier.  Maximum at timer=1 (bomb about to explode),
    minimum at timer=6.  At timer≤2 the penalty exceeds danger_enter (-0.15),
    forcing the agent to flee own bombs even into enemy blast zones.
  - IMPROVEMENT 5: Wasted-bomb penalty (−0.05) when one of our bombs detonates
    (timer==1 or chain-triggered) and it destroyed 0 boxes AND 0 enemies died
    in this step.  Detected per-exploding-bomb; at most one penalty per step
    regardless of how many bombs expire.
  - All other v4 logic preserved verbatim (chain-reaction detection, BC
    attribution, kill attribution, item contest, etc.)
  - FIX (Stage 1 convergence): approach_enemy gate removed (_APPROACH_ENEMY_MIN_STEP=0).
    Previously gated at step>300, but agent typically died at step 80-120 so kill
    incentive was never active.  Now active from step 0 (still gated on bombs_left>0).
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
    "kill_assist":          0.75,  # 0.3 × kill_credit (2.5) — partial credit for assist
    # Zone control
    "zone_control":         0.005, # within 4 tiles of centre (6,6) AND ≥2 enemies alive
    # Danger shaping
    "danger_evasion":       0.15,
    "danger_enter":        -0.15,
    "own_blast_loiter_max": 0.25,  # Improvement 4: max loiter penalty (timer==1) — exceeds danger_enter at timer≤2
    # Wasted bomb
    "wasted_bomb":         -0.05,  # Improvement 5: bomb explodes, 0 boxes + 0 kills
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
_LATE_GAME_RAMP_START: int = 350      # Improvement 1: ramp begins here
_LATE_GAME_RAMP_END: int = 500        # Improvement 1: ramp reaches ×1.3 here
_LATE_GAME_MAX_BONUS: float = 0.3     # Improvement 1: maximum additive bonus to multiplier
_APPROACH_ENEMY_MIN_STEP: int = 0     # approach_enemy active from step 0 — early kill incentive required to escape local optimum
_TOTAL_STEPS: int = 500               # max episode length
_MAP_CENTRE: tuple[int, int] = (6, 6) # Improvement 3: zone control centre tile
_ZONE_CONTROL_RADIUS: int = 4         # Improvement 3: Manhattan radius for zone control
_RECENT_BLAST_HISTORY: int = 3        # Improvement 2: number of recent bomb blasts to track
_MAX_BOMB_TIMER: int = 7              # Improvement 2: bomb lifetime (steps)

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


def _late_game_multiplier(current_step: int) -> float:
    """
    IMPROVEMENT 1: Smooth ramp from 1.0 at step 350 to 1.3 at step 500.

    Formula: 1.0 + 0.3 * clamp((step - 350) / 150, 0, 1)
    Before step 350: multiplier = 1.0 (no effect)
    At step 350: multiplier = 1.0
    At step 425: multiplier = 1.15
    At step 500: multiplier = 1.3
    """
    ramp = max(0.0, (current_step - _LATE_GAME_RAMP_START) / float(_LATE_GAME_RAMP_END - _LATE_GAME_RAMP_START))
    return 1.0 + _LATE_GAME_MAX_BONUS * min(ramp, 1.0)


# ─────────────────────────────────────────────────────────────────────────── #
# Public API                                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

def compute_reward(
    prev_obs: dict | None,
    curr_obs: dict,
    action: int,
    agent_id: int,
    episode_stats: dict[str, object],
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
                        caller can track cumulative stats for aux features 5-6.
                        Also uses "recent_bomb_blasts" (list[np.ndarray]) for
                        kill-assist tracking (Improvement 2) — initialise to []
                        if absent; the function manages it automatically.
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

    # ── IMPROVEMENT 1: Smooth late-game multiplier ────────────────────── #
    lm = _late_game_multiplier(current_step)

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
    newly_dead_enemies: list[int] = []
    for i in range(len(prev_p)):
        if i == aid:
            continue
        was_alive = int(prev_p[i][2]) == 1
        now_dead = int(curr_p[i][2]) == 0
        if was_alive and now_dead:
            newly_dead_enemies.append(i)
            # Only credit if that enemy was in OUR blast zone last step
            if _enemy_in_my_blast(prev_obs, aid, i):
                kills += 1
    if kills:
        kill_r = kills * REWARDS["kill_credit"] * lm
        reward += kill_r
        info["kill_credit"] = kill_r
        episode_stats["kills"] = episode_stats.get("kills", 0) + kills  # type: ignore[operator]
        # Last kill secures win
        if curr_enemies == 0:
            reward += REWARDS["win"]
            info["win"] = float(REWARDS["win"])

    # ── IMPROVEMENT 2: Kill-assist reward ────────────────────────────── #
    # For each enemy who died this step but was NOT directly credited above,
    # check whether any of our recently-expired bomb blasts covered their
    # last known position.  We store up to _RECENT_BLAST_HISTORY blast masks
    # in episode_stats["recent_bomb_blasts"].
    recent_blasts: list[np.ndarray] = episode_stats.get("recent_bomb_blasts", [])  # type: ignore[assignment]
    if not isinstance(recent_blasts, list):
        recent_blasts = []
    assist_count = 0
    for enemy_id in newly_dead_enemies:
        # Skip if we already claimed a direct kill credit for this enemy
        if _enemy_in_my_blast(prev_obs, aid, enemy_id):
            continue
        ex = int(prev_p[enemy_id][0])
        ey = int(prev_p[enemy_id][1])
        for blast_mask in recent_blasts:
            if blast_mask[ex, ey]:
                assist_count += 1
                break  # one assist per enemy, regardless of how many blasts match
    if assist_count:
        assist_r = assist_count * REWARDS["kill_assist"] * lm
        reward += assist_r
        info["kill_assist"] = assist_r

    # ── Update recent_bomb_blasts for future assist tracking ─────────── #
    # When a bomb we placed expires this step (timer==1 or chain-triggered),
    # record its blast mask so the next _MAX_BOMB_TIMER steps can use it.
    # We cap the list at _RECENT_BLAST_HISTORY entries (oldest dropped first).
    prev_grid = np.asarray(prev_obs["map"], dtype=np.int32)
    curr_grid = np.asarray(curr_obs["map"], dtype=np.int32)
    prev_bombs_arr = _parse_bombs(prev_obs["bombs"])
    curr_bombs_arr_for_box = _parse_bombs(curr_obs["bombs"])

    if curr_bombs_arr_for_box is not None and len(curr_bombs_arr_for_box) > 0:
        curr_bomb_positions = set(
            (int(r[0]), int(r[1])) for r in curr_bombs_arr_for_box
        )
    else:
        curr_bomb_positions = set()

    new_blasts_this_step: list[np.ndarray] = []
    if prev_bombs_arr is not None:
        for row in prev_bombs_arr:
            bx, by, timer, owner = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            if int(owner) != aid:
                continue
            natural_explosion = int(timer) == 1
            chain_triggered = (bx, by) not in curr_bomb_positions
            if natural_explosion or chain_triggered:
                radius = 1 + int(prev_p[aid][4])
                new_blasts_this_step.append(_blast_mask(prev_grid, bx, by, radius))

    if new_blasts_this_step:
        recent_blasts = (recent_blasts + new_blasts_this_step)[-_RECENT_BLAST_HISTORY:]
    episode_stats["recent_bomb_blasts"] = recent_blasts  # type: ignore[assignment]

    # ── Confirmed box destruction (MY bombs: timer==1 OR chain-triggered) #
    # Chain-triggered: a bomb owned by me disappears between prev and curr
    # even though its timer was > 1 — it was detonated by another explosion
    # (chain reaction).  We detect this by comparing prev bomb positions to
    # curr bomb positions: any bomb that was present in prev but absent in
    # curr has exploded (either naturally at timer==1 or via chain).
    my_boxes_destroyed = 0
    wasted_bomb_fired = False  # Improvement 5 tracking

    if prev_bombs_arr is not None:
        for row in prev_bombs_arr:
            bx, by, timer, owner = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            if int(owner) != aid:
                continue
            natural_explosion = int(timer) == 1
            chain_triggered = (bx, by) not in curr_bomb_positions
            if not (natural_explosion or chain_triggered):
                continue
            radius = 1 + int(prev_p[aid][4])
            bmask = _blast_mask(prev_grid, bx, by, radius)
            # Count cells that were BOX in prev and are not BOX in curr
            destroyed = int(np.sum(
                (prev_grid == BOX) & (curr_grid != BOX) & bmask
            ))
            my_boxes_destroyed += destroyed
            # Improvement 5: track whether this bomb destroyed nothing and
            # killed nobody (attributed to me) — use `kills` not `newly_dead_enemies`
            # to avoid false-negative when enemies died from other causes this step.
            if destroyed == 0 and kills == 0:
                wasted_bomb_fired = True

    if my_boxes_destroyed > 0:
        box_r = my_boxes_destroyed * REWARDS["box_destroyed"] * lm
        reward += box_r
        info["box_destroyed"] = box_r
        episode_stats["boxes_destroyed"] = (
            episode_stats.get("boxes_destroyed", 0) + my_boxes_destroyed  # type: ignore[operator]
        )

    # ── IMPROVEMENT 5: Wasted-bomb penalty ───────────────────────────── #
    # Only penalise once per step even if multiple bombs expired with no effect.
    # We also skip the penalty if the bomb DID destroy boxes (handled above in
    # the per-bomb loop) or if enemies died this step (even if not attributed).
    if wasted_bomb_fired and my_boxes_destroyed == 0 and kills == 0:
        reward += REWARDS["wasted_bomb"]
        info["wasted_bomb"] = float(REWARDS["wasted_bomb"])

    # ── Item collection ───────────────────────────────────────────────── #
    prev_radius_bonus = int(prev_p[aid][4])
    curr_radius_bonus = int(curr_p[aid][4])
    prev_cap = int(prev_p[aid][3])
    curr_cap = int(curr_p[aid][3])
    px, py = int(prev_p[aid][0]), int(prev_p[aid][1])
    cx, cy = int(curr_p[aid][0]), int(curr_p[aid][1])
    # Radius items: radius_bonus only increases from item pickup — safe direct check.
    # Capacity items: bombs_left ALSO increases when a bomb detonates (slot returned).
    # Use tile check: the tile the agent moved to must have been code=4 (capacity item)
    # in prev_obs to count. This distinguishes item pickup from bomb-return.
    radius_item = curr_radius_bonus > prev_radius_bonus
    cap_item = curr_cap > prev_cap and prev_grid[cx, cy] == 4
    if radius_item or cap_item:
        item_r = REWARDS["item_collected"] * lm
        reward += item_r
        info["item_collected"] = item_r
        episode_stats["items_collected"] = episode_stats.get("items_collected", 0) + 1  # type: ignore[operator]

    # ── Item contest bonus ────────────────────────────────────────────── #
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

    # ── IMPROVEMENT 4: Own blast loiter — smooth penalty ─────────────── #
    # Penalty = (7 - timer) / 7 * 0.08, ranging from 0.01 (timer=6) to
    # 0.08 (timer=1).  Continuous deterrent rather than a stepped multiplier.
    own_timer = _own_blast_timer_at(curr_obs, aid, cx, cy)
    if curr_alive == 1 and own_timer is not None:
        t = max(1, min(int(own_timer), _DEFAULT_BOMB_TIMER - 1))  # clamp to [1, 6]
        loiter_scale = (_DEFAULT_BOMB_TIMER - t) / float(_DEFAULT_BOMB_TIMER)
        loiter_r = -(REWARDS["own_blast_loiter_max"] * loiter_scale)
        # Floor at a small minimum so there is always some deterrent
        loiter_r = min(loiter_r, -0.01)
        reward += loiter_r
        info["own_blast_loiter"] = loiter_r

    # ── IMPROVEMENT 3: Zone control ───────────────────────────────────── #
    # Small bonus when our agent sits near the map centre AND ≥2 enemies
    # are still alive (zone control is not meaningful in a 1v1 end-game).
    if curr_alive == 1 and curr_enemies >= 2:
        dist_to_centre = abs(cx - _MAP_CENTRE[0]) + abs(cy - _MAP_CENTRE[1])
        if dist_to_centre <= _ZONE_CONTROL_RADIUS:
            reward += REWARDS["zone_control"]
            info["zone_control"] = float(REWARDS["zone_control"])

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
            # Clamp delta to [-1.0, 1.0]: an agent moves at most 1 tile per step,
            # so (pd - cd) should naturally be in {-1, 0, 1}.  The clamp guards
            # against edge cases where Manhattan distance jumps > 1 due to agent
            # respawn / death events in the same step, preventing reward spikes.
            delta = float(np.clip(pd - cd, -1.0, 1.0))
            approach_r = REWARDS["approach_enemy"] * delta
            reward += approach_r
            if approach_r != 0.0:
                info["approach_enemy"] = approach_r

    return float(reward), info
