from __future__ import annotations

import numpy as np

REWARDS: dict[str, float] = {
    "win":                  5.0,
    "agent_death":         -3.0,
    "kill_credit":          2.5,
    "box_destroyed":        0.5,
    "item_collected":       0.4,
    "bomb_placed":          0.003,
    "chain_reaction":       0.5,
    "item_contest":         0.1,
    "kill_assist":          0.75,
    "zone_control":         0.005,
    "danger_evasion":       0.15,
    "danger_enter":        -0.15,
    "own_blast_loiter_max": 0.25,
    "wasted_bomb":         -0.05,
    "survival_step":        0.005,
    "standing_still":      -0.015,
    "time_penalty":        -0.003,
    "approach_enemy":       0.008,
}

_DEFAULT_BOMB_TIMER: int = 7
_LATE_GAME_RAMP_START: int = 350
_LATE_GAME_RAMP_END: int = 500
_LATE_GAME_MAX_BONUS: float = 0.3
_APPROACH_ENEMY_MIN_STEP: int = 0
_TOTAL_STEPS: int = 500
_MAP_CENTRE: tuple[int, int] = (6, 6)
_ZONE_CONTROL_RADIUS: int = 4
_RECENT_BLAST_HISTORY: int = 3
_MAX_BOMB_TIMER: int = 7

WALL: int = 1
BOX: int = 2


def _parse_bombs(bombs_raw: object) -> np.ndarray | None:
    if bombs_raw is None:
        return None
    arr = np.asarray(bombs_raw, dtype=np.int32)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _blast_mask(grid: np.ndarray, bx: int, by: int, radius: int) -> np.ndarray:
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
    return int(sum(int(players[i][2]) for i in range(len(players)) if i != agent_id))


def _manhattan_nearest_enemy(
    players: np.ndarray,
    agent_id: int,
    x: int,
    y: int,
) -> int | None:
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
    blast = _blast_mask(grid, placed_bx, placed_by, placed_radius)
    for row in all_bombs:
        bx, by, _timer, owner_id = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        if int(owner_id) == agent_id and bx == placed_bx and by == placed_by:
            continue
        if blast[bx, by]:
            return True
    return False


def _late_game_multiplier(current_step: int) -> float:
    ramp = max(0.0, (current_step - _LATE_GAME_RAMP_START) / float(_LATE_GAME_RAMP_END - _LATE_GAME_RAMP_START))
    return 1.0 + _LATE_GAME_MAX_BONUS * min(ramp, 1.0)


def compute_reward(
    prev_obs: dict | None,
    curr_obs: dict,
    action: int,
    agent_id: int,
    episode_stats: dict[str, object],
    current_step: int,
) -> tuple[float, dict[str, float]]:
    info: dict[str, float] = {}

    if prev_obs is None:
        return 0.0, info

    aid = int(agent_id)
    prev_p = np.asarray(prev_obs["players"], dtype=np.int32)
    curr_p = np.asarray(curr_obs["players"], dtype=np.int32)

    prev_alive = int(prev_p[aid][2])
    curr_alive = int(curr_p[aid][2])

    if prev_alive == 1 and curr_alive == 0:
        info["agent_death"] = float(REWARDS["agent_death"])
        return float(REWARDS["agent_death"]), info

    lm = _late_game_multiplier(current_step)

    reward = 0.0

    info["survival_step"] = float(REWARDS["survival_step"])
    info["time_penalty"] = float(REWARDS["time_penalty"])
    reward += REWARDS["survival_step"] + REWARDS["time_penalty"]

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
            if _enemy_in_my_blast(prev_obs, aid, i):
                kills += 1
    if kills:
        kill_r = kills * REWARDS["kill_credit"] * lm
        reward += kill_r
        info["kill_credit"] = kill_r
        episode_stats["kills"] = episode_stats.get("kills", 0) + kills
        if curr_enemies == 0:
            reward += REWARDS["win"]
            info["win"] = float(REWARDS["win"])

    recent_blasts: list[np.ndarray] = episode_stats.get("recent_bomb_blasts", [])
    if not isinstance(recent_blasts, list):
        recent_blasts = []
    assist_count = 0
    for enemy_id in newly_dead_enemies:
        if _enemy_in_my_blast(prev_obs, aid, enemy_id):
            continue
        ex = int(prev_p[enemy_id][0])
        ey = int(prev_p[enemy_id][1])
        for blast_mask in recent_blasts:
            if blast_mask[ex, ey]:
                assist_count += 1
                break
    if assist_count:
        assist_r = assist_count * REWARDS["kill_assist"] * lm
        reward += assist_r
        info["kill_assist"] = assist_r

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
    episode_stats["recent_bomb_blasts"] = recent_blasts

    my_boxes_destroyed = 0
    wasted_bomb_fired = False

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
            destroyed = int(np.sum(
                (prev_grid == BOX) & (curr_grid != BOX) & bmask
            ))
            my_boxes_destroyed += destroyed
            if destroyed == 0 and kills == 0:
                wasted_bomb_fired = True

    if my_boxes_destroyed > 0:
        box_r = my_boxes_destroyed * REWARDS["box_destroyed"] * lm
        reward += box_r
        info["box_destroyed"] = box_r
        episode_stats["boxes_destroyed"] = (
            episode_stats.get("boxes_destroyed", 0) + my_boxes_destroyed
        )

    if wasted_bomb_fired and my_boxes_destroyed == 0 and kills == 0:
        reward += REWARDS["wasted_bomb"]
        info["wasted_bomb"] = float(REWARDS["wasted_bomb"])

    prev_radius_bonus = int(prev_p[aid][4])
    curr_radius_bonus = int(curr_p[aid][4])
    prev_cap = int(prev_p[aid][3])
    curr_cap = int(curr_p[aid][3])
    px, py = int(prev_p[aid][0]), int(prev_p[aid][1])
    cx, cy = int(curr_p[aid][0]), int(curr_p[aid][1])
    radius_item = curr_radius_bonus > prev_radius_bonus
    cap_item = curr_cap > prev_cap and prev_grid[cx, cy] == 4
    if radius_item or cap_item:
        item_r = REWARDS["item_collected"] * lm
        reward += item_r
        info["item_collected"] = item_r
        episode_stats["items_collected"] = episode_stats.get("items_collected", 0) + 1

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

    bomb_placed = (action == 5) and (prev_cap > curr_cap)
    if bomb_placed:
        reward += REWARDS["bomb_placed"]
        info["bomb_placed"] = float(REWARDS["bomb_placed"])

        curr_bombs_arr = _parse_bombs(curr_obs["bombs"])
        if curr_bombs_arr is not None and len(curr_bombs_arr) > 0:
            placed_radius = 1 + curr_radius_bonus
            if _chain_reaction_bonus(
                curr_grid, cx, cy, placed_radius, curr_bombs_arr, aid
            ):
                reward += REWARDS["chain_reaction"]
                info["chain_reaction"] = float(REWARDS["chain_reaction"])

    if px == cx and py == cy and not bomb_placed:
        reward += REWARDS["standing_still"]
        info["standing_still"] = float(REWARDS["standing_still"])

    bombs_exist = (
        (prev_obs["bombs"] is not None and len(prev_obs["bombs"]) > 0)
        or (curr_obs["bombs"] is not None and len(curr_obs["bombs"]) > 0)
    )
    if bombs_exist:
        prev_in_blast, prev_timer = _danger_at(prev_obs, px, py)
        curr_in_blast, _ = _danger_at(curr_obs, cx, cy)
        if prev_in_blast and not curr_in_blast:
            urgency = 1.5 if (prev_timer is not None and prev_timer <= 1) else 1.0
            evasion_r = REWARDS["danger_evasion"] * urgency
            reward += evasion_r
            info["danger_evasion"] = evasion_r
        elif not prev_in_blast and curr_in_blast and (px != cx or py != cy):
            reward += REWARDS["danger_enter"]
            info["danger_enter"] = float(REWARDS["danger_enter"])

    own_timer = _own_blast_timer_at(curr_obs, aid, cx, cy)
    if curr_alive == 1 and own_timer is not None:
        t = max(1, min(int(own_timer), _DEFAULT_BOMB_TIMER - 1))
        loiter_scale = (_DEFAULT_BOMB_TIMER - t) / float(_DEFAULT_BOMB_TIMER)
        loiter_r = -(REWARDS["own_blast_loiter_max"] * loiter_scale)
        loiter_r = min(loiter_r, -0.01)
        reward += loiter_r
        info["own_blast_loiter"] = loiter_r

    if curr_alive == 1 and curr_enemies >= 2:
        dist_to_centre = abs(cx - _MAP_CENTRE[0]) + abs(cy - _MAP_CENTRE[1])
        if dist_to_centre <= _ZONE_CONTROL_RADIUS:
            reward += REWARDS["zone_control"]
            info["zone_control"] = float(REWARDS["zone_control"])

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
            delta = float(np.clip(pd - cd, -1.0, 1.0))
            approach_r = REWARDS["approach_enemy"] * delta
            reward += approach_r
            if approach_r != 0.0:
                info["approach_enemy"] = approach_r

    return float(reward), info
