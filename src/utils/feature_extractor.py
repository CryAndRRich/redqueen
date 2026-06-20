from __future__ import annotations

import numpy as np

H: int = 13
W: int = 13
N_SPATIAL: int = 15
N_AUX: int = 7
BOMB_TIMER_MAX: int = 7
MAX_BOMB_CAPACITY: int = 5
MAX_BOMB_RADIUS_BONUS: int = 4
MAX_BOMBS: int = 16

GRASS: int = 0
WALL: int = 1
BOX: int = 2
ITEM_RADIUS: int = 3
ITEM_CAPACITY: int = 4


def _blast_mask_single(grid: np.ndarray, bx: int, by: int, radius: int) -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.bool_)
    mask[bx, by] = True
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        for r in range(1, radius + 1):
            x, y = bx + dx * r, by + dy * r
            if not (0 <= x < H and 0 <= y < W):
                break
            if grid[x, y] == WALL:
                break
            mask[x, y] = True
            if grid[x, y] == BOX:
                break
    return mask


def compute_blast_channels(
    grid: np.ndarray,
    bombs: np.ndarray,
    players: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    danger_now = np.zeros((H, W), dtype=np.bool_)
    danger_soon = np.zeros((H, W), dtype=np.bool_)
    danger_medium = np.zeros((H, W), dtype=np.bool_)

    if bombs is None or (hasattr(bombs, "__len__") and len(bombs) == 0):
        return danger_now, danger_soon, danger_medium

    bombs_arr = np.asarray(bombs)
    if bombs_arr.ndim == 1:
        if bombs_arr.size == 0:
            return danger_now, danger_soon, danger_medium
        bombs_arr = bombs_arr.reshape(1, -1)

    n_players = len(players)
    for row in bombs_arr:
        bx, by, timer, owner_id = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        if 0 <= owner_id < n_players:
            radius = 1 + int(players[owner_id][4])
        else:
            radius = 2
        blast = _blast_mask_single(grid, bx, by, radius)
        danger_medium |= blast
        if timer <= 3:
            danger_soon |= blast
        if timer <= 1:
            danger_now |= blast

    return danger_now, danger_soon, danger_medium


def extract_features(
    obs: dict,
    agent_id: int,
    step: int = 0,
    total_steps: int = 500,
    initial_boxes: int = 50,
    boxes_remaining: int | None = None,
    my_kills: int = 0,
    my_boxes_destroyed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    grid: np.ndarray = np.asarray(obs["map"], dtype=np.int8)
    players: np.ndarray = np.asarray(obs["players"], dtype=np.float32)
    bombs_raw = obs["bombs"]

    n_players = len(players)
    aid = int(agent_id)

    spatial = np.zeros((N_SPATIAL, H, W), dtype=np.float32)
    for ch, tile_val in enumerate([GRASS, WALL, BOX, ITEM_RADIUS, ITEM_CAPACITY]):
        spatial[ch] = (grid == tile_val)

    my_row = players[aid]
    if int(my_row[2]) == 1:
        spatial[5, int(my_row[0]), int(my_row[1])] = 1.0

    enemies = [
        (i, players[i])
        for i in range(n_players)
        if i != aid and int(players[i][2]) == 1
    ]
    if int(my_row[2]) == 1 and enemies:
        my_x, my_y = int(my_row[0]), int(my_row[1])
        enemies.sort(key=lambda t: abs(int(t[1][0]) - my_x) + abs(int(t[1][1]) - my_y))
    for slot, (_, ep) in enumerate(enemies[:3]):
        spatial[6 + slot, int(ep[0]), int(ep[1])] = 1.0

    bombs_arr = np.asarray(bombs_raw) if (bombs_raw is not None and len(bombs_raw) > 0) else None
    if bombs_arr is not None and bombs_arr.ndim == 1:
        bombs_arr = bombs_arr.reshape(1, -1)

    if bombs_arr is not None and len(bombs_arr) > 0:
        for row in bombs_arr:
            bx, by, timer, owner_id = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            t_norm = float(timer) / BOMB_TIMER_MAX
            if t_norm > spatial[9, bx, by]:
                spatial[9, bx, by] = t_norm
            if int(owner_id) == aid:
                spatial[10, bx, by] = 1.0
            else:
                spatial[11, bx, by] = 1.0

    danger_now, danger_soon, danger_medium = compute_blast_channels(
        grid, bombs_raw, players
    )
    spatial[12] = danger_now.astype(np.float32)
    spatial[13] = danger_soon.astype(np.float32)
    spatial[14] = danger_medium.astype(np.float32)

    my_bombs_left = float(my_row[3]) / MAX_BOMB_CAPACITY
    my_radius_bonus = float(my_row[4]) / MAX_BOMB_RADIUS_BONUS
    enemies_alive = float(sum(int(players[i][2]) for i in range(n_players) if i != aid)) / 3.0
    step_norm = float(step) / float(total_steps)

    if boxes_remaining is None:
        boxes_remaining = int((grid == BOX).sum())
    box_ratio = float(boxes_remaining) / max(1, initial_boxes)

    kills_norm = float(my_kills) / 3.0
    boxes_norm = float(my_boxes_destroyed) / 20.0

    aux = np.array(
        [my_bombs_left, my_radius_bonus, enemies_alive, step_norm,
         box_ratio, kills_norm, boxes_norm],
        dtype=np.float32,
    )

    return spatial, aux


def extract_features_padded(
    obs: dict,
    agent_id: int,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    return extract_features(obs, agent_id, **kwargs)


def obs_to_dict(obs: dict, agent_id: int, **kwargs) -> dict[str, np.ndarray]:
    spatial, aux = extract_features(obs, agent_id, **kwargs)
    return {"spatial": spatial, "aux": aux}


def count_boxes(grid: np.ndarray) -> int:
    return int((np.asarray(grid) == BOX).sum())
