from __future__ import annotations

from collections import deque

import numpy as np

STOP = 0
LEFT = 1
RIGHT = 2
UP = 3
DOWN = 4
PLACE_BOMB = 5

MOVES: dict[int, tuple[int, int]] = {
    LEFT:  (-1, 0),
    RIGHT: (1,  0),
    UP:    (0, -1),
    DOWN:  (0,  1),
}

BOMB_TIMER: int = 7
H: int = 13
W: int = 13
WALL: int = 1
BOX: int = 2


def _passable(grid: np.ndarray, x: int, y: int, bomb_set: frozenset) -> bool:
    if not (0 <= x < H and 0 <= y < W):
        return False
    cell = int(grid[x, y])
    if cell in (WALL, BOX):
        return False
    if (x, y) in bomb_set:
        return False
    return True


def _blast_mask(grid: np.ndarray, bx: int, by: int, radius: int) -> set[tuple[int, int]]:
    tiles: set[tuple[int, int]] = {(bx, by)}
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        for r in range(1, radius + 1):
            x, y = bx + dx * r, by + dy * r
            if not (0 <= x < H and 0 <= y < W):
                break
            cell = int(grid[x, y])
            if cell == WALL:
                break
            tiles.add((x, y))
            if cell == BOX:
                break
    return tiles


def _danger_set(
    grid: np.ndarray,
    bombs: np.ndarray,
    players: np.ndarray,
    extra_bomb: tuple[int, int] | None = None,
    extra_radius: int = 1,
) -> set[tuple[int, int]]:
    danger: set[tuple[int, int]] = set()
    if bombs is not None and len(bombs) > 0:
        bombs_arr = np.asarray(bombs)
        if bombs_arr.ndim == 1:
            bombs_arr = bombs_arr.reshape(1, -1)
        n_players = len(players)
        for row in bombs_arr:
            bx, by, _, owner_id = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            if 0 <= owner_id < n_players:
                radius = 1 + int(players[owner_id][4])
            else:
                radius = 2
            danger |= _blast_mask(grid, bx, by, radius)
    if extra_bomb is not None:
        danger |= _blast_mask(grid, extra_bomb[0], extra_bomb[1], extra_radius)
    return danger


def _has_escape(
    grid: np.ndarray,
    start: tuple[int, int],
    bomb_set: frozenset,
    danger: set[tuple[int, int]],
    max_steps: int = BOMB_TIMER,
) -> bool:
    if start not in danger:
        return True

    visited: set[tuple[int, int]] = {start}
    queue: deque[tuple[tuple[int, int], int]] = deque([(start, 0)])

    while queue:
        pos, depth = queue.popleft()
        if depth >= max_steps:
            continue
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = pos[0] + dx, pos[1] + dy
            npos = (nx, ny)
            if npos in visited:
                continue
            if not _passable(grid, nx, ny, bomb_set):
                continue
            visited.add(npos)
            if npos not in danger:
                return True
            queue.append((npos, depth + 1))

    return False


def compute_action_mask(obs: dict, agent_id: int) -> np.ndarray:
    mask = np.ones(6, dtype=np.bool_)

    grid: np.ndarray = np.asarray(obs["map"], dtype=np.int8)
    players: np.ndarray = np.asarray(obs["players"])
    bombs_raw = obs["bombs"]
    bombs_arr = np.asarray(bombs_raw) if (bombs_raw is not None and len(bombs_raw) > 0) else None

    aid = int(agent_id)
    if int(players[aid][2]) == 0:
        mask[:] = False
        mask[STOP] = True
        return mask

    my_x = int(players[aid][0])
    my_y = int(players[aid][1])
    my_bombs_left = int(players[aid][3])
    my_radius = 1 + int(players[aid][4])

    bomb_positions: frozenset
    if bombs_arr is not None and bombs_arr.ndim >= 1 and len(bombs_arr) > 0:
        if bombs_arr.ndim == 1:
            bombs_arr = bombs_arr.reshape(1, -1)
        bomb_positions = frozenset((int(r[0]), int(r[1])) for r in bombs_arr)
    else:
        bomb_positions = frozenset()

    for action in (LEFT, RIGHT, UP, DOWN):
        dx, dy = MOVES[action]
        nx, ny = my_x + dx, my_y + dy
        if not _passable(grid, nx, ny, bomb_positions):
            mask[action] = False

    if my_bombs_left <= 0:
        mask[PLACE_BOMB] = False
    elif (my_x, my_y) in bomb_positions:
        mask[PLACE_BOMB] = False
    else:
        danger = _danger_set(
            grid, bombs_raw, players,
            extra_bomb=(my_x, my_y),
            extra_radius=my_radius,
        )
        new_bomb_positions = frozenset(bomb_positions | {(my_x, my_y)})
        if not _has_escape(grid, (my_x, my_y), new_bomb_positions, danger):
            mask[PLACE_BOMB] = False

    if not mask.any():
        mask[STOP] = True

    return mask


def apply_mask_to_logits(logits: np.ndarray, mask: np.ndarray, fill: float = -1e9) -> np.ndarray:
    out = logits.copy()
    out[~mask] = fill
    return out
