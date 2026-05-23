"""
Action Masking — hard safety layer (Ngoại Công).

*** PROTECTED FILE — do not modify without explicit instruction (Golden Rule 1) ***

Masks actions that are physically impossible or guaranteed-suicide:
  - Movement into Wall / Box / active Bomb tile
  - PLACE_BOMB with 0 bombs_left
  - PLACE_BOMB when already standing on a bomb tile
  - PLACE_BOMB when no escape route exists within BOMB_TIMER steps

Returns a (6,) bool mask: True = action is allowed.
"""

from __future__ import annotations

from collections import deque

import numpy as np

# Action indices
STOP = 0
LEFT = 1
RIGHT = 2
UP = 3
DOWN = 4
PLACE_BOMB = 5

MOVES: dict[int, tuple[int, int]] = {
    STOP:  (0,  0),
    LEFT:  (-1, 0),
    RIGHT: (1,  0),
    UP:    (0, -1),
    DOWN:  (0,  1),
}

BOMB_TIMER: int = 7   # steps until explosion
H: int = 13
W: int = 13
WALL: int = 1
BOX: int = 2


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #

def _passable(grid: np.ndarray, x: int, y: int, bomb_set: frozenset) -> bool:
    """Tile can be stepped on (not wall, not box, not occupied by a bomb)."""
    if not (0 <= x < H and 0 <= y < W):
        return False
    cell = int(grid[x, y])
    if cell in (WALL, BOX):
        return False
    if (x, y) in bomb_set:
        return False
    return True


def _blast_mask(grid: np.ndarray, bx: int, by: int, radius: int) -> set[tuple[int, int]]:
    """Compute blast tile set for a single bomb."""
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
    """
    Build set of ALL danger tiles from current bombs + optional hypothetical bomb.
    Used by the escape BFS.
    """
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
    """
    BFS: can the agent reach a safe cell (not in danger) within max_steps moves?
    A cell is safe if it is NOT in the combined danger set.
    The agent may pass through danger tiles while escaping.
    """
    # Fast-path: current position already safe
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


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def compute_action_mask(obs: dict, agent_id: int) -> np.ndarray:
    """
    Compute a (6,) bool action mask for the given agent.

    True  = action is physically valid and not guaranteed-suicide.
    False = action must be masked out.

    Args:
        obs:       env observation dict ("map", "players", "bombs")
        agent_id:  index of our agent

    Returns:
        mask: np.ndarray shape (6,) dtype bool
    """
    mask = np.ones(6, dtype=np.bool_)

    grid: np.ndarray = np.asarray(obs["map"], dtype=np.int8)
    players: np.ndarray = np.asarray(obs["players"])
    bombs_raw = obs["bombs"]
    bombs_arr = np.asarray(bombs_raw) if (bombs_raw is not None and len(bombs_raw) > 0) else None

    aid = int(agent_id)
    if int(players[aid][2]) == 0:
        # Dead agent — mask everything except STOP
        mask[:] = False
        mask[STOP] = True
        return mask

    my_x = int(players[aid][0])
    my_y = int(players[aid][1])
    my_bombs_left = int(players[aid][3])
    my_radius = 1 + int(players[aid][4])

    # Build set of all current bomb positions (block movement)
    bomb_positions: frozenset
    if bombs_arr is not None and bombs_arr.ndim >= 1 and len(bombs_arr) > 0:
        if bombs_arr.ndim == 1:
            bombs_arr = bombs_arr.reshape(1, -1)
        bomb_positions = frozenset((int(r[0]), int(r[1])) for r in bombs_arr)
    else:
        bomb_positions = frozenset()

    # ------------------------------------------------------------------ #
    # Movement actions (1–4)                                              #
    # ------------------------------------------------------------------ #
    for action in (LEFT, RIGHT, UP, DOWN):
        dx, dy = MOVES[action]
        nx, ny = my_x + dx, my_y + dy
        if not _passable(grid, nx, ny, bomb_positions):
            mask[action] = False

    # ------------------------------------------------------------------ #
    # PLACE_BOMB (5)                                                       #
    # ------------------------------------------------------------------ #
    if my_bombs_left <= 0:
        mask[PLACE_BOMB] = False
    elif (my_x, my_y) in bomb_positions:
        # Already a bomb at current tile — engine won't place a new one
        mask[PLACE_BOMB] = False
    else:
        # Escape-route check: will we be able to leave the blast zone?
        danger = _danger_set(
            grid, bombs_raw, players,
            extra_bomb=(my_x, my_y),
            extra_radius=my_radius,
        )
        # The new bomb blocks the current tile; use updated bomb_positions
        new_bomb_positions = frozenset(bomb_positions | {(my_x, my_y)})
        if not _has_escape(grid, (my_x, my_y), new_bomb_positions, danger):
            mask[PLACE_BOMB] = False

    # Ensure at least STOP is always valid
    if not mask.any():
        mask[STOP] = True

    return mask


def apply_mask_to_logits(logits: np.ndarray, mask: np.ndarray, fill: float = -1e9) -> np.ndarray:
    """
    Zero-out forbidden actions in logit space.

    Args:
        logits: (6,) float array from policy network
        mask:   (6,) bool array (True = valid)
        fill:   value to assign to masked actions (default -1e9 ≈ -∞)

    Returns:
        masked_logits: (6,) float array
    """
    out = logits.copy()
    out[~mask] = fill
    return out
