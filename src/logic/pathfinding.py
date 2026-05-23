"""
BFS / A* pathfinding helpers for Bomberland.
Used by rule-based logic and training utilities — NOT imported inside agent.py.
"""

from __future__ import annotations

from collections import deque
from typing import Callable

import numpy as np

H: int = 13
W: int = 13
WALL: int = 1
BOX: int = 2

MOVES: dict[int, tuple[int, int]] = {
    1: (-1, 0),   # LEFT
    2: (1,  0),   # RIGHT
    3: (0, -1),   # UP
    4: (0,  1),   # DOWN
}


def _in_bounds(x: int, y: int) -> bool:
    return 0 <= x < H and 0 <= y < W


def _passable_move(grid: np.ndarray, x: int, y: int, blocked: frozenset) -> bool:
    """Tile is traversable for movement (not wall/box/bomb)."""
    if not _in_bounds(x, y):
        return False
    return int(grid[x, y]) not in (WALL, BOX) and (x, y) not in blocked


# --------------------------------------------------------------------------- #
# BFS: first action toward a set of targets                                    #
# --------------------------------------------------------------------------- #

def bfs_to_targets(
    grid: np.ndarray,
    start: tuple[int, int],
    targets: set[tuple[int, int]],
    blocked: frozenset = frozenset(),
    avoid: set[tuple[int, int]] | None = None,
) -> int | None:
    """
    Return the first action (1–4) that moves toward the nearest target.

    Args:
        grid:    (H, W) game grid
        start:   current position
        targets: goal tiles
        blocked: tiles that block movement (bomb positions, etc.)
        avoid:   tiles to avoid during search (danger zones)

    Returns:
        action int (1–4), or None if no path exists
    """
    if not targets:
        return None
    if avoid is None:
        avoid = set()

    visited: set[tuple[int, int]] = {start}
    queue: deque[tuple[tuple[int, int], int | None]] = deque([(start, None)])

    while queue:
        pos, first_action = queue.popleft()
        if pos in targets and first_action is not None:
            return first_action
        for action, (dx, dy) in MOVES.items():
            nx, ny = pos[0] + dx, pos[1] + dy
            npos = (nx, ny)
            if npos in visited:
                continue
            if not _passable_move(grid, nx, ny, blocked):
                continue
            if npos in avoid and npos not in targets:
                continue
            visited.add(npos)
            fa = action if first_action is None else first_action
            queue.append((npos, fa))

    return None


# --------------------------------------------------------------------------- #
# BFS: nearest safe cell (escape pathfinding)                                  #
# --------------------------------------------------------------------------- #

def bfs_to_safety(
    grid: np.ndarray,
    start: tuple[int, int],
    blocked: frozenset,
    danger: set[tuple[int, int]],
    search_depth: int = 8,
) -> int | None:
    """
    Return the first action (0–4) that leads to a safe cell.
    0 (STOP) is returned if current position is already safe.

    Args:
        danger: set of tiles that are dangerous

    Returns:
        action int, or None if no safe cell reachable
    """
    if start not in danger:
        return 0  # already safe

    visited: set[tuple[int, int]] = {start}
    queue: deque[tuple[tuple[int, int], int, int | None]] = deque([(start, 0, None)])

    while queue:
        pos, depth, first_action = queue.popleft()
        if pos not in danger and depth > 0:
            return first_action
        if depth >= search_depth:
            continue
        for action, (dx, dy) in MOVES.items():
            nx, ny = pos[0] + dx, pos[1] + dy
            npos = (nx, ny)
            if npos in visited:
                continue
            if not _passable_move(grid, nx, ny, blocked):
                continue
            visited.add(npos)
            fa = action if first_action is None else first_action
            queue.append((npos, depth + 1, fa))

    return None


# --------------------------------------------------------------------------- #
# Dijkstra: shortest path distance to all reachable tiles                      #
# --------------------------------------------------------------------------- #

def distance_map(
    grid: np.ndarray,
    start: tuple[int, int],
    blocked: frozenset = frozenset(),
) -> np.ndarray:
    """
    Return (H, W) int array of BFS distances from start. -1 = unreachable.
    """
    dist = np.full((H, W), -1, dtype=np.int32)
    dist[start] = 0
    queue: deque[tuple[tuple[int, int], int]] = deque([(start, 0)])

    while queue:
        pos, d = queue.popleft()
        for dx, dy in MOVES.values():
            nx, ny = pos[0] + dx, pos[1] + dy
            npos = (nx, ny)
            if not _in_bounds(nx, ny):
                continue
            if dist[nx, ny] != -1:
                continue
            if not _passable_move(grid, nx, ny, blocked):
                continue
            dist[nx, ny] = d + 1
            queue.append((npos, d + 1))

    return dist


# --------------------------------------------------------------------------- #
# Utility: adjacent passable tiles                                             #
# --------------------------------------------------------------------------- #

def neighbors(
    grid: np.ndarray,
    pos: tuple[int, int],
    blocked: frozenset = frozenset(),
) -> list[tuple[int, int]]:
    """Return list of passable neighboring positions."""
    result = []
    for dx, dy in MOVES.values():
        nx, ny = pos[0] + dx, pos[1] + dy
        if _passable_move(grid, nx, ny, blocked):
            result.append((nx, ny))
    return result
