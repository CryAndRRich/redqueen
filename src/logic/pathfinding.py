from __future__ import annotations

from collections import deque

import numpy as np

H: int = 13
W: int = 13
WALL: int = 1
BOX: int = 2

MOVES: dict[int, tuple[int, int]] = {
    1: (-1, 0),
    2: (1,  0),
    3: (0, -1),
    4: (0,  1),
}


def _in_bounds(x: int, y: int) -> bool:
    return 0 <= x < H and 0 <= y < W


def _passable_move(grid: np.ndarray, x: int, y: int, blocked: frozenset) -> bool:
    if not _in_bounds(x, y):
        return False
    return int(grid[x, y]) not in (WALL, BOX) and (x, y) not in blocked


def bfs_to_targets(
    grid: np.ndarray,
    start: tuple[int, int],
    targets: set[tuple[int, int]],
    blocked: frozenset = frozenset(),
    avoid: set[tuple[int, int]] | None = None,
) -> int | None:
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


def bfs_to_safety(
    grid: np.ndarray,
    start: tuple[int, int],
    blocked: frozenset,
    danger: set[tuple[int, int]],
    search_depth: int = 8,
) -> int | None:
    if start not in danger:
        return 0

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


def distance_map(
    grid: np.ndarray,
    start: tuple[int, int],
    blocked: frozenset = frozenset(),
) -> np.ndarray:
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


def neighbors(
    grid: np.ndarray,
    pos: tuple[int, int],
    blocked: frozenset = frozenset(),
) -> list[tuple[int, int]]:
    result = []
    for dx, dy in MOVES.values():
        nx, ny = pos[0] + dx, pos[1] + dy
        if _passable_move(grid, nx, ny, blocked):
            result.append((nx, ny))
    return result
