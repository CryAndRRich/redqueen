from __future__ import annotations

import random
from collections import deque


class TrapperRuleAgent:

    MOVES = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
    team_id = "TrapperRuleAgent"

    def __init__(self, agent_id: int) -> None:
        self.agent_id = int(agent_id)

    def act(self, obs: dict) -> int:
        grid    = obs["map"]
        players = obs["players"]
        bombs   = obs["bombs"]

        if self.agent_id >= len(players) or players[self.agent_id][2] != 1:
            return 0

        my_x, my_y, _, bombs_left, bomb_bonus = players[self.agent_id]
        my_pos    = (int(my_x), int(my_y))
        my_radius = max(1, int(bomb_bonus) + 1)
        bomb_pos  = {(int(b[0]), int(b[1])) for b in bombs}

        enemies = [(int(p[0]), int(p[1])) for i, p in enumerate(players)
                   if i != self.agent_id and p[2] == 1]

        blocked = {(int(p[0]), int(p[1])) for i, p in enumerate(players)
                   if p[2] == 1 and i != self.agent_id} | bomb_pos
        blocked.discard(my_pos)

        danger_soon, danger_now = self._danger_tiles(grid, bombs, players)

        if my_pos in danger_now or my_pos in danger_soon:
            action = self._escape(grid, my_pos, blocked, danger_now, danger_soon)
            return action if action is not None else 0

        if bombs_left > 0 and my_pos not in bomb_pos:
            if self._can_bomb_hit_enemy(grid, my_pos, enemies, my_radius):
                if self._can_escape_after_placing(grid, my_pos, blocked, danger_soon, my_radius):
                    return 5
            if enemies and my_pos in self._adjacent_to_enemies(enemies):
                if self._can_escape_after_placing(grid, my_pos, blocked, danger_soon, my_radius):
                    return 5

        if enemies:
            action = self._pursue_enemy(grid, my_pos, enemies, blocked, danger_now)
            if action is not None:
                return action

        items = self._item_tiles(grid)
        if items:
            action = self._bfs_move(grid, my_pos, items, blocked, avoid=danger_soon)
            if action is not None:
                return action

        if bombs_left > 0 and my_pos not in bomb_pos:
            if self._count_boxes_in_blast(grid, my_pos, my_radius) > 0:
                if self._can_escape_after_placing(grid, my_pos, blocked, danger_soon, my_radius):
                    return 5
        box_spots = self._box_bomb_spots(grid, blocked)
        if box_spots:
            action = self._bfs_move(grid, my_pos, box_spots, blocked, avoid=danger_soon)
            if action is not None:
                return action

        valid = self._valid_actions(grid, my_pos, blocked)
        safe  = [a for a in valid if self._next_pos(my_pos, a) not in danger_soon]
        return random.choice(safe) if safe else (random.choice(valid) if valid else 0)

    def _pursue_enemy(
        self,
        grid,
        start: tuple,
        enemies: list[tuple],
        blocked: set,
        danger_now: set,
    ) -> int | None:
        targets: set[tuple] = set(enemies) | self._adjacent_to_enemies(enemies)
        q: deque = deque([(start, None)])
        seen: set = {start}
        while q:
            pos, first_action = q.popleft()
            if pos in targets and first_action is not None:
                return first_action
            for a in [1, 2, 3, 4]:
                nx, ny = self._next_pos(pos, a)
                npos = (nx, ny)
                if npos in seen:
                    continue
                if not self._passable(grid, nx, ny):
                    continue
                if npos in blocked and npos not in targets:
                    continue
                if npos in danger_now:
                    continue
                seen.add(npos)
                q.append((npos, a if first_action is None else first_action))
        return None

    def _adjacent_to_enemies(self, enemies: list[tuple]) -> set[tuple]:
        adj: set[tuple] = set()
        for ex, ey in enemies:
            for dx, dy in self.MOVES.values():
                adj.add((ex + dx, ey + dy))
        return adj

    def _bfs_move(
        self,
        grid,
        start: tuple,
        targets: set,
        blocked: set,
        avoid: set,
    ) -> int | None:
        if not targets:
            return None
        q: deque = deque([(start, None)])
        seen: set = {start}
        while q:
            pos, first_action = q.popleft()
            if pos in targets and first_action is not None:
                return first_action
            for a in [1, 2, 3, 4]:
                nx, ny = self._next_pos(pos, a)
                npos = (nx, ny)
                if npos in seen or not self._passable(grid, nx, ny):
                    continue
                if npos in blocked and npos not in targets:
                    continue
                if npos in avoid:
                    continue
                seen.add(npos)
                q.append((npos, a if first_action is None else first_action))
        return None

    def _escape(
        self,
        grid,
        my_pos: tuple,
        blocked: set,
        danger_now: set,
        danger_soon: set,
    ) -> int | None:
        best_action = None
        best_score  = -10 ** 9
        for a in self._valid_actions(grid, my_pos, blocked):
            if a == 0:
                continue
            npos = self._next_pos(my_pos, a)
            if npos in danger_now:
                continue
            score = (6 if npos not in danger_soon else 0) + self._open_neighbors(grid, npos, blocked)
            if score > best_score:
                best_score  = score
                best_action = a
        return best_action

    def _can_bomb_hit_enemy(self, grid, my_pos: tuple, enemies: list[tuple], radius: int) -> bool:
        blast = self._blast_tiles(grid, my_pos[0], my_pos[1], radius)
        return any(e in blast for e in enemies)

    def _can_escape_after_placing(
        self,
        grid,
        my_pos: tuple,
        blocked: set,
        danger_soon: set,
        radius: int,
    ) -> bool:
        combined = set(danger_soon) | self._blast_tiles(grid, my_pos[0], my_pos[1], radius)
        new_blocked = blocked | {my_pos}
        return self._bfs_to_safe(grid, my_pos, new_blocked, combined) is not None

    def _bfs_to_safe(self, grid, start: tuple, blocked: set, danger: set) -> int | None:
        q: deque = deque([(start, 0, None)])
        seen: set = {start}
        while q:
            pos, d, first_action = q.popleft()
            if pos not in danger and d > 0:
                return first_action
            if d >= 7:
                continue
            for a in [1, 2, 3, 4]:
                nx, ny = self._next_pos(pos, a)
                npos = (nx, ny)
                if npos in seen or not self._passable(grid, nx, ny):
                    continue
                if npos in blocked:
                    continue
                seen.add(npos)
                q.append((npos, d + 1, a if first_action is None else first_action))
        return None

    def _count_boxes_in_blast(self, grid, my_pos: tuple, radius: int) -> int:
        return sum(1 for x, y in self._blast_tiles(grid, my_pos[0], my_pos[1], radius)
                   if grid[x, y] == 2)

    def _blast_tiles(self, grid, bx: int, by: int, radius: int) -> set[tuple]:
        h, w = grid.shape
        tiles: set[tuple] = {(bx, by)}
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            for r in range(1, radius + 1):
                x, y = bx + dx * r, by + dy * r
                if not (0 <= x < h and 0 <= y < w):
                    break
                if grid[x, y] == 1:
                    break
                tiles.add((x, y))
                if grid[x, y] == 2:
                    break
        return tiles

    def _danger_tiles(self, grid, bombs, players) -> tuple[set, set]:
        soon: set = set()
        now:  set = set()
        for b in bombs:
            bx, by, timer = int(b[0]), int(b[1]), int(b[2])
            oid = int(b[3]) if len(b) > 3 else -1
            if timer <= 0:
                continue
            radius = max(1, int(players[oid][4]) + 1) if 0 <= oid < len(players) else 2
            blast  = self._blast_tiles(grid, bx, by, radius)
            soon  |= blast
            if timer <= 1:
                now |= blast
        return soon, now

    def _passable(self, grid, x: int, y: int) -> bool:
        return (0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]
                and int(grid[x, y]) in (0, 3, 4))

    def _next_pos(self, pos: tuple, action: int) -> tuple:
        dx, dy = self.MOVES[action]
        return pos[0] + dx, pos[1] + dy

    def _valid_actions(self, grid, my_pos: tuple, blocked: set) -> list[int]:
        actions = [0]
        for a in [1, 2, 3, 4]:
            nx, ny = self._next_pos(my_pos, a)
            if self._passable(grid, nx, ny) and (nx, ny) not in blocked:
                actions.append(a)
        return actions

    def _open_neighbors(self, grid, pos: tuple, blocked: set) -> int:
        return sum(
            1 for a in [1, 2, 3, 4]
            if self._passable(grid, *(self._next_pos(pos, a)))
            and self._next_pos(pos, a) not in blocked
        )

    def _item_tiles(self, grid) -> set[tuple]:
        return {(x, y) for x in range(grid.shape[0]) for y in range(grid.shape[1])
                if int(grid[x, y]) in (3, 4)}

    def _box_bomb_spots(self, grid, blocked: set) -> set[tuple]:
        spots: set[tuple] = set()
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if grid[x, y] != 2:
                    continue
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if self._passable(grid, nx, ny) and (nx, ny) not in blocked:
                        spots.add((nx, ny))
        return spots
