"""
RedQueen submission agent — ONNX Runtime inference, CPU-only.

Self-contained: all feature extraction and action masking logic is inlined
so the file can be submitted with just model.onnx and requirements.txt.

Required files in same directory (zip root):
    agent.py        ← this file
    model.onnx      ← primary ONNX inference
    model.pt        ← TorchScript fallback (if onnxruntime unavailable)

Public interface (competition standard):
    class Agent:
        def __init__(self, agent_id: int)
        def act(self, obs: dict) -> int
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import onnxruntime as ort
    _HAS_ORT = True
except ImportError:
    _HAS_ORT = False

# ═══════════════════════════════════════════════════════════════════════════ #
#  INLINED CONSTANTS                                                           #
# ═══════════════════════════════════════════════════════════════════════════ #

_H = _W = 13
_GRASS, _WALL, _BOX = 0, 1, 2
_ITEM_RADIUS, _ITEM_CAP = 3, 4
_BOMB_TIMER_MAX = 7
_MAX_BOMB_CAP = 5
_MAX_RADIUS_BONUS = 4

_STOP, _LEFT, _RIGHT, _UP, _DOWN, _PLACE_BOMB = 0, 1, 2, 3, 4, 5
_MOVES = {_LEFT: (-1, 0), _RIGHT: (1, 0), _UP: (0, -1), _DOWN: (0, 1)}


# ═══════════════════════════════════════════════════════════════════════════ #
#  INLINED BLAST / DANGER COMPUTATION (vectorized)                            #
# ═══════════════════════════════════════════════════════════════════════════ #

def _blast_mask_single(grid: np.ndarray, bx: int, by: int, radius: int) -> np.ndarray:
    mask = np.zeros((_H, _W), dtype=np.bool_)
    mask[bx, by] = True
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        for r in range(1, radius + 1):
            x, y = bx + dx * r, by + dy * r
            if not (0 <= x < _H and 0 <= y < _W):
                break
            if grid[x, y] == _WALL:
                break
            mask[x, y] = True
            if grid[x, y] == _BOX:
                break
    return mask


def _compute_danger_channels(
    grid: np.ndarray, bombs_arr: Optional[np.ndarray], players: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    now  = np.zeros((_H, _W), dtype=np.bool_)
    soon = np.zeros((_H, _W), dtype=np.bool_)
    med  = np.zeros((_H, _W), dtype=np.bool_)
    if bombs_arr is None or len(bombs_arr) == 0:
        return now, soon, med
    n_p = len(players)
    for row in bombs_arr:
        bx, by, timer, oid = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        radius = 1 + int(players[oid][4]) if 0 <= oid < n_p else 2
        blast = _blast_mask_single(grid, bx, by, radius)
        med |= blast
        if timer <= 3:
            soon |= blast
        if timer <= 1:
            now |= blast
    return now, soon, med


# ═══════════════════════════════════════════════════════════════════════════ #
#  INLINED FEATURE EXTRACTOR                                                  #
# ═══════════════════════════════════════════════════════════════════════════ #

def _extract(
    obs: dict,
    agent_id: int,
    step: int = 0,
    total_steps: int = 500,
    initial_boxes: int = 50,
    boxes_remaining: Optional[int] = None,
    my_kills: int = 0,
    my_boxes_destroyed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (spatial (15,13,13), aux (7,)) float32."""
    grid    = np.asarray(obs["map"], dtype=np.int8)
    players = np.asarray(obs["players"], dtype=np.float32)
    bombs_raw = obs["bombs"]
    bombs_arr = (np.asarray(bombs_raw) if (bombs_raw is not None and len(bombs_raw) > 0)
                 else None)
    if bombs_arr is not None and bombs_arr.ndim == 1:
        bombs_arr = bombs_arr.reshape(1, -1)

    aid = int(agent_id)
    sp  = np.zeros((15, _H, _W), dtype=np.float32)

    # Channels 0–4: map one-hot
    for ch, v in enumerate([_GRASS, _WALL, _BOX, _ITEM_RADIUS, _ITEM_CAP]):
        sp[ch] = (grid == v)

    # Channel 5: my position
    my = players[aid]
    if int(my[2]) == 1:
        sp[5, int(my[0]), int(my[1])] = 1.0

    # Channels 6–8: enemy positions sorted by Manhattan distance
    enemies = [(i, players[i]) for i in range(len(players))
               if i != aid and int(players[i][2]) == 1]
    if int(my[2]) == 1 and enemies:
        mx, my_ = int(my[0]), int(my[1])
        enemies.sort(key=lambda t: abs(int(t[1][0]) - mx) + abs(int(t[1][1]) - my_))
    for slot, (_, ep) in enumerate(enemies[:3]):
        sp[6 + slot, int(ep[0]), int(ep[1])] = 1.0

    # Channels 9–11: bomb timer / ownership
    if bombs_arr is not None:
        for row in bombs_arr:
            bx, by, timer, oid = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            t_norm = float(timer) / _BOMB_TIMER_MAX
            if t_norm > sp[9, bx, by]:
                sp[9, bx, by] = t_norm
            if int(oid) == aid:
                sp[10, bx, by] = 1.0
            else:
                sp[11, bx, by] = 1.0

    # Channels 12–14: danger maps
    dnow, dsoon, dmed = _compute_danger_channels(grid, bombs_arr, players)
    sp[12] = dnow;  sp[13] = dsoon;  sp[14] = dmed

    # Aux scalars
    if boxes_remaining is None:
        boxes_remaining = int((grid == _BOX).sum())
    enemies_alive = float(sum(int(players[i][2]) for i in range(len(players)) if i != aid)) / 3.0
    aux = np.array([
        float(my[3]) / _MAX_BOMB_CAP,
        float(my[4]) / _MAX_RADIUS_BONUS,
        enemies_alive,
        float(step) / float(total_steps),
        float(boxes_remaining) / max(1, initial_boxes),
        float(my_kills) / 3.0,
        float(my_boxes_destroyed) / 20.0,
    ], dtype=np.float32)

    return sp, aux


# ═══════════════════════════════════════════════════════════════════════════ #
#  INLINED ACTION MASKING                                                      #
# ═══════════════════════════════════════════════════════════════════════════ #

def _passable(grid: np.ndarray, x: int, y: int, bomb_set: frozenset) -> bool:
    if not (0 <= x < _H and 0 <= y < _W):
        return False
    return int(grid[x, y]) not in (_WALL, _BOX) and (x, y) not in bomb_set


def _blast_set(grid: np.ndarray, bx: int, by: int, radius: int) -> set:
    tiles = {(bx, by)}
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        for r in range(1, radius + 1):
            x, y = bx + dx * r, by + dy * r
            if not (0 <= x < _H and 0 <= y < _W): break
            if grid[x, y] == _WALL: break
            tiles.add((x, y))
            if grid[x, y] == _BOX: break
    return tiles


def _has_escape(grid: np.ndarray, start: tuple, bomb_set: frozenset,
                danger: set, max_steps: int = _BOMB_TIMER_MAX) -> bool:
    if start not in danger:
        return True
    visited = {start}
    q = deque([(start, 0)])
    while q:
        pos, d = q.popleft()
        if d >= max_steps:
            continue
        for dx, dy in _MOVES.values():
            nx, ny = pos[0] + dx, pos[1] + dy
            npos = (nx, ny)
            if npos in visited:
                continue
            if not _passable(grid, nx, ny, bomb_set):
                continue
            visited.add(npos)
            if npos not in danger:
                return True
            q.append((npos, d + 1))
    return False


def _compute_mask(obs: dict, agent_id: int) -> np.ndarray:
    mask = np.ones(6, dtype=np.bool_)
    grid    = np.asarray(obs["map"], dtype=np.int8)
    players = np.asarray(obs["players"])
    bombs_raw = obs["bombs"]

    aid = int(agent_id)
    if int(players[aid][2]) == 0:
        mask[:] = False; mask[_STOP] = True
        return mask

    my_x, my_y     = int(players[aid][0]), int(players[aid][1])
    bombs_left      = int(players[aid][3])
    my_radius       = 1 + int(players[aid][4])

    bombs_arr = (np.asarray(bombs_raw)
                 if (bombs_raw is not None and len(bombs_raw) > 0) else None)
    if bombs_arr is not None and bombs_arr.ndim == 1:
        bombs_arr = bombs_arr.reshape(1, -1)

    bomb_pos = (frozenset((int(r[0]), int(r[1])) for r in bombs_arr)
                if bombs_arr is not None else frozenset())

    for action, (dx, dy) in _MOVES.items():
        nx, ny = my_x + dx, my_y + dy
        if not _passable(grid, nx, ny, bomb_pos):
            mask[action] = False

    if bombs_left <= 0 or (my_x, my_y) in bomb_pos:
        mask[_PLACE_BOMB] = False
    else:
        n_p = len(players)
        danger: set = set()
        if bombs_arr is not None:
            for row in bombs_arr:
                bx, by, _, oid = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                r = 1 + int(players[oid][4]) if 0 <= oid < n_p else 2
                danger |= _blast_set(grid, bx, by, r)
        danger |= _blast_set(grid, my_x, my_y, my_radius)
        new_bp = frozenset(bomb_pos | {(my_x, my_y)})
        if not _has_escape(grid, (my_x, my_y), new_bp, danger):
            mask[_PLACE_BOMB] = False

    if not mask.any():
        mask[_STOP] = True
    return mask


# ═══════════════════════════════════════════════════════════════════════════ #
#  AGENT CLASS                                                                 #
# ═══════════════════════════════════════════════════════════════════════════ #

class Agent:
    """
    RedQueen ONNX submission agent.
    Loads model.onnx from the same directory as this file.
    """

    def __init__(self, agent_id: int) -> None:
        self.agent_id = int(agent_id)
        _dir = Path(__file__).parent

        onnx_path = _dir / "model.onnx"
        pt_path   = _dir / "model.pt"

        if _HAS_ORT and onnx_path.exists():
            self._sess = ort.InferenceSession(
                str(onnx_path),
                providers=["CPUExecutionProvider"],
            )
            self._backend = "onnx"
        elif pt_path.exists():
            import torch
            self._ts_model = torch.jit.load(str(pt_path), map_location="cpu")
            self._ts_model.eval()
            self._backend = "torch"
        else:
            raise FileNotFoundError(
                f"Neither model.onnx nor model.pt found in {_dir}. "
                "Export with: python -m src.utils.export_onnx --checkpoint <ckpt>"
            )

        # Per-game state
        self._step: int = 0
        self._initial_boxes: int = 50
        self._my_kills: int = 0
        self._my_boxes: int = 0
        self._prev_players: Optional[np.ndarray] = None
        self._prev_grid: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # Competition interface                                                #
    # ------------------------------------------------------------------ #

    def act(self, obs: dict) -> int:
        """
        Select an action given the current observation.

        Args:
            obs: dict with keys "map", "players", "bombs"

        Returns:
            action (int in [0, 5])
        """
        curr_players = np.asarray(obs["players"])
        curr_grid    = np.asarray(obs["map"], dtype=np.int8)

        # ── Update per-game trackers ──────────────────────────────────── #
        if self._prev_players is not None:
            prev_enemies = sum(
                int(self._prev_players[i][2]) for i in range(4) if i != self.agent_id
            )
            curr_enemies = sum(
                int(curr_players[i][2]) for i in range(4) if i != self.agent_id
            )
            self._my_kills += max(0, prev_enemies - curr_enemies)

        if self._prev_grid is not None:
            self._my_boxes += int(
                ((self._prev_grid == _BOX) & (curr_grid != _BOX)).sum()
            )

        boxes_now = int((curr_grid == _BOX).sum())

        # ── Feature extraction ────────────────────────────────────────── #
        spatial, aux = _extract(
            obs,
            agent_id=self.agent_id,
            step=self._step,
            initial_boxes=self._initial_boxes,
            boxes_remaining=boxes_now,
            my_kills=self._my_kills,
            my_boxes_destroyed=self._my_boxes,
        )

        # ── Inference ─────────────────────────────────────────────────── #
        if self._backend == "onnx":
            logits: np.ndarray = self._sess.run(
                None,
                {
                    "spatial": spatial[np.newaxis],   # (1, 15, 13, 13)
                    "aux":     aux[np.newaxis],        # (1, 7)
                },
            )[0][0]  # shape (6,)
        else:
            import torch
            with torch.no_grad():
                sp_t = torch.from_numpy(spatial[np.newaxis])
                ax_t = torch.from_numpy(aux[np.newaxis])
                logits = self._ts_model(sp_t, ax_t).numpy()[0]  # shape (6,)

        # ── Action masking ────────────────────────────────────────────── #
        mask = _compute_mask(obs, self.agent_id)
        logits[~mask] = -1e9

        action = int(np.argmax(logits))

        # ── Advance state ─────────────────────────────────────────────── #
        self._step += 1
        self._prev_players = curr_players.copy()
        self._prev_grid    = curr_grid.copy()

        # Reset on new game (detect by step reset to 0 externally is not
        # possible; engine calls __init__ per match, so no reset needed)

        return action
