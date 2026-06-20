from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PIL import Image, ImageDraw, ImageFont

pygame = importlib.import_module("pygame")


_TOP_BAR = 48
_RIGHT_PANEL = 280
_CELL = 40
_WALL, _BOX = 1, 2


def _blast_tiles(grid: list, bx: int, by: int, radius: int) -> set[tuple[int, int]]:
    H, W = len(grid), len(grid[0])
    tiles = {(bx, by)}
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for r in range(1, radius + 1):
            tx, ty = bx + dx * r, by + dy * r
            if not (0 <= tx < H and 0 <= ty < W):
                break
            cell = int(grid[tx][ty])
            if cell == _WALL:
                break
            tiles.add((tx, ty))
            if cell == _BOX:
                break
    return tiles


def _explosion_tiles(prev_obs: dict | None, obs: dict) -> set[tuple[int, int]]:
    if prev_obs is None:
        return set()
    prev_bombs = prev_obs.get("bombs", [])
    curr_bombs = obs.get("bombs", [])
    curr_pos = {(int(b[0]), int(b[1])) for b in curr_bombs}
    prev_players = prev_obs["players"]
    grid = prev_obs["map"]
    tiles: set = set()
    for b in prev_bombs:
        bx, by, timer, oid = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        if timer > 1 and (bx, by) in curr_pos:
            continue
        radius = 1 + int(prev_players[oid][4]) if 0 <= oid < len(prev_players) else 2
        tiles |= _blast_tiles(grid, bx, by, radius)
    return tiles


_PLAYER_COLORS = [(220, 50, 50, 255), (50, 50, 220, 255), (30, 150, 30, 255), (200, 140, 0, 255)]


def render_frame(obs: dict, prev_obs: dict | None = None, agent_names: list[str] | None = None) -> Image.Image:
    grid = obs["map"]
    players = obs["players"]
    bombs = obs.get("bombs", [])
    H, W = len(grid), len(grid[0])

    board_w = W * _CELL
    total_w = board_w + _RIGHT_PANEL
    total_h = H * _CELL + _TOP_BAR

    img = Image.new("RGBA", (total_w, total_h), (245, 245, 245, 255))
    draw = ImageDraw.Draw(img, "RGBA")

    try:
        fn = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
        fs = ImageFont.truetype("DejaVuSans.ttf", 12)
    except Exception:
        fn = fs = ImageFont.load_default()

    draw.rectangle([0, 0, total_w, _TOP_BAR], fill=(30, 30, 30, 255))
    draw.text((10, 14), f"Step {obs.get('_step', '?')}", fill=(245, 245, 245, 255), font=fn)

    draw.rectangle([0, _TOP_BAR, board_w, total_h], fill=(144, 238, 144, 255))

    _COLORS = {0: (144, 238, 144), 1: (80, 80, 80), 2: (139, 69, 19), 3: (255, 200, 200), 4: (200, 200, 255)}

    for row in range(H):
        for col in range(W):
            x0, y0 = col * _CELL, row * _CELL + _TOP_BAR
            x1, y1 = x0 + _CELL, y0 + _CELL
            ct = int(grid[row][col])
            color = _COLORS.get(ct, (144, 238, 144))
            draw.rectangle([x0, y0, x1, y1], fill=color + (255,))
            if ct == 2:
                draw.line([x0, y0, x1, y1], fill=(101, 67, 33, 255), width=2)
                draw.line([x1, y0, x0, y1], fill=(101, 67, 33, 255), width=2)
            if ct in (3, 4):
                label = "R" if ct == 3 else "C"
                draw.text((x0 + 14, y0 + 12), label, fill=(80, 80, 80, 255), font=fn)

    for row, col in _explosion_tiles(prev_obs, obs):
        x0, y0 = col * _CELL, row * _CELL + _TOP_BAR
        draw.rectangle([x0, y0, x0 + _CELL, y0 + _CELL], fill=(255, 140, 0, 140))

    for b in bombs:
        bx, by, timer = int(b[0]), int(b[1]), int(b[2])
        cx = by * _CELL + _CELL // 2
        cy = bx * _CELL + _TOP_BAR + _CELL // 2
        r = _CELL // 4
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(20, 20, 20, 255))
        draw.text((cx - 4, cy - 7), str(timer), fill=(255, 255, 255, 255), font=fs)

    for i, p in enumerate(players):
        if int(p[2]) != 1:
            continue
        px_col, px_row = int(p[1]), int(p[0])
        cx = px_col * _CELL + _CELL // 2
        cy = px_row * _CELL + _TOP_BAR + _CELL // 2
        r = _CELL // 3
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=_PLAYER_COLORS[i % 4])
        draw.text((cx - 4, cy - 7), str(i), fill=(255, 255, 255, 255), font=fn)

    px0 = board_w
    draw.rectangle([px0, 0, total_w, total_h], fill=(52, 58, 64, 255))
    draw.line([px0, 0, px0, total_h], fill=(20, 20, 20, 255), width=2)
    draw.text((px0 + 10, _TOP_BAR + 8), "Players", fill=(245, 245, 245, 255), font=fn)
    y = _TOP_BAR + 36
    for i, p in enumerate(players):
        alive = int(p[2]) == 1
        name = (agent_names[i] if agent_names and i < len(agent_names) else f"Agent {i}")[:24]
        color = _PLAYER_COLORS[i % 4][:3]
        draw.ellipse([px0 + 8, y, px0 + 20, y + 12], fill=color + (255,))
        draw.text((px0 + 26, y), name, fill=(240, 240, 240, 255), font=fs)
        y += 18
        status_color = (120, 220, 140, 255) if alive else (220, 100, 100, 255)
        draw.text((px0 + 10, y), "Alive" if alive else "Dead", fill=status_color, font=fs)
        y += 18
        draw.text((px0 + 10, y), f"Bombs:{int(p[3])}  +R:{int(p[4])}", fill=(180, 180, 180, 255), font=fs)
        y += 26

    return img


class ReplayViewer:
    def __init__(self, history: list[dict], meta: dict | None = None, title: str = "Bomberland Replay", fps: int = 8):
        self.history = history
        self.meta = meta or {}
        self.fps = fps
        self.paused = False
        self.step_idx = 0
        self.last_tick = time.time()

        agent_names = self.meta.get("agent_names", [])
        first = self._make_obs(history[0])
        sample_img = render_frame(first, agent_names=agent_names)
        self.screen_width, self.screen_height = sample_img.size

        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.title = title
        self.font = pygame.font.SysFont(None, 22)

    def _make_obs(self, entry: dict) -> dict:
        return {**entry, "_step": entry.get("step", 0)}

    def _surface(self, idx: int) -> pygame.Surface:
        agent_names = self.meta.get("agent_names", [])
        curr = self._make_obs(self.history[idx])
        prev = self._make_obs(self.history[idx - 1]) if idx > 0 else None
        img = render_frame(curr, prev_obs=prev, agent_names=agent_names)
        return pygame.image.fromstring(img.tobytes(), img.size, img.mode)

    def run(self) -> None:
        running = True
        while running:
            now = time.time()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_d:
                        self.step_idx = min(self.step_idx + 1, len(self.history) - 1)
                        self.paused = True
                    elif event.key == pygame.K_a:
                        self.step_idx = max(self.step_idx - 1, 0)
                        self.paused = True
                    elif event.key == pygame.K_e:
                        self.step_idx = len(self.history) - 1
                        self.paused = True
                    elif event.key == pygame.K_q:
                        self.step_idx = 0
                        self.paused = True

            if not self.paused and (now - self.last_tick) >= 1 / max(self.fps, 1):
                if self.step_idx < len(self.history) - 1:
                    self.step_idx += 1
                else:
                    self.paused = True
                self.last_tick = now

            self.screen.blit(self._surface(self.step_idx), (0, 0))

            help_txt = "SPACE play/pause | A/D step | Q/E jump | ESC quit"
            help_img = self.font.render(help_txt, True, (245, 245, 245))
            hh = help_img.get_height() + 8
            bg = pygame.Surface((self.screen_width, hh), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 140))
            self.screen.blit(bg, (0, self.screen_height - hh))
            self.screen.blit(help_img, (8, self.screen_height - hh + 4))

            step = self.history[self.step_idx].get("step", self.step_idx)
            pygame.display.set_caption(
                f"{self.title} | {'PAUSED' if self.paused else 'PLAYING'} | Step {step}/{self.history[-1].get('step', '?')}"
            )
            pygame.display.flip()
            self.clock.tick(30)

        pygame.quit()


def load_history(json_path: str) -> tuple[list, dict]:
    with open(json_path) as f:
        payload = json.load(f)
    history = payload.get("history", [])
    meta = payload.get("meta") or {}
    team_ids = payload.get("team_ids") or []
    if "agent_names" not in meta and team_ids:
        meta = {**meta, "agent_names": team_ids}
    return history, meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay a Bomberland match JSON")
    parser.add_argument("json_path", help="Path to match .json file")
    parser.add_argument("--fps",    type=int, default=8)
    parser.add_argument("--paused", action="store_true")
    args = parser.parse_args()

    history, meta = load_history(args.json_path)
    viewer = ReplayViewer(
        history=history,
        meta=meta,
        title=f"Replay — {Path(args.json_path).name}",
        fps=args.fps,
    )
    viewer.paused = args.paused
    viewer.run()
