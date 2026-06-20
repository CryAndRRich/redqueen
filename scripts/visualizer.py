from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from engine import BomberEnv
from agent import RandomAgent, SimpleRuleAgent, SmarterRuleAgent, TacticalRuleAgent, GeniusRuleAgent, BoxFarmerAgent
from scripts.agent_loader import load_agent_instance


class Viewer:
    PLAYER_COLORS = [(220, 50, 50), (50, 50, 220), (30, 150, 30), (200, 140, 0)]

    def __init__(self, width: int = 13, height: int = 13, cell_size: int = 42, fps: int = 8, panel_width: int = 200):
        import pygame
        self._pg = pygame
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.fps = fps
        self.panel_width = panel_width
        self.top_bar = 60
        self.grid_width = width * cell_size
        self.screen_width = self.grid_width + panel_width
        self.screen_height = height * cell_size + self.top_bar

        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Bomberland Viewer")
        self.clock = pygame.time.Clock()
        self.font_info = pygame.font.SysFont(None, 24)
        self.font_small = pygame.font.SysFont(None, 20)
        self.explosion_overlay = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        self.explosion_overlay.fill((255, 140, 0, 130))

    def draw_grid(self, grid: np.ndarray) -> None:
        pg = self._pg
        for row in range(self.height):
            for col in range(self.width):
                rect = pg.Rect(col * self.cell_size, row * self.cell_size + self.top_bar, self.cell_size, self.cell_size)
                ct = int(grid[row, col])
                if ct == 1:
                    pg.draw.rect(self.screen, (80, 80, 80), rect)
                    pg.draw.rect(self.screen, (40, 40, 40), rect, 2)
                elif ct == 2:
                    pg.draw.rect(self.screen, (139, 69, 19), rect)
                    pg.draw.rect(self.screen, (101, 67, 33), rect, 2)
                    pg.draw.line(self.screen, (101, 67, 33), rect.topleft, rect.bottomright, 2)
                    pg.draw.line(self.screen, (101, 67, 33), rect.topright, rect.bottomleft, 2)
                elif ct == 3:
                    pg.draw.rect(self.screen, (225, 225, 225), rect)
                    pg.draw.circle(self.screen, (255, 0, 0), rect.center, self.cell_size // 4)
                    self.screen.blit(self.font_small.render("R", True, (255, 255, 255)), (rect.centerx - 5, rect.centery - 8))
                elif ct == 4:
                    pg.draw.rect(self.screen, (225, 225, 225), rect)
                    pg.draw.circle(self.screen, (0, 0, 255), rect.center, self.cell_size // 4)
                    self.screen.blit(self.font_small.render("C", True, (255, 255, 255)), (rect.centerx - 5, rect.centery - 8))
                else:
                    pg.draw.rect(self.screen, (144, 238, 144), rect)
                    pg.draw.rect(self.screen, (120, 200, 120), rect, 1)

    def draw_players(self, players: np.ndarray) -> None:
        pg = self._pg
        for i, p in enumerate(players):
            if int(p[2]) != 1:
                continue
            center = (int(p[1]) * self.cell_size + self.cell_size // 2,
                      int(p[0]) * self.cell_size + self.top_bar + self.cell_size // 2)
            pg.draw.circle(self.screen, self.PLAYER_COLORS[i % 4], center, self.cell_size // 3)
            self.screen.blit(self.font_small.render(str(i), True, (255, 255, 255)), (center[0] - 5, center[1] - 8))
            self.screen.blit(self.font_small.render(f"B:{int(p[3])} R:{int(p[4])}", True, (0, 0, 0)),
                             (center[0] - 16, center[1] + 12))

    def draw_bombs(self, bombs: np.ndarray) -> None:
        pg = self._pg
        for b in bombs:
            if int(b[2]) <= 0:
                continue
            center = (int(b[1]) * self.cell_size + self.cell_size // 2,
                      int(b[0]) * self.cell_size + self.top_bar + self.cell_size // 2)
            pg.draw.circle(self.screen, (20, 20, 20), center, self.cell_size // 4)
            self.screen.blit(self.font_small.render(str(int(b[2])), True, (255, 255, 255)), (center[0] - 5, center[1] - 8))

    def draw_explosions(self, explosion_tiles: set) -> None:
        for row, col in explosion_tiles:
            px = col * self.cell_size
            py = row * self.cell_size + self.top_bar
            self.screen.blit(self.explosion_overlay, (px, py))
            self._pg.draw.circle(self.screen, (255, 220, 120),
                                 (px + self.cell_size // 2, py + self.cell_size // 2), self.cell_size // 6)

    def draw_sidebar(self, players: np.ndarray, agent_names: list[str]) -> None:
        pg = self._pg
        x0 = self.grid_width
        pg.draw.rect(self.screen, (52, 58, 64), (x0, 0, self.panel_width, self.screen_height))
        pg.draw.line(self.screen, (30, 30, 30), (x0, 0), (x0, self.screen_height), 2)
        self.screen.blit(self.font_info.render("Agents", True, (245, 245, 245)), (x0 + 10, self.top_bar + 8))
        y = self.top_bar + 40
        for i, p in enumerate(players):
            name = agent_names[i] if i < len(agent_names) else f"Agent {i}"
            alive = int(p[2]) == 1
            pg.draw.circle(self.screen, self.PLAYER_COLORS[i % 4], (x0 + 14, y + 8), 6)
            self.screen.blit(self.font_small.render(str(name)[:28], True, (240, 240, 240)), (x0 + 28, y))
            y += 22
            status_color = (120, 220, 140) if alive else (220, 100, 100)
            self.screen.blit(self.font_small.render("Alive" if alive else "Dead", True, status_color), (x0 + 10, y))
            y += 22
            self.screen.blit(self.font_small.render(f"Bombs: {int(p[3])}  +Radius: {int(p[4])}", True, (200, 200, 200)), (x0 + 10, y))
            y += 32

    def draw_header(self, ep: int, total_ep: int, step: int, total_steps: int, paused: bool) -> None:
        self._pg.draw.rect(self.screen, (30, 30, 30), (0, 0, self.screen_width, self.top_bar))
        text = f"Ep {ep+1}/{total_ep} | Step {step}/{max(total_steps-1,0)} | {'PAUSED' if paused else 'PLAYING'}"
        self.screen.blit(self.font_info.render(text, True, (245, 245, 245)), (10, 5))
        self.screen.blit(self.font_small.render("[A/D] Step  [W/S] Ep  [SPACE] Pause  [ESC] Quit", True, (210, 210, 210)), (10, 35))

    def render(self, obs: dict, prev_obs: dict | None, ep: int, total_ep: int,
               step: int, total_steps: int, paused: bool, agent_names: list[str]) -> None:
        self.screen.fill((245, 245, 245))
        self.draw_grid(obs["map"])
        self.draw_explosions(_explosion_tiles(prev_obs, obs))
        self.draw_players(obs["players"])
        self.draw_bombs(obs["bombs"])
        self.draw_sidebar(obs["players"], agent_names)
        self.draw_header(ep, total_ep, step, total_steps, paused)
        self._pg.display.flip()
        self.clock.tick(self.fps)

    def close(self) -> None:
        self._pg.quit()


def _explosion_tiles(prev_obs: dict | None, obs: dict) -> set:
    if prev_obs is None:
        return set()
    prev_bombs = np.asarray(prev_obs["bombs"]) if len(prev_obs["bombs"]) > 0 else np.zeros((0, 4), int)
    curr_bombs = np.asarray(obs["bombs"]) if len(obs["bombs"]) > 0 else np.zeros((0, 4), int)
    curr_positions = {(int(b[0]), int(b[1])) for b in curr_bombs}
    prev_players = np.asarray(prev_obs["players"])
    grid = np.asarray(prev_obs["map"])
    H, W = grid.shape
    tiles: set = set()
    for b in prev_bombs:
        bx, by, timer, oid = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        if timer > 1 and (bx, by) in curr_positions:
            continue
        radius = 1 + int(prev_players[oid][4]) if 0 <= oid < len(prev_players) else 2
        tiles.add((bx, by))
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            for r in range(1, radius + 1):
                tx, ty = bx + dx * r, by + dy * r
                if not (0 <= tx < H and 0 <= ty < W):
                    break
                cell = int(grid[tx, ty])
                if cell == 1:
                    break
                tiles.add((tx, ty))
                if cell == 2:
                    break
    return tiles


_BASELINE_MAP = {
    "randomagent":       lambda i: (RandomAgent(i),      "RandomAgent"),
    "simpleruleagent":   lambda i: (SimpleRuleAgent(i),  "SimpleRuleAgent"),
    "smarterruleagent":  lambda i: (SmarterRuleAgent(i), "SmarterRuleAgent"),
    "geniusruleagent":   lambda i: (GeniusRuleAgent(i),  "GeniusRuleAgent"),
    "boxfarmeragent":    lambda i: (BoxFarmerAgent(i),   "BoxFarmerAgent"),
    "tacticalruleagent": lambda i: (TacticalRuleAgent(i),"TacticalRuleAgent"),
}

_BASELINES_LIST = [
    ("RandomAgent",       RandomAgent),
    ("SimpleRuleAgent",   SimpleRuleAgent),
    ("SmarterRuleAgent",  SmarterRuleAgent),
    ("GeniusRuleAgent",   GeniusRuleAgent),
    ("BoxFarmerAgent",    BoxFarmerAgent),
    ("TacticalRuleAgent", TacticalRuleAgent),
]


def _make_agents(agent_paths: list[str], seed: int | None) -> tuple[list, list[str]]:
    if seed is not None:
        random.seed(seed)
    agents, names = [], []
    for i, path in enumerate(agent_paths):
        key = path.strip().lower()
        if key in ("none", "random"):
            name, cls = random.choice(_BASELINES_LIST)
            agents.append(cls(i)); names.append(name)
        elif key in _BASELINE_MAP:
            a, n = _BASELINE_MAP[key](i)
            agents.append(a); names.append(n)
        else:
            p = Path(path)
            if p.is_dir():
                p = p / "agent.py"
            a = load_agent_instance(str(p), i)
            agents.append(a)
            names.append(getattr(a, "team_id", p.parent.name or p.stem))
    return agents, names


def simulate_episodes(
    agent_paths: list[str],
    num_episodes: int = 10,
    max_steps: int = 500,
    seed: int | None = None,
) -> tuple[list[list[dict]], list[str]]:
    env = BomberEnv(max_steps=max_steps)
    agents, names = _make_agents(agent_paths, seed)
    episodes: list[list[dict]] = []

    def _clone(o: dict) -> dict:
        return {"map": np.array(o["map"], copy=True),
                "players": np.array(o["players"], copy=True),
                "bombs": np.array(o["bombs"], copy=True)}

    for ep in range(num_episodes):
        ep_seed = None if seed is None else seed + ep
        obs = env.reset(seed=ep_seed)
        done = False
        step = 0
        trajectory = [_clone(obs)]
        while not done and step < max_steps:
            actions = []
            for i in range(4):
                try:
                    actions.append(int(agents[i].act(obs)))
                except Exception:
                    actions.append(0)
            obs, terminated, truncated = env.step(actions)
            trajectory.append(_clone(obs))
            done = terminated or truncated
            step += 1
        episodes.append(trajectory)

    return episodes, names


def run_simple_viewer(
    agent_paths: list[str],
    num_episodes: int = 10,
    max_steps: int = 500,
    seed: int | None = None,
    autoplay: bool = True,
) -> None:
    episodes, agent_names = simulate_episodes(agent_paths, num_episodes, max_steps, seed)
    if not episodes:
        print("No episodes to display.")
        return

    import pygame
    first_obs = episodes[0][0]
    viewer = Viewer(width=first_obs["map"].shape[1], height=first_obs["map"].shape[0])
    print("Agents:", ", ".join(agent_names))
    print("Controls: A/D step | W/S episode | SPACE pause | ESC quit")

    ep_idx = 0
    step_idx = 0
    paused = not autoplay
    last_tick = time.time()
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
                    paused = not paused
                elif event.key == pygame.K_d:
                    step_idx = min(step_idx + 1, len(episodes[ep_idx]) - 1)
                    paused = True
                elif event.key == pygame.K_a:
                    step_idx = max(step_idx - 1, 0)
                    paused = True
                elif event.key == pygame.K_s:
                    ep_idx = min(ep_idx + 1, len(episodes) - 1)
                    step_idx = 0
                elif event.key == pygame.K_w:
                    ep_idx = max(ep_idx - 1, 0)
                    step_idx = 0

        if not paused and (now - last_tick) >= 1 / max(viewer.fps, 1):
            if step_idx < len(episodes[ep_idx]) - 1:
                step_idx += 1
            else:
                paused = True
            last_tick = now

        curr = episodes[ep_idx][step_idx]
        prev = episodes[ep_idx][step_idx - 1] if step_idx > 0 else None
        viewer.render(curr, prev, ep_idx, len(episodes), step_idx, len(episodes[ep_idx]), paused, agent_names)

    viewer.close()


def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y", "t"}:
        return True
    if s in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyGame viewer for Bomberland agents")
    parser.add_argument("--agent_paths", nargs="+", default=["None", "None", "None", "None"])
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--max_steps",    type=int, default=500)
    parser.add_argument("--seed",         type=int, default=None)
    parser.add_argument("--autoplay",     type=str2bool, default=True)
    args = parser.parse_args()
    run_simple_viewer(args.agent_paths, args.num_episodes, args.max_steps, args.seed, args.autoplay)
