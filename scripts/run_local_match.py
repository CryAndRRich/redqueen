"""
Local match runner — pit agents against each other for headless testing.

Usage:
    # 4 random baselines
    python -m scripts.run_local_match

    # Your submission agent vs 3 genius bots
    python -m scripts.run_local_match --agent_paths agent/agent.py GeniusRuleAgent GeniusRuleAgent GeniusRuleAgent --num_episodes 20

    # Visual mode (requires pygame)
    python -m scripts.run_local_match --agent_paths agent/agent.py None None None --visualize true

Agent path formats:
    path/to/agent.py         — load Agent class from file
    path/to/folder/          — load Agent class from folder/agent.py
    RandomAgent              — built-in random baseline
    SimpleRuleAgent          — built-in simple rule agent
    SmarterRuleAgent         — built-in smarter rule agent
    GeniusRuleAgent          — built-in genius rule agent (strongest baseline)
    BoxFarmerAgent           — built-in box-focused agent
    TacticalRuleAgent        — built-in tactical rule agent
    None / random            — randomly picked built-in baseline
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from engine.game import BomberEnv
from agent import RandomAgent, SimpleRuleAgent, SmarterRuleAgent, GeniusRuleAgent, BoxFarmerAgent, TacticalRuleAgent
from scripts.agent_loader import load_agent_instance


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in {"true", "1", "yes", "y", "t"}:
        return True
    if v in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


_BASELINE_MAP = {
    "randomagent":     lambda i: RandomAgent(i),
    "simpleruleagent": lambda i: SimpleRuleAgent(i),
    "smarterruleagent":lambda i: SmarterRuleAgent(i),
    "geniusruleagent": lambda i: GeniusRuleAgent(i),
    "boxfarmeragent":  lambda i: BoxFarmerAgent(i),
    "tacticalruleagent": lambda i: TacticalRuleAgent(i),
}

_BASELINE_NAMES = [
    ("RandomAgent",      RandomAgent),
    ("SimpleRuleAgent",  SimpleRuleAgent),
    ("SmarterRuleAgent", SmarterRuleAgent),
    ("GeniusRuleAgent",  GeniusRuleAgent),
    ("BoxFarmerAgent",   BoxFarmerAgent),
    ("TacticalRuleAgent",TacticalRuleAgent),
]


def make_agents(agent_paths: list[str], seed: int | None = None) -> tuple[list, list[str]]:
    if seed is not None:
        random.seed(seed)

    agents: list = []
    names: list[str] = []

    for i, path in enumerate(agent_paths):
        key = path.strip().lower()

        if key in ("none", "random"):
            name, cls = random.choice(_BASELINE_NAMES)
            agents.append(cls(i))
            names.append(name)
            continue

        if key in _BASELINE_MAP:
            agents.append(_BASELINE_MAP[key](i))
            names.append(path)
            continue

        # Custom file/folder path
        p = Path(path)
        if p.is_dir():
            p = p / "agent.py"
        if not p.exists():
            raise FileNotFoundError(f"Agent file not found: {p}")
        agent = load_agent_instance(str(p), i)
        agents.append(agent)
        names.append(getattr(agent, "team_id", p.parent.name or p.stem))

    return agents, names


def run_match(
    agent_paths: list[str],
    num_episodes: int = 10,
    max_steps: int = 500,
    seed: int | None = None,
) -> list[dict]:
    env = BomberEnv(max_steps=max_steps, seed=seed)
    agents, names = make_agents(agent_paths, seed)
    info = [{"name": names[i], "wins": 0, "kills": 0, "boxes": 0} for i in range(4)]

    for episode in range(num_episodes):
        ep_seed = None if seed is None else seed + episode
        obs = env.reset(seed=ep_seed)
        done = False
        step = 0
        death_order: list[str] = []
        prev_alive = [bool(p[2]) for p in obs["players"]]

        while not done and step < max_steps:
            actions = []
            for i in range(4):
                try:
                    actions.append(int(agents[i].act(obs)))
                except Exception as e:
                    print(f"  Agent {names[i]} act() error: {e}")
                    actions.append(0)

            obs, terminated, truncated = env.step(actions)
            done = terminated or truncated
            step += 1

            alive_now = [bool(p[2]) for p in obs["players"]]
            for i in range(4):
                if prev_alive[i] and not alive_now[i]:
                    death_order.append(names[i])
            prev_alive = alive_now

        alive_final = [bool(obs["players"][i][2]) for i in range(4)]
        survivors = [i for i in range(4) if alive_final[i]]

        if len(survivors) == 1:
            winner = survivors[0]
            info[winner]["wins"] += 1
            print(f"  Ep {episode+1:3d}: {names[winner]} wins | deaths: {death_order}")
        else:
            print(f"  Ep {episode+1:3d}: Draw ({len(survivors)} alive) | deaths: {death_order}")

    print("\n=== Summary ===")
    for d in info:
        print(f"  {d['name']:30s}  wins: {d['wins']}/{num_episodes}")
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run local Bomberland matches")
    parser.add_argument(
        "--agent_paths", nargs="+",
        default=["None", "None", "None", "None"],
        help="4 agent paths/names. 'None' or 'random' = random baseline.",
    )
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--max_steps",    type=int, default=500)
    parser.add_argument("--seed",         type=int, default=None)
    parser.add_argument("--visualize",    type=str2bool, default=False,
                        help="Open pygame window (requires pygame installed)")
    parser.add_argument("--autoplay",     type=str2bool, default=True)
    args = parser.parse_args()

    if args.visualize:
        from scripts.visualizer import run_simple_viewer
        run_simple_viewer(
            agent_paths=args.agent_paths,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            autoplay=args.autoplay,
        )
    else:
        run_match(
            agent_paths=args.agent_paths,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            seed=args.seed,
        )
