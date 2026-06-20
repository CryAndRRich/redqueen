from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import trueskill
from engine.game import BomberEnv
from scripts.run_local_match import make_agents


def estimate_rankings(agent_path: str, num_matches: int = 100, max_steps: int = 500) -> None:
    print(f"Agent: {agent_path}")
    print(f"Matches: {num_matches}  Max steps: {max_steps}\n")

    ts_env = trueskill.TrueSkill(mu=100.0, sigma=33.333, draw_probability=0.1)
    agent_rating = ts_env.Rating()
    baseline_rating = ts_env.Rating()

    env = BomberEnv(max_steps=max_steps)
    wins = draws = 0
    total_rank = 0
    agent_name = "?"

    for i in range(num_matches):
        agents, names = make_agents([agent_path, "Random", "Random", "Random"], seed=None)
        agent_name = names[0]

        obs = env.reset()
        done = False
        step = 0
        prev_alive = [bool(p[2]) for p in obs["players"]]
        death_order: list[int] = []

        while not done and step < max_steps:
            actions = []
            for j in range(4):
                try:
                    actions.append(int(agents[j].act(obs)))
                except Exception:
                    actions.append(0)
            obs, terminated, truncated = env.step(actions)
            done = terminated or truncated
            step += 1
            alive_now = [bool(p[2]) for p in obs["players"]]
            for j in range(4):
                if prev_alive[j] and not alive_now[j]:
                    death_order.append(j)
            prev_alive = alive_now

        alive_final = [bool(obs["players"][j][2]) for j in range(4)]
        survivors = [j for j in range(4) if alive_final[j]]

        ranks = [0] * 4
        current_rank = 1
        for j in reversed(death_order):
            ranks[j] = current_rank
            current_rank += 1

        if len(survivors) == 1 and survivors[0] == 0:
            wins += 1
        elif len(survivors) > 1 and 0 in survivors:
            draws += 1
        total_rank += ranks[0]

        rating_groups = [(agent_rating,), (baseline_rating,), (baseline_rating,), (baseline_rating,)]
        new = ts_env.rate(rating_groups, ranks=ranks)
        agent_rating = new[0][0]

        score = agent_rating.mu - 3 * agent_rating.sigma
        print(
            f"Match {i+1:4d}/{num_matches} | rank {ranks[0]} | score {score:.2f}"
            f" | mu {agent_rating.mu:.2f} σ {agent_rating.sigma:.2f}",
            end="\r",
        )

    print(f"\n\n=== Results for {agent_name} ===")
    print(f"Win rate  : {wins/num_matches*100:.1f}%")
    print(f"Draw rate : {draws/num_matches*100:.1f}%")
    print(f"Avg rank  : {total_rank/num_matches:.2f}  (0=winner, 3=first to die)")
    score = agent_rating.mu - 3 * agent_rating.sigma
    print(f"TrueSkill : {score:.2f}  (mu={agent_rating.mu:.2f}, σ={agent_rating.sigma:.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate TrueSkill ranking vs random baselines")
    parser.add_argument("--agent_path",  required=True, help="Path to agent.py or agent folder")
    parser.add_argument("--num_matches", type=int, default=100)
    parser.add_argument("--max_steps",   type=int, default=500)
    args = parser.parse_args()
    estimate_rankings(args.agent_path, args.num_matches, args.max_steps)
