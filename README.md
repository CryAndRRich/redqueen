# RedQueen — GDGoC-HCMUS AI Challenge 2026

> **Bomberland** · 4-player FFA · 13×13 grid · 6 actions · 500 steps

RedQueen is the AI agent system for **GDGoC-HCMUS AI Challenge 2026**. It fuses a **hard-logic safety layer** (Action Masking + BFS Pathfinding — *Ngoại Công*) with a **learned neural policy** trained through a **5-phase pipeline** culminating in **League Training** (*Nội Năng*).

![Bomberland](competition/match_20260520_112320_513617.gif)

---

## Architecture: Nội Năng & Ngoại Công

```
┌─────────────────────────────────────────────────────────────────┐
│                        HYBRID AGENT                             │
│                                                                 │
│  ┌──────────────────────┐     ┌──────────────────────────────┐  │
│  │    NGOẠI CÔNG        │     │         NỘI NĂNG             │  │
│  │  (External Power)    │     │      (Internal Power)        │  │
│  │                      │     │                              │  │
│  │  • Action Masking    │────▶│  • Policy Network (CNN+FC)   │  │
│  │  • BFS Pathfinding   │     │  • Value Network             │  │
│  │  • Danger Detection  │     │  • PPO optimizer             │  │
│  │  • Suicide Prevention│     │  • Behavioral Cloning init   │  │
│  │                      │     │                              │  │
│  │  Pure NumPy / Python │     │  PyTorch  →  ONNX Runtime    │  │
│  └──────────────────────┘     └──────────────────────────────┘  │
│                                                                 │
│   Hard rules filter INVALID actions → NN chooses among VALID   │
└─────────────────────────────────────────────────────────────────┘
```

**Ngoại Công** runs first and computes a **valid action mask** — blocking moves that would trigger suicide, walk into active blast zones, or place bombs with no escape route. **Nội Năng** then samples only from the safe action subset.

---

## Training Pipeline — 5 Phases

```
Phase 1 ──▶ Phase 2 ──▶ Phase 3 ──▶ Phase 4 ──▶ Phase 5
Heuristic   Behavior    PPO Fine-    Self-Play   League
Baseline    Cloning     tuning +     (Cont.)     Training
(Teacher)   (BC)        Action Mask
```

| Phase | Name | Method | Opponent | Output |
|-------|------|---------|----------|--------|
| 1 | **Heuristic Baseline** | Rule-based (`GeniusRuleAgent`) | — | Teacher policy |
| 2 | **Behavioral Cloning** | Supervised cross-entropy | Teacher demos | BC checkpoint |
| 3 | **PPO + Action Masking** | PPO from BC init | Curriculum: Static → Simple → Full | RL checkpoint |
| 4 | **Continuous Self-Play** | PPO vs past selves | Past Agents pool | Self-play checkpoint |
| 5 | **League Training** | PPO + prioritized matchmaking | Main + Candidate + Past Agents | League champion |

### Phase 5 — League Training Detail

```
                    ┌─────────────────┐
                    │   LEAGUE POOL   │
                    │                 │
          ┌─────────│  Past Agents    │──────────┐
          │         │  (frozen pool)  │          │
          ▼         └────────┬────────┘          ▼
   ┌─────────────┐           │           ┌─────────────────┐
   │ Main Agent  │◀──────────┘           │ Candidate Agent │
   │ (champion)  │──── The Gauntlet ────▶│  (challenger)   │
   └─────────────┘   (200 offline games) └─────────────────┘
          ▲                                       │
          └──── promote if win-rate ≥ 55% ────────┘
```

- **Main Agent**: Current best model, used as the active submission.
- **Candidate Agent**: New model under training; must clear The Gauntlet to be promoted.
- **Past Agents**: Frozen snapshots saved every N steps; prevent mode collapse.
- **The Gauntlet**: 200-game offline evaluation vs. Main + top-3 Past Agents. Win rate ≥ 55% required for promotion.

---

## Project Structure

```
redqueen/
├── agent/                       # Agent implementations
│   ├── agent.py                 # ← SUBMISSION ENTRY POINT
│   ├── genius_rule_agent.py     # Phase-1 teacher (Ngoại Công)
│   └── dqn_agent/               # Legacy DQN baseline
│
├── src/                         # Core AI development
│   ├── logic/
│   │   ├── action_masking.py    # Hard safety layer — read CLAUDE.md before touching
│   │   └── pathfinding.py       # BFS / A* helpers (vectorized)
│   ├── models/
│   │   ├── policy_network.py    # CNN + FC policy head
│   │   └── value_network.py
│   ├── training/
│   │   ├── bc_trainer.py        # Phase 2: Behavioral Cloning
│   │   ├── ppo_trainer.py       # Phase 3-4: PPO
│   │   └── league_trainer.py    # Phase 5: League Training
│   ├── wrappers/
│   │   └── bomberland_env.py    # Gymnasium wrapper around BomberEnv
│   └── utils/
│       ├── feature_extractor.py # Obs → tensor (vectorized NumPy, no nested loops)
│       └── replay_buffer.py
│
├── engine/                      # Bomberland game engine (do not modify)
│   ├── game.py                  # BomberEnv: step(), reset(), _get_obs()
│   ├── map.py / bomb.py / player.py
│
├── competition/                 # Competition infrastructure (organizers only)
├── scripts/                     # Utility scripts
├── plan/                        # Research papers & strategy docs
│
├── checkpoints/                 # Model weights — gitignored
├── exports/                     # ONNX exports — gitignored
└── logs/                        # TensorBoard logs — gitignored
```

---

## Environment Spec

| Property | Value |
|----------|-------|
| Grid | 13 × 13 |
| Players | 4 (FFA) |
| Actions | 6: STOP(0) LEFT(1) RIGHT(2) UP(3) DOWN(4) PLACE_BOMB(5) |
| Observation | `map` (13×13 int8), `players` (4×5 int8), `bombs` (N×4 int8) |
| Max steps | 500 (tie-break applied at termination) |
| Tile types | 0=Grass, 1=Wall, 2=Box, 3=ItemRadius, 4=ItemCapacity |
| Inference budget | **< 100 ms / step** |

---

## Quick Start

```bash
# 1. Clone & setup
git clone <repo-url> && cd redqueen
conda create -n redqueen python=3.11 -y && conda activate redqueen
pip install -r requirements.txt

# 2. Sanity check — run a heuristic game
python -m scripts.participant.run_local_game

# 3. Generate BC demonstrations from GeniusRuleAgent
python -m src.training.bc_trainer --collect --games 50000

# 4. Train BC policy
python -m src.training.bc_trainer --train --epochs 50

# 5. Fine-tune with PPO + Action Masking (Phase 3)
python -m src.training.ppo_trainer --init-from checkpoints/bc_best.pt

# 6. Launch League Training (Phase 5)
python -m src.training.league_trainer --main checkpoints/ppo_best.pt

# 7. Export to ONNX for fast inference
python -m src.utils.export_onnx --checkpoint checkpoints/league_champion.pt

# 8. Test submission agent locally
python -m scripts.participant.test_agent --agent agent/agent.py
```

---

## Coding Standards (Enforced)

- **Python 3.11** — Type hints on every function, Docstrings on every public API
- **Vectorized NumPy** for all grid operations — no nested `for` loops over the 13×13 grid
- **Inference target**: < 100 ms / step (use `onnxruntime` in production, not raw PyTorch)
- **Checkpoint discipline**: never overwrite existing weights; timestamped filenames always
- **Action Masking module** (`src/logic/action_masking.py`): do not modify without explicit instruction

---

## Competition Infrastructure (Organizers)

```bash
# Run registration server
sudo python3 -m competition.registration.app

# Run background evaluation worker (5 matches)
sudo python3 -m scripts.organizer.run_evaluation background

# Calibrate baselines (600 matches)
sudo python3 -m scripts.organizer.calibrate_baselines --matches 600 --workers 6

# Reset leaderboard
sudo python3 -m scripts.organizer.reset_to_baselines --db_path competition.db --yes

# Post daily highlights to Discord
sudo python3 -m scripts.organizer.post_daily_highlights

# Run Grand Finals (top-8 + best baseline, 50 matches per combo)
sudo python3 -m scripts.organizer.run_final_evaluation --matches_per_combo 50 --workers 6
```

---

## Community & Support

- Discord: [GDGoC AI Challenge](https://discord.gg/GqQJzuunBY)
- Issues: [GitHub Issues](https://github.com/VLTisME/Bomberland-GDGoC-AI-Challenge/issues)
- Agent Guide: [agent/README.md](agent/README.md)
- Competition Guide: [docs/COMPETITION_GUIDE.md](docs/COMPETITION_GUIDE.md)
