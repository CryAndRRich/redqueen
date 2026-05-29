# Project: RedQueen — GDGoC AI Challenge 2026

## Architecture
Hybrid **Ngoai Cong + Noi Nang**:
- **Noi Nang**: PPO neural net (BomberPolicyNet) — actor-critic, 15×13×13 spatial + 7 aux scalars
- **Ngoai Cong**: Hard rule-based layer — Action Masking (PROTECTED) + BFS Pathfinding + Danger Detection

## Training Pipeline (current)
```
Phase 0  TacticalAgent BC  src/training/tactical_bc.py
           └─ Replaces history_parser.py + bc_trainer.py
           └─ Generates BC data from GeniusRuleAgent + TacticalRuleAgent rollouts
           └─ Filter: rank=0 AND survival>=120 AND bombs_placed>=5

Phase 2  Behavioral Cloning  src/training/tactical_bc.py (--train flag)
           └─ Focal loss (gamma=2), action masking active, early stopping patience=5

Phase 3  PPO + 7-Stage Curriculum  src/training/ppo_trainer.py --curriculum
           └─ Init from tactical_bc checkpoint
           └─ Reward: v4 (tie-break aware + late-game multiplier)
           └─ STAGE_ENT_COEF = [0.08, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04]
           └─ STAGE_LR       = [3e-4, 1.5e-4, 1.2e-4, 1e-4, 8e-5, 8e-5, 5e-5]
           └─ MIN_STEPS_PER_STAGE = 100_000
           └─ Stage 0->1: reload BC (not Stage 0 best) + clear optimizer state

Phase 4  Continuous Self-Play  ppo_trainer.py --self-play
           └─ Rolling pool of past snapshots, exponential decay sampling

Phase 5  League Training  league_trainer.py --league
           └─ The Gauntlet: 200 games, win-rate >= 55% for promotion
```

## Reward Function
**Current version: v4** (file: src/training/reward.py)

Key v4 additions over v3:
- Late-game multiplier: reward scaling increases after step ~350 to bias toward kills/boxes
- v3 base: kill_credit=2.0, box_destroyed=0.4, item_collected=0.3, chain_reaction=0.3

## Key Files
| File | Role |
|------|------|
| src/training/tactical_bc.py | Primary BC pipeline (replaces history_parser + bc_trainer) |
| src/training/ppo_trainer.py | PPO curriculum + self-play |
| src/training/reward.py | Reward function v4 (current) |
| src/logic/action_masking.py | PROTECTED — never edit without explicit instruction |
| src/utils/feature_extractor.py | obs dict -> (15×13×13, 7) float32 |
| agent/agent.py | Submission entry point — ONNX inference only, no PyTorch |

## Dead Files (removed 2026-05-29)
- `src/training/bc_trainer.py` — superseded by tactical_bc.py
- `src/training/history_parser.py` — superseded by tactical_bc.py

## Golden Rules
1. action_masking.py is PROTECTED — no edits without explicit instruction
2. Checkpoints are immutable — always timestamped, never overwritten
3. No Candidate becomes Main without passing The Gauntlet (200 games, win-rate >= 55%)
4. agent.py must be at ZIP ROOT — validate: `unzip -l submission.zip | grep agent.py`
5. Check redqueen.ipynb after any src/ edit — keep notebook in sync with source changes

## Submission Format
```
submission.zip
├── agent.py     <- at root, not in subfolder
├── model.onnx
└── model.pt
```
Validate: `unzip -l submission.zip | grep agent.py` must show `agent.py` NOT `*/agent.py`
Do NOT include requirements.txt.
