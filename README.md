# RedQueen

Bomberland AI agent for the GDGoC-HCMUS AI Challenge 2026. Combines a ResNet CNN policy trained with PPO and a hard safety layer (action masking + BFS pathfinding) for robust play in 4-player free-for-all matches.

## Architecture

**Policy network** - 15-channel spatial input (13x13) + 7 auxiliary scalars -> ResNet CNN encoder -> MLP actor/critic heads. Exported to ONNX for inference; no PyTorch at runtime.

**Action masking** - hard-coded binary filter removes physically invalid or suicidal moves before the policy samples. Lives in `src/logic/action_masking.py`.

**Tie-break priority** - Kills > Boxes Destroyed > Items Collected > Bombs Placed.

## Setup

```bash
pip install torch numpy gymnasium stable-baselines3 sb3-contrib onnxruntime tqdm trueskill
```

## Training pipeline

```bash
# Phase 0 - generate behavioral cloning data (TacticalRuleAgent self-rollout)
python -m src.training.tactical_bc --generate --n-games 200

# Phase 2 - train BC model
python -m src.training.tactical_bc --train --epochs 20

# Phase 3 - PPO 7-stage curriculum
python -m src.training.ppo_trainer --curriculum --init-from checkpoints/tactical_bc_best_<ts>.pt

# Phase 4 - PFSP self-play
python -m src.training.ppo_trainer --self-play --init-from checkpoints/ppo_curriculum_best.pt
```

## Export and submission

```bash
# Export to ONNX
python -m src.utils.export_onnx --checkpoint checkpoints/<ckpt>.pt --output exports/model.onnx

# Build submission zip (agent.py must be at root, not in a subfolder)
cp exports/model.onnx agent/model.onnx
zip -j submission.zip agent/agent.py agent/model.onnx

# Validate
unzip -l submission.zip | grep agent.py   # must show "agent.py", not "*/agent.py"
```

## Local testing

```bash
# Headless match
python scripts/run_local_match.py --agent_paths agent/agent.py GeniusRuleAgent GeniusRuleAgent GeniusRuleAgent --num_episodes 20

# TrueSkill estimation vs random baselines
python scripts/estimate_rankings.py --agent_path agent/agent.py --num_matches 100

# TensorBoard
tensorboard --logdir logs/
```

## Directory structure

```
redqueen/
├── agent/              submission entry point + all baseline rule agents
├── src/
│   ├── logic/          action masking, BFS pathfinding
│   ├── models/         BomberCNNExtractor, BomberPolicyNet
│   ├── training/       PPO trainer, behavioral cloning, reward function
│   ├── utils/          feature extractor, ONNX export
│   └── wrappers/       Gymnasium single-agent wrapper
├── engine/             game engine (read-only)
├── scripts/            local match runner, visualizer, ranking estimator
├── checkpoints/        timestamped .pt checkpoints (gitignored)
├── exports/            ONNX model outputs (gitignored)
└── logs/               TensorBoard runs (gitignored)
```

## Curriculum stages

| Stage | Opponents | Threshold (avg_rank) |
|-------|-----------|---------------------|
| 0 | 3x Random | <= 0.8 |
| 1 | 3x Simple | <= 1.5 |
| 2 | 1x Smarter + 2x Simple | <= 1.5 |
| 3 | 2x Smarter + 1x Simple | <= 1.8 |
| 4 | 2x Tactical + 1x Smarter | <= 2.0 |
| 5 | 2x Trapper + 1x Tactical | <= 2.0 |
| 6 | 2x Genius + 1x Tactical | <= 2.0 |

Self-play (Phase 4) uses PFSP: blends recency decay with loss-rate weighting so the agent trains hardest against opponents it loses to.

## Observation schema

```
map:     (13, 13) int8    0=Grass 1=Wall 2=Box 3=ItemRadius 4=ItemCapacity
players: (4, 5)   int8    [x, y, alive, bombs_left, radius_bonus]
bombs:   (N, 4)   int8    [x, y, timer, owner_id]
```

Actions: `0=STOP  1=LEFT  2=RIGHT  3=UP  4=DOWN  5=PLACE_BOMB`