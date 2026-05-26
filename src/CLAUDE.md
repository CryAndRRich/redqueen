# src/ — Core AI Source Directory

## Protected / Read-Only Files

- **NEVER modify `src/logic/action_masking.py`** without an explicit instruction to do so (Golden Rule 1)
- **NEVER modify anything under `engine/`** — game engine is read-only

## Module Responsibilities

| File | Role |
|------|------|
| `logic/action_masking.py` | BFS-based safety filter — eliminates suicidal/invalid actions |
| `logic/pathfinding.py` | BFS/A* navigation helpers |
| `models/policy_network.py` | BomberPolicyNet (actor-critic) + BomberCNNExtractor for SB3 |
| `training/reward_v2.py` | Reward function (v3 logic despite filename) |
| `training/ppo_trainer.py` | Phase 3-4 MaskablePPO — 7-stage curriculum + self-play |
| `training/bc_trainer.py` | Phase 2 behavioral cloning (focal loss, γ=2) |
| `training/history_parser.py` | Phase 0 history mining from history_game/ |
| `utils/feature_extractor.py` | obs dict → (spatial 15×13×13, aux 7) float32 |
| `utils/export_onnx.py` | BomberPolicyNet → model.onnx + model.pt |
| `wrappers/bomberland_env.py` | Gymnasium single-agent wrapper for MaskablePPO |

## Invariants — Never Break

1. **Vectorized NumPy only** — no nested Python loops over the 13×13 grid
2. **Inference path**: `feature_extractor.py` → ONNX only — no PyTorch imports in `agent.py`
3. **Checkpoint naming**: `{phase}_{description}_{timestamp}.pt` — never overwrite an existing file
4. **BC weight transfer**: all 6 layer groups must be loaded — `spatial_enc`, `aux_enc`, `fusion`, `policy_net[0]`, `action_net`, `value_net`

## Architecture Note

`BomberPolicyNet.fusion = [Linear(3168→256), ReLU, Linear(256→128), ReLU]` — indices 0, 1, 2, 3.
