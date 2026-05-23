# CLAUDE.md — RedQueen AI Agent System
> Auto-updated before each conversation compact via PreCompact hook.
> Last manual update: 2026-05-24

---

## 1. Project Context

**Competition**: GDGoC-HCMUS AI Challenge 2026
**Game**: Bomberland — 4-player FFA on a 13×13 grid, 6 discrete actions, 500 steps max
**Goal**: Build the strongest Bomberland agent via a 5-phase training pipeline culminating in League Training

**Tech stack** (non-negotiable):
- Python 3.11
- PyTorch — model definition and training only
- NumPy — all grid/observation processing; vectorized operations required
- ONNX + onnxruntime — production inference (must hit < 100 ms/step)
- Gymnasium — environment wrapper
- stable-baselines3 — PPO backbone
- imitation — Behavioral Cloning helpers

---

## 2. Competition Live Rules *(Last updated: 2026-05-24)*

### Tie-break at Step 500

When a game reaches step 500 with multiple survivors, ranking among survivors is determined **in this order**:

```
1. Kills              (highest priority)
2. Boxes Destroyed
3. Items Collected
4. Bombs Placed       (lowest priority)
```

**Strategic consequence**: passive survival is no longer optimal. An agent with 1 kill and 0 boxes beats an agent with 0 kills and 10 boxes destroyed. Every surviving step must be oriented toward getting kills or destroying boxes.

### Submission Format (CRITICAL)

```
submission.zip
├── agent.py          ← MUST be at root, not inside any subfolder
├── model.onnx        (or .pth)
└── requirements.txt
```

If `agent.py` is inside a subfolder (`submission/agent.py`), the evaluation engine cannot find it and the team scores ~0.

---

## 3. Directory Structure

```
redqueen/
├── agent/                        # All agent code
│   ├── agent.py                  # ★ Submission entry point — ONNX inference, self-contained
│   ├── genius_rule_agent.py      # Strongest baseline — teacher for BC + curriculum opponent
│   ├── tactical_rule_agent.py    # Advanced rule agent — curriculum stage 3
│   ├── smarter_rule_agent.py     # Medium rule agent — local testing opponent
│   ├── simple_rule_agent.py      # Simple rule agent — curriculum stage 1
│   ├── box_farmer_agent.py       # Box-focused agent — local testing opponent
│   ├── random_agent.py           # Random agent — curriculum stage 0
│   └── __init__.py               # Exports all 6 baseline agents
│
├── src/                          # Core AI development
│   ├── logic/
│   │   ├── action_masking.py     # *** PROTECTED — see Golden Rules ***
│   │   └── pathfinding.py        # BFS / A* (vectorized NumPy)
│   ├── models/
│   │   └── policy_network.py     # BomberPolicyNet (actor-critic) + BomberCNNExtractor (SB3)
│   ├── training/
│   │   ├── history_parser.py     # Phase 0: Extract BC data from history_game/
│   │   ├── bc_trainer.py         # Phase 2: Behavioral Cloning (focal loss)
│   │   ├── ppo_trainer.py        # Phase 3-4: MaskablePPO + curriculum + self-play
│   │   └── reward_v2.py          # Reward function v2 (tie-break aware)
│   ├── wrappers/
│   │   └── bomberland_env.py     # Gymnasium single-agent wrapper for MaskablePPO
│   └── utils/
│       ├── feature_extractor.py  # obs dict → (spatial 15×13×13, aux 7) float32
│       └── export_onnx.py        # Export BomberPolicyNet → model.onnx
│
├── engine/                       # Game engine — READ-ONLY, never modify
│   └── game.py / map.py / bomb.py / player.py
│
├── scripts/                      # Helper scripts for local development & testing
│   ├── run_local_match.py        # Run headless or visual matches locally
│   ├── estimate_rankings.py      # Estimate TrueSkill rating vs random baselines
│   ├── replay_viewer.py          # Replay match JSON files (pygame + PIL)
│   ├── visualizer.py             # Live pygame viewer (called by run_local_match)
│   ├── agent_loader.py           # Dynamic agent.py loader (used by match scripts)
│   └── pre_compact_update.py     # Auto-updates CLAUDE.md before each compact
│
├── notebooks/
│   └── train_on_kaggle.ipynb     # Full training pipeline on Kaggle GPU
│
├── history_game/                 # 9,278 real competition match JSON files (gitignored subset)
│   └── YYYY-MM-DD/match_*.json   # See Section 5 for schema
│
├── checkpoints/                  # Timestamped .pt files (gitignored)
├── exports/                      # ONNX exports (gitignored)
├── data/                         # bc_dataset.npz (gitignored — can be 1GB+)
└── logs/                         # TensorBoard runs (gitignored)
```

---

## 4. Observation Schema (from `engine/game.py`)

```python
obs = {
    "map":     np.ndarray  # shape (13, 13), dtype int8
                           # 0=Grass, 1=Wall, 2=Box, 3=ItemRadius, 4=ItemCapacity
    "players": np.ndarray  # shape (4, 5), dtype int8
                           # columns: [x, y, alive, bombs_left, bomb_radius_bonus]
    "bombs":   np.ndarray  # shape (N, 4), dtype int8
                           # columns: [x, y, timer, owner_id]
}
```

**Engine constants** (from `engine/bomb.py`, `engine/player.py`):
- Bomb timer: 7 steps until explosion
- Max bomb radius bonus: 4 (radius = 1 + bonus, max radius = 5)
- Max bomb capacity: 5
- Starting positions: (1,1), (11,11), (1,11), (11,1) — corners, 1-indexed
- Movement: players CAN overlap each other, cannot enter Wall/Box/active Bomb tile

Action space: `{0: STOP, 1: LEFT, 2: RIGHT, 3: UP, 4: DOWN, 5: PLACE_BOMB}`

---

## 5. History Game Data

**Location**: `history_game/YYYY-MM-DD/match_*.json`
**Total files**: ~9,278 matches (as of 2026-05-23)
**Use**: Primary BC training data source. Do NOT run self-rollouts from scratch — use this first.

### Match File Schema

```python
{
  "seed": int,
  "team_ids": [str, str, str, str],      # UUID or "baseline_*" for baselines
  "meta": {"agent_names": [...]},
  "ranks": [int, int, int, int],          # rank[i] = placement of player i (0 = best)
  "survival_steps": [int, int, int, int], # steps survived by each player
  "runtime_stats": {                      # per-player: timeouts, errors, invalid_actions
    "0": {"timeouts": 0, "errors": 0, "invalid_actions": 0, "fallback_uses": 0},
    ...
  },
  "history": [                            # list of 501 frames (step 0 → 500)
    {
      "step": int,
      "actions": [int, int, int, int],    # None for step 0; action taken THIS step
      "alive": [bool, bool, bool, bool],
      "map": [[int, ...], ...],           # 13×13
      "players": [[x,y,alive,bombs_left,radius_bonus], ...],
      "bombs": [[x,y,timer,owner_id], ...]
    },
    ...
  ]
}
```

### Frame alignment (CRITICAL for BC)
- `history[t]["map/players/bombs"]` = **state AFTER** actions of step t were applied
- To get training pair: `obs_t = history[t]`, `action_t = history[t+1]["actions"][agent_id]`
- Correct: `obs = history[t]` → `label = history[t+1]["actions"][i]`

### BC Data Quality Filter
```python
# Keep only high-quality demonstrations
def is_quality_demo(match, agent_idx):
    return (
        match["ranks"][agent_idx] == 0          # winner
        and match["survival_steps"][agent_idx] >= 120  # survived long enough
        and sum(a == 5 for step in match["history"]
                for a in ([step["actions"][agent_idx]]
                          if step["actions"] else []))  >= 5  # placed bombs
    )
```

### Competitive Intelligence
- `baseline_*` agents: **0 wins** in 200 sampled games — usable as opponents, not as BC teachers
- Top teams by game count: `f9f492f0`, `cd455db7` — extract their winning demonstrations
- Average survival across all agents: ~175 steps (many early deaths from random agents)

---

## 6. Feature Engineering Specification (v2 — 4-player)

**Shape**: `(15, 13, 13)` spatial channels + `(7,)` auxiliary scalars

### Spatial Channels (15 total)

```
Ch 0   grass_mask          = (grid == 0).astype(float32)
Ch 1   wall_mask           = (grid == 1)
Ch 2   box_mask            = (grid == 2)
Ch 3   item_radius_mask    = (grid == 3)
Ch 4   item_capacity_mask  = (grid == 4)
Ch 5   my_position         = one-hot at (my_x, my_y) if alive
Ch 6   enemy1_position     = nearest alive enemy (by Manhattan distance)
Ch 7   enemy2_position     = second nearest alive enemy
Ch 8   enemy3_position     = third alive enemy (or zeros if fewer than 3 enemies)
Ch 9   bomb_timer          = timer/7.0 for each bomb cell (max if overlap)
Ch 10  bomb_owned          = 1.0 if bomb owned by me, else 0.0
Ch 11  bomb_enemy          = 1.0 if bomb owned by any enemy
Ch 12  danger_now          = blast tiles of bombs with timer <= 1
Ch 13  danger_soon         = blast tiles of bombs with timer <= 3 (includes danger_now)
Ch 14  danger_medium       = blast tiles of ALL active bombs
```

**Implementation constraint**: channels 12–14 must be computed via vectorized blast expansion — no Python loops over bombs. See `src/utils/feature_extractor.py`.

### Auxiliary Scalars (7 total)

```
0   my_bombs_left      / MAX_BOMB_CAPACITY (5)
1   my_radius_bonus    / (MAX_BOMB_RADIUS - 1) (4)
2   enemies_alive      / 3.0
3   step_normalized    = current_step / 500.0
4   boxes_remaining    = boxes_count / initial_boxes_count
5   my_kills_score     = kills / 3.0              # tracked externally, default 0
6   my_boxes_score     = boxes_destroyed / 20.0   # tracked externally, default 0
```

Scalars 5–6 require external stat tracking (not in raw obs). Use `0.0` during BC training, enable during PPO/League training when stats can be accumulated.

---

## 7. Reward Function (v2 — Tie-break Aware)

**File**: `agent/dqn_agent/reward.py` (for DQN legacy) and `src/training/reward_v2.py` (for PPO).

### Reward Table

```python
REWARD_V2 = {
    # === TERMINAL ===
    "win":              3.0,    # last agent standing
    "agent_death":     -2.0,    # own death (immediate return)

    # === TIE-BREAK STATS (in priority order) ===
    "kill_credit":      1.5,    # confirmed kill of enemy
    "box_destroyed":    0.4,    # per box confirmed destroyed by MY bomb
    "item_collected":   0.15,   # picking up radius or capacity item
    "bomb_placed":      0.003,  # placing a bomb (tie-break #4 padding)

    # === DANGER SHAPING ===
    "danger_evasion":   0.12,   # leaving blast zone (urgency multiplier ×1.5 if timer ≤ 3)
    "danger_enter":    -0.08,   # stepping INTO blast zone voluntarily
    "own_blast_loiter":-0.04,   # standing in own bomb blast (× (8-timer) urgency)

    # === MOVEMENT ===
    "survival_step":    0.005,  # alive bonus per step (encourage surviving)
    "standing_still":  -0.008,  # repeated STOP actions
    "time_penalty":    -0.003,  # small time cost to prevent infinite games

    # === ENEMY PRESSURE (reduced from v1 — dangerous in FFA) ===
    "approach_enemy":   0.006,  # × (prev_dist - curr_dist); was 0.02, reduced
}
```

### Key changes from v1

| Change | v1 | v2 | Reason |
|--------|----|----|--------|
| `kill_credit` | 1.0 (`enemy_death`) | **1.5** | Kill is tie-break #1 |
| `box_destroyed` | None | **+0.4** | Tie-break #2, confirmed (not proxy) |
| `plant_near_box` | +0.05 | **Removed** | Misleading proxy — plant != destroy |
| `bomb_placed` | None | **+0.003** | Tie-break #4, minimal |
| `approach_enemy` | 0.02 | **0.006** | Dangerous in 4-player FFA |
| `survival_step` | None | **+0.005/step** | Explicit survival signal |

### Box destruction detection
```python
# In compute_reward: compare grid between frames
prev_boxes = (prev_obs["map"] == 2)
curr_boxes = (curr_obs["map"] == 2)
destroyed_tiles = prev_boxes & ~curr_boxes  # boxes that disappeared
# Check if destroyed tile was in MY bomb's blast zone
```

---

## 8. Training Pipeline (v2)

```
Phase 0  History Mining     MỚI — Extract BC data from history_game/
  └─ Parser: src/training/history_parser.py
  └─ Filter: rank=0 AND survival≥120 AND bombs_placed≥5
  └─ Sources: baseline winners + top participant UUIDs (f9f492f0, cd455db7)
  └─ Estimated yield: 40,000–80,000 quality (obs, action) pairs

Phase 1  GeniusRuleAgent    Teacher for self-rollout demos + opponent in Phase 3 curriculum
  └─ Also run 10,000 self-rollout games to augment Phase 0 data

Phase 2  Behavioral Cloning Supervised from Phase 0+1 data
  └─ Feature: 15-channel (Section 6) + 7 aux scalars
  └─ Action Masking ACTIVE from this phase onward
  └─ Loss: Focal loss (γ=2) to handle class imbalance (PLACE_BOMB is rare ~15%)
  └─ Checkpoint: bc_{epoch}ep_{timestamp}.pt

Phase 3  PPO + Curriculum   RL fine-tune from BC init
  └─ Action Masking: ACTIVE, hard-coded
  └─ Reward: v2 (tie-break aware)
  └─ SubprocVecEnv: 8 parallel environments
  └─ Curriculum: Random(3) → Simple(3) → Genius(3) NoBomb → Genius(3) Full
  └─ Advance stage when: win_rate > 60% for 3 consecutive eval windows (100 games each)
  └─ Checkpoint: ppo_{step}_{timestamp}.pt

Phase 4  Continuous Self-Play  PPO vs rolling Past Agents pool
  └─ Snapshot every 50k steps → Past Agents
  └─ Sample weights: exponential decay (α=0.9^age) — favor recent agents
  └─ Pool size: keep latest 20 snapshots
  └─ Checkpoint: selfplay_{step}_{timestamp}.pt

Phase 5  League Training   Full league with 4-player matchmaking
  └─ Game composition: [Candidate] + [Main] + [Past_i] + [Past_j]
  └─ The Gauntlet: 200 games, win-rate ≥ 55% for promotion
  └─ ONNX export before each Gauntlet run
  └─ Checkpoint: league_{step}_{timestamp}.pt
```

### Hyperparameter Baseline (PPO)

```python
PPO_DEFAULTS = {
    "learning_rate":    3e-4,
    "n_steps":          2048,     # per env per rollout
    "batch_size":       256,
    "n_epochs":         10,
    "gamma":            0.995,    # high gamma — reward is delayed (bomb timer = 7)
    "gae_lambda":       0.95,
    "clip_range":       0.2,
    "ent_coef":         0.01,     # entropy: prevent premature convergence
    "vf_coef":          0.5,
    "max_grad_norm":    0.5,
    "n_envs":           8,        # SubprocVecEnv parallel
}
```

### Pipeline Summary Table

| Phase | Script | Key Flag | Checkpoint |
|-------|--------|----------|-----------|
| 0 | `history_parser.py` | `--extract` | `data/bc_dataset.npz` |
| 2 | `bc_trainer.py` | `--train` | `bc_{epoch}ep_{ts}.pt` |
| 3 | `ppo_trainer.py` | `--curriculum` | `ppo_{step}_{ts}.pt` |
| 4 | `ppo_trainer.py` | `--self-play` | `selfplay_{step}_{ts}.pt` |
| 5 | `league_trainer.py` | `--league` | `league_{step}_{ts}.pt` |

---

## 9. Strategic Priority Order

```
Priority 1: SURVIVE  (can't score if dead)
Priority 2: GET KILLS  (tie-break #1 — 1 kill beats 0 kills unconditionally)
Priority 3: DESTROY BOXES  (tie-break #2 — most likely differentiator in competitive play)
Priority 4: COLLECT ITEMS  (enables Priority 1, 2, 3 more effectively)
Priority 5: PLACE BOMBS  (tie-break #4 fallback)
```

**FFA-specific rules**:
- Do NOT approach an enemy fight if you are not the aggressor — let enemies kill each other
- If 2 enemies are in a fight, farm boxes elsewhere and claim the kill only if safe
- Late game (step > 350): bias heavily toward kills and boxes; survival is assumed

---

## 10. Strict Terminology

| Term | Definition |
|------|------------|
| **Action Masking** | Hard-logic binary filter that removes physically invalid or suicidal actions before the neural policy samples. Lives in `src/logic/action_masking.py`. |
| **Behavioral Cloning (BC)** | Supervised imitation learning (Phase 2) where the policy network is trained via focal cross-entropy loss to mimic demonstrations from `history_game/` and `GeniusRuleAgent`. |
| **Proximal Policy Optimization (PPO)** | On-policy RL algorithm (Phase 3–4) used to fine-tune the BC-initialised policy. Always combined with Action Masking. |
| **League Training** | Phase-5 training regime with three agent classes competing in a managed pool to produce increasingly robust policies. |
| **Continuous Self-Play** | Phase-4 training where the agent trains against a rolling pool of its own past checkpoints. |
| **Main Agent** | The current league champion. Serves as the active submission. |
| **Candidate Agent** | A new model currently in training. Must clear The Gauntlet to be promoted to Main. |
| **Past Agents** | Immutable frozen snapshots. Prevent mode collapse during self-play. |
| **The Gauntlet** | 200-game offline evaluation vs. Main + top-3 Past Agents. Win-rate ≥ 55% required. |
| **Nội Năng** | Internal Power — the neural network (policy + value networks). |
| **Ngoại Công** | External Power — hard rule-based layer (Action Masking, BFS Pathfinding, Danger Detection). |
| **Tie-break Stats** | The 4 criteria used when multiple players survive to step 500: Kills, Boxes Destroyed, Items Collected, Bombs Placed. |

---

## 11. Golden Rules

### Rule 1 — Action Masking is Sacred
> **Never modify `src/logic/action_masking.py` without an explicit instruction to do so.**

Any proposed change must be explicitly requested, then regression-tested against all edge cases before merging.

### Rule 2 — Checkpoint Immutability
> **Never overwrite an existing checkpoint. Always save with timestamped filename.**

Format: `{phase}_{description}_{timestamp}.pt`

### Rule 3 — The Gauntlet Before Promotion
> **No Candidate Agent becomes the Main Agent without passing The Gauntlet.**

Steps: 200 games → win-rate ≥ 55% → log to `logs/gauntlet/` with timestamp + model hashes.

### Rule 4 — Submission Zip Format
> **agent.py must be at the root of submission.zip, never inside a subfolder.**

Validate before every submission: `unzip -l submission.zip | grep agent.py` must show `agent.py` not `*/agent.py`.

---

## 12. Coding Standards

### Type Hints and Docstrings — Required

```python
def compute_blast_mask(grid: np.ndarray, bx: int, by: int, radius: int) -> np.ndarray:
    """Return boolean mask of tiles within blast range, blocked by walls/boxes."""
    ...
```

### Vectorized NumPy — Required (no nested for-loops over grid)

```python
# FORBIDDEN:
for x in range(13):
    for y in range(13):
        if grid[x, y] == 2: danger[x, y] = True

# REQUIRED:
danger = (grid == 2)
```

For bomb blast expansion: use `np.zeros((H,W))` accumulation with `np.ix_` or explicit vectorized ray-marching, not Python iteration over grid cells.

### Inference Path (production)

```
obs dict → feature_extractor.py (NumPy only) → onnxruntime.InferenceSession → action
```

Never import PyTorch inside `agent.py`. Export to ONNX after every training phase.

### ONNX Compatibility

Bomb array has variable length — must pad to fixed shape `(MAX_BOMBS=16, 4)` with a validity mask before ONNX export. Handle this in `feature_extractor.py`.

### No Silent Fallbacks

Do not use `try/except` that returns a default action. If agent crashes, surface the traceback.

---

## 13. FAQ for AI Assistants

**Q: Can I refactor `action_masking.py` while fixing an unrelated bug?**
No. See Rule 1.

**Q: The model isn't converging — should I lower entropy coefficient?**
Diagnose first: check reward curves, verify action distribution isn't collapsed, confirm Action Masking isn't blocking all actions in edge states. Tune hyperparameters last.

**Q: Should I use PyTorch in `agent.py`?**
No. ONNX Runtime only. See Inference Path.

**Q: The tie-break rewards feel conflicting with survival rewards — which wins?**
Survival is still Priority 1 (you can't score tie-break points if dead). But past step ~350 with no kills, the reward bias should shift toward risky aggressive plays. This is handled by the `step_normalized` aux feature allowing the network to learn this implicitly.

**Q: How should I extract stats (kills, boxes) for reward computation?**
Track them externally: compare `alive` flags between obs frames for kill credit; compare `(grid == 2)` masks between frames for box destruction. The engine's `Player.stats` dict is not in the obs output.

**Q: Can I merge two checkpoints?**
No. League Training is the mechanism for combining knowledge across model versions.

---

## 14. Session Update Log
*(Auto-appended by `scripts/pre_compact_update.py` before each compact)*

### 2026-05-24 — Initial full plan established
- Architecture defined: Hybrid Nội Năng (PPO neural net) + Ngoại Công (Action Masking + BFS)
- 5-phase pipeline: History Mining → BC → PPO+Curriculum → Self-Play → League Training
- Tie-break rule update received from BTC: Kills > Boxes > Items > Bombs
- history_game/ directory analyzed: 9,278 match files, avg survival 175 steps, baselines never win
- Reward function v2 designed: tie-break-aware, plant_near_box removed, confirmed box reward added
- Feature engineering v2: 15 spatial channels + 7 aux scalars, 4-player aware, danger maps
- Submission format clarified: agent.py must be at zip root
- encode_obs in legacy DQN identified as 2-player only — must rebuild for 4-player
- Pre-compact auto-update hook configured: scripts/pre_compact_update.py + .claude/settings.json

### 2026-05-24 01:05 — Auto-compact snapshot
  {
    "session_id": "172dfe2f-1195-4ec7-b47d-1751357f98e5",
    "transcript_path": "/Users/luuvanson/.claude/projects/-Users-luuvanson-Desktop-redqueen/172dfe2f-1195-4ec7-b47d-1751357f98e5.jsonl",
    "cwd": "/Users/luuvanson/Desktop/redqueen",
    "hook_event_name": "PreCompact",
    "trigger": "manual",
    "custom_instructions": ""
  }
